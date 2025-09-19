import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
from verl import DataProto
import requests
import json

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool = False
    search_url: str = None
    topk: int = 3
    include_information: bool = False  # Whether to include search results in feedback
    feedback_prompt_file: str = "prompts/feedback_prompt.txt"  # Path to feedback prompt template
    answer_prompt_file: str = "prompts/answer_prompt.txt"  # Path to answer prompt template
    zero_shot_prompt_file: str = "prompts/zero_shot_prompt.txt"  # Path to zero-shot prompt template
    zero_shot_store_file: str = "data/nq_hotpotqa_zeroshot_claude3/zeroshot_answers.json"
    generator_llm: str = "claude-3.5"  # Which LLM to use for generation ("claude-3.5", "claude-3   ", "gpt-3.5", "gpt-4o-mini", "gpt-4o")

class LLMGenerationManager:
    """
    Search-C1: A search copilot that can be trained separately from the generator LLM.
    The generator LLM is treated as part of the environment.

    This implementation maintains output compatibility with Search-R1 for training,
    but conceptually separates the search copilot from the generator LLM.
    """
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,  # Worker group for the search copilot (to be trained)
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        self.timing_raw = {}
        
        # Load prompt templates for the generator LLM
        self.feedback_prompt = self._load_prompt(config.feedback_prompt_file)
        self.answer_prompt = self._load_prompt(config.answer_prompt_file)
        self.zero_shot_prompt = self._load_prompt(config.zero_shot_prompt_file)
        self.zeroshot_answers = self._load_zeroshot_answers(config.zero_shot_store_file)
        self.save_zeroshot_flag = False
        # Initialize tensor helper for handling tensors
        from .tensor_helper import TensorHelper, TensorConfig
        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))
        
        # Import external LLM modules
        try:
            from generator_llms import get_claude_response, gpt_chat_35_msg, gpt_chat_4omini, gpt_chat_4o
            self.get_claude_response = get_claude_response
            self.gpt_chat_35_msg = gpt_chat_35_msg
            self.gpt_chat_4omini = gpt_chat_4omini
            self.gpt_chat_4o = gpt_chat_4o
        except ImportError:
            print("Warning: generator_llms module not found. Using mock LLM responses.")
            raise ImportError("generator_llms module not found.")

    def _load_zeroshot_answers(self, filename):
        """Load zeroshot answers from file."""
        try:
            with open(filename, 'r') as file:
                return json.load(file)
        except (FileNotFoundError, IOError):
            print(f"Zeroshot answers file {filename} not found.")
            return {}
        
    def _save_zeroshot_answers(self, filename):
        """Save zeroshot answers to file."""
        if self.save_zeroshot_flag:
            with open(filename, 'w') as file:
                json.dump(self.zeroshot_answers, file, indent=2)

    def _load_prompt(self, filename):
        """Load prompt template from file."""
        try:
            with open(filename, 'r') as file:
                return file.read().strip()
        except (FileNotFoundError, IOError):
            # Return a default prompt if file not found
            raise ValueError(f"Prompt file {filename} not found.")

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> Tuple[torch.Tensor, List[str], List[str]]:
        """Process responses to extract search queries."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        # Ensure responses end with </query> tag if it exists
        responses_str = [resp.split('</query>')[0] + '</query>'
                if '</query>' in resp 
                else resp
                for resp in responses_str]

        # Extract query information
        queries = []
        for resp in responses_str:
            query_match = re.search(r'<query>(.*?)</query>', resp, re.DOTALL)
            if query_match:
                query_text = query_match.group(1).strip()
                # Extract Query without End flag
                query = re.search(r'Query:\s*(.*?)(?=End:|$)', query_text, re.DOTALL)
                
                if query:
                    queries.append(query.group(1).strip())
                else:
                    queries.append("")
            else:
                queries.append("")
                    
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str, queries

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> DataProto:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids is not None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
        Wrapper for generation that handles multi-GPU padding requirements.
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> DataProto:
        """
        Run the Search-C1 loop with environment-based feedback and rewards.
        
        In Search-C1:
        1. Search copilot generates queries with End=True/False
        2. If End=False, search engine retrieves documents and external LLM provides feedback
        3. If End=True, external LLM would generate the final answer
        """
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
        
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch
        
        # Reset conversation histories for this batch
        self.conversation_histories = [""] * len(active_mask)
        
        # Extract the original questions from the initial inputs
        self.original_questions = [""] * len(active_mask)
        initial_inputs_str = self.tokenizer.batch_decode(
            initial_input_ids, 
            skip_special_tokens=True
        )
        
        # Extract questions from the initial inputs
        for i, input_text in enumerate(initial_inputs_str):
            question_matches = re.findall(r'<question>(.*?)</question>', input_text, re.DOTALL)
            if question_matches:
                # Use the last match of <question>...</question>
                self.original_questions[i] = question_matches[-1].strip()
            else:
                print(f"No <question>...</question> tags found in the initial input {input_text}")

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
                
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # Generate with active sequences
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            # Process outputs 
            meta_info = gen_output.meta_info            
            responses_ids, responses_str, queries = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
            
            # Execute search and get feedback from the environment
            next_obs, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask
            )
            
            # Update active sequences
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            
            # Process observations (search results + feedback)
            next_obs_ids = self._process_next_obs(next_obs)
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
            
        # Final LLM rollout
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            # Process outputs - keeping exact compatibility with Search-R1
            meta_info = gen_output.meta_info            
            responses_ids, responses_str, queries = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
            
            # Execute final predictions (without doing search)
            next_obs, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, do_search=False
            )

            # Update stats
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            
            next_obs_ids = self._process_next_obs(next_obs)

            # Update right side
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
        
        # Store metadata for reward computation
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        final_output = self._compose_final_output(original_left_side, original_right_side, meta_info)
        return final_output

    def _example_level_pad(self, tensor, str_list, queries, end_flags, active_mask):
        """Pad tensors and lists back to full batch size - similar to original Search-R1."""
        if active_mask.all():
            return tensor, str_list, queries, end_flags
            
        full_size = active_mask.shape[0]
        active_indices = torch.where(active_mask)[0]
        
        # Pad tensor
        padded_tensor = torch.zeros(
            (full_size, tensor.shape[1]), 
            dtype=tensor.dtype, 
            device=tensor.device
        )
        padded_tensor[active_indices] = tensor
        
        # Pad string list
        padded_str = [""] * full_size
        for i, idx in enumerate(active_indices):
            padded_str[idx.item()] = str_list[i]
            
        # Pad queries
        padded_queries = [""] * full_size
        for i, idx in enumerate(active_indices):
            padded_queries[idx.item()] = queries[i]
            
        # Pad end flags
        padded_end_flags = [False] * full_size
        for i, idx in enumerate(active_indices):
            padded_end_flags[idx.item()] = end_flags[i]
            
        return padded_tensor, padded_str, padded_queries, padded_end_flags

    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=True) -> Tuple[List[str], List[bool], List[int], List[int]]:
        """
        Execute predictions and generate external LLM feedback for Search-C1.
        """
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_search = [], [], [], []
        
        # Track conversation history for each active example
        if not hasattr(self, 'conversation_histories'):
            self.conversation_histories = [""] * len(active_mask)
            # self.original_questions = [""] * len(active_mask)
            
        # # Get original questions if not already set
        # if all(not q for q in self.original_questions):
        #     for i, active in enumerate(active_mask):
        #         if active:
        #             # Extract original question from the batch
        #             self.original_questions[i] = self._extract_original_question(predictions[i])
        
        search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
        if do_search and search_queries:
            search_results = self.batch_search(search_queries)
            search_counter = 0
        else:
            search_results = [''] * len(search_queries)
            search_counter = 0

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            if not active and do_search:
                next_obs.append('')
                dones.append(True)
                valid_action.append(0)
                is_search.append(0)
            elif not active and not do_search:
                final_answer, zero_shot_answer = self._generate_final_answer(
                    self.original_questions[i], 
                    self.conversation_histories[i]
                )
                
                # Store in conversation history
                self.conversation_histories[i] += f"\nFinal answer: {final_answer}\n"
                
                # CHANGE: Include the answer in the observation
                next_obs.append(f'\n\n<answer>{final_answer}</answer>\n\n<zeroshot_answer>{zero_shot_answer}</zeroshot_answer>\n\n')
                
                dones.append(True)
                valid_action.append(1)
                is_search.append(0)
            else:
                if action == 'answer' or not do_search:
                    final_answer, zero_shot_answer = self._generate_final_answer(
                        self.original_questions[i], 
                        self.conversation_histories[i]
                    )
                    
                    # Store in conversation history
                    self.conversation_histories[i] += f"\nFinal answer: {final_answer}\n"
                    
                    # CHANGE: Include the answer in the observation
                    next_obs.append(f'\n\n<answer>{final_answer}</answer>\n\n<zeroshot_answer>{zero_shot_answer}</zeroshot_answer>\n\n')
                    dones.append(True)
                    valid_action.append(1)
                    is_search.append(0)
                    
                elif action == 'search':
                    # Execute search and provide feedback
                    if do_search:
                        search_result = search_results[search_counter]
                        search_counter += 1
                        
                        # Generate feedback using the external LLM
                        feedback = self._generate_feedback(
                            original_question=self.original_questions[i],
                            query=contents[i], 
                            search_result=search_result,
                            conversation_history=self.conversation_histories[i]
                        )
                        
                        should_stop = self._check_feedback_for_stop(feedback)
                        if should_stop:
                            feedback = "Stop Search: Yes"
                        else:
                            feedback = "Stop Search: No"
                        
                        # Update conversation history
                        self.conversation_histories[i] += f"\nQuery: {contents[i]}\nSearch results: {search_result}\nFeedback: {feedback}"
                        
                        # CHANGE: Always include information tags
                        next_obs.append(f'\n\n<information>{search_result.strip()}</information>\n\n<feedback>{feedback}</feedback>\n\n')
                    else:
                        feedback = "No search was performed as search is disabled."
                        next_obs.append(f'\n\n<information>Search disabled.</information>\n\n<feedback>{feedback}</feedback>\n\n')
                    
                    # CHANGE: Check feedback for stop search signal
                    dones.append(should_stop)  # End if feedback indicates to stop
                    valid_action.append(1)
                    is_search.append(1)
                else:
                    # Invalid action
                    feedback = "My previous action is invalid. I should put my search query between <query> and </query> tags. Let me try again."
                    next_obs.append(f'\n<feedback>{feedback}</feedback>\n')
                    
                    # Update conversation history
                    self.conversation_histories[i] += f"\nInvalid action\nFeedback: {feedback}"
                    
                    dones.append(False)
                    valid_action.append(0)
                    is_search.append(0)
            
        return next_obs, dones, valid_action, is_search
    
    
    def _generate_feedback(self, original_question: str, query: str, search_result: str, conversation_history: str = "") -> str:
        """
        Generate feedback based on search results using an external LLM.
        
        Args:
            query: The search query
            search_result: The search results
            conversation_history: The conversation history between search copilot and generator LLM
            
        Returns:
            Generated feedback
        """
        # Format the prompt with the actual values
        prompt = self.feedback_prompt.format(
            original_question=original_question,
            query=query,
            search_results=search_result,
            conversation_history=conversation_history
        )
        
        max_length = 30
        
        # Call the appropriate external LLM based on configuration
        if self.config.generator_llm == "claude-3.5":
            feedback = self.get_claude_response(prompt, llm='sonnet', max_tokens=max_length)
        elif self.config.generator_llm == "claude-3":
            feedback = self.get_claude_response(prompt, llm='haiku', max_tokens=max_length)
        elif self.config.generator_llm == "gpt-3.5":
            feedback = self.gpt_chat_35_msg(prompt, max_tokens=max_length)
        elif self.config.generator_llm == "gpt-4o-mini":
            feedback = self.gpt_chat_4omini(prompt, max_tokens=max_length)
        elif self.config.generator_llm == "gpt-4o":
            feedback = self.gpt_chat_4o(prompt, max_tokens=max_length)
        else:
            # Default to simple feedback if no valid LLM is specified
            print("[Warning] No valid LLM is specified. Using simple feedback.")
            feedback = self._generate_simple_feedback(query, search_result)
            
        return feedback
    
    def _generate_simple_feedback(self, query: str, search_result: str) -> str:
        """
        Generate simple feedback for testing without external LLM calls.
        This is a fallback for when external LLMs are not available.
        """
        # Check if search result is empty
        if not search_result or search_result.strip() == "":
            return "Your search returned no results. Try to reformulate your query with more specific terms."
        
        # Check if the search result is substantive
        if len(search_result.split()) < 10:
            return "The search results are very limited. Consider trying different search terms or being more specific."
        
        # Default positive feedback
        return "The search results contain relevant information. Consider refining your query further if you need more specific details."
    
    def _check_feedback_for_stop(self, feedback: str) -> bool:
        """
        Check if the feedback indicates that we should stop searching.
        
        Args:
            feedback: The feedback from the generator LLM
            
        Returns:
            Boolean indicating whether to stop searching
        """
        # Look for explicit stop indicators in the feedback
        stop_patterns = [
            r'Stop Search:\s*Yes',
            r'stop searching',
            r'no need to search further',
            r'sufficient information',
            r'already have all the information needed',
            r'already have the answer',
            r'can answer the question',
            r'enough information to answer'
        ]
        
        for pattern in stop_patterns:
            if re.search(pattern, feedback, re.IGNORECASE):
                return True
                
        return False
    
    def _generate_final_answer(self, original_question: str, conversation_history: str = "") -> str:
        """
        Generate final answer based on search results using an external LLM.
        
        Args:
            original_question: The original question
            conversation_history: The conversation history between search copilot and generator LLM
            
        Returns:
            answer string
        """
        # Format the prompt with the actual values
        prompt = self.answer_prompt.format(
            question=original_question,
            conversation_history=conversation_history
        )
        
        prompt_zero_shot = self.zero_shot_prompt.format(
            question=original_question,
        )
        
        max_length = 30
        
        if original_question not in self.zeroshot_answers:
            self.save_zeroshot_flag = True
        
        # Call the appropriate external LLM based on configuration
        if self.config.generator_llm == "claude-3.5":
            answer = self.get_claude_response(prompt, llm='sonnet', max_tokens=max_length)
            answer_zero_shot = self.get_claude_response(prompt_zero_shot, llm='sonnet', max_tokens=max_length) if original_question not in self.zeroshot_answers else self.zeroshot_answers[original_question]
        elif self.config.generator_llm == "claude-3":
            answer = self.get_claude_response(prompt, llm='haiku', max_tokens=max_length)
            answer_zero_shot = self.get_claude_response(prompt_zero_shot, llm='haiku', max_tokens=max_length) if original_question not in self.zeroshot_answers else self.zeroshot_answers[original_question]
        elif self.config.generator_llm == "gpt-3.5":
            answer = self.gpt_chat_35_msg(prompt, max_tokens=max_length)
            answer_zero_shot = self.gpt_chat_35_msg(prompt_zero_shot, max_tokens=max_length) if original_question not in self.zeroshot_answers else self.zeroshot_answers[original_question]
        elif self.config.generator_llm == "gpt-4o-mini":
            answer = self.gpt_chat_4omini(prompt, max_tokens=max_length)
            answer_zero_shot = self.gpt_chat_4omini(prompt_zero_shot, max_tokens=max_length) if original_question not in self.zeroshot_answers else self.zeroshot_answers[original_question]
        elif self.config.generator_llm == "gpt-4o":
            answer = self.gpt_chat_4o(prompt, max_tokens=max_length)
            answer_zero_shot = self.gpt_chat_4o(prompt_zero_shot, max_tokens=max_length) if original_question not in self.zeroshot_answers else self.zeroshot_answers[original_question]
        else:
            raise ValueError(f"Invalid generator LLM: {self.config.generator_llm}")

        # Extract content between <answer> and </answer> tags
        # match = re.search(r"<answer>(.*?)</answer>", answer, re.DOTALL)
        # if match:
        #     final_answer = match.group(1).strip()
        # else:
        #     print("No valid <answer>...</answer> tags found in the LLM response, use the whole response as the answer.")
        final_answer = answer.strip()
        final_answer_zero_shot = answer_zero_shot.strip()
        self.zeroshot_answers[original_question] = final_answer_zero_shot
        
        return final_answer, final_answer_zero_shot
        
    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[str], List[str]]:
        """
        Process predictions to extract actions and content.
        """
        actions = []
        contents = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                # Extract search queries
                answer_match = re.search(r'<answer>(.*?)</answer>', prediction, re.DOTALL)
                query_match = re.search(r'<query>(.*?)</query>', prediction, re.DOTALL)
                
                if query_match:
                    query_text = query_match.group(1).strip()
                    
                    # Check if the query is in JSON format
                    try:
                        # Try to parse as JSON
                        import json
                        json_data = json.loads(query_text)
                        if 'query' in json_data:
                            content = json_data['query']
                            if type(content) == list:
                                content = content[0]
                            elif type(content) == str:
                                content = content
                            else:
                                content = ''
                                
                            action = "search"
                        else:
                            content = ''
                            action = None
                    except json.JSONDecodeError:
                        # Fallback to regex pattern for non-JSON format
                        content = ''
                        action = None
                        
                else:
                    content = ''
                    action = None
            else:
                content = ''
                action = None
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> DataProto:
        """Compose final output for the search copilot."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and info mask
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def batch_search(self, queries: List[str] = None) -> List[str]:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        try:
            results = self._batch_search(queries)['result']
        except Exception as e:
            raise Exception(f"Error in batch_search: {e}, queries: {queries}")
            
        
        return [self._passages2string(result) for result in results]

    def _batch_search(self, queries):
        """Call the search API."""
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }
        
        return requests.post(self.config.search_url, json=payload).json()

    def _passages2string(self, retrieval_result):
        """Format retrieval results into a string."""
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference