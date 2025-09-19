import os
from generator_llms import qwen_api

# Load matching tokenizer locally
# MODEL = "Qwen/Qwen3-14B" 
# MODEL = "meta-llama/Meta-Llama-3-8B"
 

def generate_answer(context: str, prompt: str) -> str:
    """
    Generate an answer using the Qwen API given context and prompt.
    
    Args:
        context: The context information
        prompt: The question or prompt to answer
        
    Returns:
        str: The generated answer
    """
    context, prompt = format_qa_prompt(prompt, context)
    full_prompt = context + "\n\n" + prompt
    return qwen_api.generate_answer(full_prompt)

def format_qa_prompt(question: str, context: str) -> tuple[str, str]:
    """
    Format a question and context into the specified prompt format.
    
    Args:
        question: The question to be answered
        context: The context information
        
    Returns:
        tuple: (formatted_context, formatted_prompt)
    """
    formatted_context = f"""Use the following contexts (some might be irrelevant) on demand:\nContexts:
{context}"""
    
    formatted_prompt = f"Question: {question}\n\nImportant: You MUST directly answer the question without any other text and thinking.\n\nYour Answer: "
    
    return formatted_context, formatted_prompt

def generate_answer_zero_shot(prompt: str) -> str:
    """
    Generate an answer using the Qwen API without context.
    
    Args:
        prompt: The question or prompt to answer
        
    Returns:
        str: The generated answer
    """
    prompt = format_zero_shot_prompt(prompt)
    return qwen_api.generate_answer(prompt)

def format_zero_shot_prompt(question: str) -> str:
    """
    Format a question into the specified zero-shot prompt format.
    
    Args:
        question: The question to be answered
        
    Returns:
        str: The formatted zero-shot prompt
    """
    formatted_prompt = f"Question: {question}\n\nImportant: You MUST directly answer the question without any other text and thinking.\n\nYour Answer: "
    return formatted_prompt

def call_llm(prompt: str) -> str:
    """
    Call the Qwen API with a given prompt.
    
    Args:
        prompt: The prompt to send to the API
        
    Returns:
        str: The generated response
    """
    return qwen_api.generate_answer(prompt)

def check_if_response_is_correct_llm(response: str, gold_answers: list[str]) -> bool:
    """
    Check if the generated answer is correct by comparing it to the gold answers.
    
    Args:
        response: The response from the LLM
        gold_answers: The gold answers
        
    Returns:
        bool: True if the generated answer is correct, False otherwise
    """
    prompt = f"Please check if any of the golden answers is contained in the following response: {response}\n\nGolden answers: {str(gold_answers)}"
    prompt += "\nPlease directly answer with 'yes' or 'no'. \n\nYour Answer: "
    
    yes_or_no = call_llm(prompt)
    
    if "yes" in yes_or_no.lower():
        return True
    elif "no" in yes_or_no.lower():
        return False
    else:
        yes_or_no = call_llm(prompt)
        if "yes" in yes_or_no.lower():
            return True
        elif "no" in yes_or_no.lower():
            return False
        else:
            return False
