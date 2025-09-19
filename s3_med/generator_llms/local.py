import os
import requests
from transformers import AutoTokenizer

# Load matching tokenizer locally
MODEL = "Qwen/Qwen3-14B" 
# MODEL = "meta-llama/Meta-Llama-3-8B"
 
tokenizer = AutoTokenizer.from_pretrained(MODEL)


def generate_answer(context: str, prompt: str) -> str:
    """
    Generate an answer using the local LLM given context and prompt.
    
    Args:
        context: The context information
        prompt: The question or prompt to answer
        
    Returns:
        str: The generated answer
    """
    headers = {"Content-Type": "application/json"}
    
    context, prompt = format_qa_prompt(prompt, context)
    
    # Combine context and prompt
    full_prompt = context + "\n\n" + prompt
    
    # Prepare the payload for the API call
    payload = {
        "model": MODEL,
        "prompt": full_prompt,
        "temperature": 0.0,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0    
        }
    
    try:
        response = requests.post("http://localhost:8000/v1/completions", headers=headers, json=payload)
        response.raise_for_status()
        res = response.json()
        
        if "choices" not in res or not res["choices"]:
            raise ValueError("Invalid response from LLM")
            
        # Extract and return the generated text
        return res["choices"][0]["text"].strip()
        
    except Exception as e:
        print(f"Error generating answer: {e}")
        return ""

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
    Generate an answer using the local LLM given context and prompt.
    
    Args:
        context: The context information
        prompt: The question or prompt to answer
        
    Returns:
        str: The generated answer
    """
    headers = {"Content-Type": "application/json"}
    
    prompt = format_zero_shot_prompt(prompt)
    
    # Combine context and prompt
    full_prompt = prompt
    
    # Prepare the payload for the API call
    payload = {
        "model": MODEL,
        "prompt": full_prompt,
        "temperature": 0.0,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0    
        }
    
    try:
        response = requests.post("http://localhost:8000/v1/completions", headers=headers, json=payload)
        response.raise_for_status()
        res = response.json()
        
        if "choices" not in res or not res["choices"]:
            raise ValueError("Invalid response from LLM")
            
        # Extract and return the generated text
        return res["choices"][0]["text"].strip()
        
    except Exception as e:
        print(f"Error generating answer: {e}")
        return ""
    

def format_zero_shot_prompt(question: str) -> str:
    """
    Format a question and context into the specified zero-shot prompt format.
    
    Args:
        question: The question to be answered
        context: The context information
        
    Returns:
        str: The formatted zero-shot prompt
    """
    formatted_prompt = f"Question: {question}\n\nImportant: You MUST directly answer the question without any other text and thinking.\n\nYour Answer: "
    return formatted_prompt


def call_llm(prompt: str) -> str:
    # Prepare the payload for the API call
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 10,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0    
        }
    
    try:
        response = requests.post("http://localhost:8000/v1/completions", headers=headers, json=payload)
        response.raise_for_status()
        res = response.json()
        
        if "choices" not in res or not res["choices"]:
            raise ValueError("Invalid response from LLM")
            
        # Extract and return the generated text
        return res["choices"][0]["text"].strip()
        
    except Exception as e:
        print(f"Error generating answer: {e}")
        return ""

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
