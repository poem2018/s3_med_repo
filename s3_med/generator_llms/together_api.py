from together import Together
import math

# Load API key
with open('generator_llms/together_api_key.key', 'r') as f:
    API_KEY = f.read().strip()
client = Together(api_key=API_KEY)

def compute_generation_perplexity(question: str, answer: str) -> float:
    """
    Compute perplexity of the answer given the question (+ context).
    """
    full_prompt = question.strip() + "\nAnswer:"  # Add 'Answer:' if you want it natural
    full_text = full_prompt + " " + answer.strip()

    # Now ask Together to get token logprobs
    completion = client.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",  # NOT .chat
        prompt=full_text,
        max_tokens=0,  # Do not generate
        logprobs=True,
    )

    tokens = completion.choices[0].logprobs.tokens
    token_logprobs = completion.choices[0].logprobs.token_logprobs

    # Find where the answer tokens start
    prompt_len = len(client.tokenize(full_prompt))  # You may need Together's tokenizer API here
    answer_logprobs = token_logprobs[prompt_len:]  # Only score the answer part

    if len(answer_logprobs) == 0:
        raise ValueError("No answer tokens found for scoring.")

    avg_logprob = sum(answer_logprobs) / len(answer_logprobs)
    perplexity = math.exp(-avg_logprob)

    return perplexity

# --- Zero-shot (no retrieval) setting ---
question_zero_shot = "Who discovered the circulation of blood in the human body?"
answer_zero_shot = "I believe it was discovered by William Harvey."

ppl_zero_shot = compute_generation_perplexity(question_zero_shot, answer_zero_shot)
print(f"Zero-shot Perplexity: {ppl_zero_shot}")

# --- Retrieval-augmented setting ---
retrieved_context = (
    "Context: William Harvey (1578–1657) was an English physician who first described "
    "the complete circulation of blood in the human body, demonstrating that it was pumped by the heart.\n"
)
question_retrieval = (
    retrieved_context + 
    "Question: Who discovered the circulation of blood in the human body?"
)
answer_retrieval = "William Harvey discovered the circulation of blood in the human body."

ppl_retrieval = compute_generation_perplexity(question_retrieval, answer_retrieval)
print(f"Retrieval-augmented Perplexity: {ppl_retrieval}")
