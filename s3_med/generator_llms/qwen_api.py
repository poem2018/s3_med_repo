import os
from openai import OpenAI
import time
with open("generator_llms/ali_api.key", "r") as f:
    api_key = f.read().strip()

client = OpenAI(
    # If environment variables are not configured, replace the following line with: api_key="sk-xxx",
    api_key=api_key,
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)


def generate_answer(prompt: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="qwen2.5-14b-instruct", # This example uses qwen-plus. You can change the model name as needed. Model list: https://www.alibabacloud.com/help/en/model-studio/getting-started/models
                messages=[
                    {'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': prompt}],
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error generating answer (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                raise Exception(f"Failed to generate answer after {max_retries} attempts: {str(e)}")
