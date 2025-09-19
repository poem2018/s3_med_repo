with open('generator_llms/deepinfra.key', 'r') as f:
    API_KEY = f.read().strip()
    
    
import requests
import json
import os

def call_deepinfra_model_with_logprobs_streaming(
    model_name,
    prompt,
    api_key=API_KEY,
    temperature=0,
    top_p=1.0,
):
    """
    Call a DeepInfra model with streaming and return the log probability for the token "dream".
    
    Args:
        model_name (str): The name of the DeepInfra model to call
        prompt (str): The input prompt
        api_key (str): Your DeepInfra API key (if None, uses environment)
        temperature (float): Sampling temperature
        top_p (float): Nucleus sampling parameter
    
    Returns:
        float: Log probability for the token "dream" if found, None otherwise
    """
    url = f"https://api.deepinfra.com/v1/inference/{model_name}"
    
    # Use provided API key or get from environment/deepctl auth
    if api_key is None:
        # Try to get from environment variable or use deepctl auth token
        api_key = os.environ.get("DEEPINFRA_API_TOKEN", None)
        if api_key is None:
            # Try to get from deepctl if available
            import subprocess
            try:
                api_key = subprocess.check_output(["deepctl", "auth", "token"]).decode("utf-8").strip()
            except:
                print("Error: No API key provided and couldn't get from deepctl auth token")
                return None
    
    headers = {
        "Authorization": f"bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "input": prompt,
        "stream": True
    }
    
    # Process the streaming response
    with requests.post(url, headers=headers, json=data, stream=True) as response:
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                
                # Skip lines that don't start with "data: "
                if not line.startswith("data: "):
                    continue
                
                # Extract JSON content after "data: "
                json_str = line[len("data: "):]
                try:
                    data = json.loads(json_str)
                    if "token" in data:
                        token_info = data["token"]
                        print(token_info)
                        token_text = token_info.get("text", "").strip()
                        print(token_text)
                        if "player" in token_text:
                            return token_info.get("logprob", None)
                        
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON: {json_str}")
        
        return None


# Example usage
if __name__ == "__main__":
    # Example model
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    prompt = "I have this dream of becoming a basketball"
    
    logprob = call_deepinfra_model_with_logprobs_streaming(
        model_name=model_name,
        prompt=prompt
    )
    
    if logprob is not None:
        print(f"Log probability for 'player': {logprob}")
    else:
        print("Token 'player' not found in the response")