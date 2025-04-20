import requests
import json

QUERY_CHAT = True
# API URL
api_url1 = "http://10.29.226.196:11434/api/generate"
api_url2 = "http://10.29.226.196:11434/api/chat"

# Request payload
payload1 = {
    "model": "llava:34b",
    "prompt": "Why is the sky blue?",
    "stream": False,
    "logprobs": True,
}

payload2 = {
    "model": "llava:34b",
    "messages": [
        {"role": "user", "content": "Why is the sky blue?"}
    ],
    "stream": False,
    # "logprobs": True,
}


# Send the request
try:
    response = requests.post(api_url2, json=payload2) if QUERY_CHAT else requests.post(api_url1, json=payload1)
    response.raise_for_status()  # Raise an error for failed requests
    data = response.json()
    
    # Print the response
    print("Response:", data.get("message", {}).get("content", "No response")) if QUERY_CHAT else print("Response:", data.get("response", "No response"))
except requests.exceptions.RequestException as e:
    print(f"Error querying model: {e}")
