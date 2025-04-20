import requests
import json
import numpy as np
import base64

class LlamaHelper:
    def __init__(self, api_url="http://10.29.226.196:8080/api/generate", model="moondream:latest", username="vittalt@mit.edu", password="Soccorb@l1"):
        """
        Initializes the LlamaHelper with authentication for querying the API.
        
        :param api_url: The API endpoint for the model.
        :param model: The preferred model to query.
        :param username: Your login username.
        :param password: Your login password.
        """
        self.api_url = api_url
        self.model = model
        # Encode username and password for Basic Auth
        credentials = f"{username}:{password}"
        self.auth_header = {
            "Authorization": "Basic " + base64.b64encode(credentials.encode()).decode(),
            "Content-Type": "application/json"
        }

    def query_llm(self, prompt):
        """
        Queries the model with a given prompt.
        
        :param prompt: The input query for the model.
        :return: Model response and uncertainty score.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(self.api_url, json=payload, headers=self.auth_header)
            response.raise_for_status()
            data = response.json()
            return data.get("response", ""), 0.0  # Assuming no uncertainty info is provided

        except requests.exceptions.RequestException as e:
            print(f"Error querying Llama model: {e}")
            return None, 1.0  # Return high uncertainty on failure


# Example Usage
if __name__ == "__main__":
    # Initialize with credentials
    llama_helper = LlamaHelper(username="your_username", password="your_password")

    # Example prompt
    response, uncertainty = llama_helper.query_llm("Why is the sky blue?")
    print(f"Response: {response}\nUncertainty: {uncertainty:.4f}")
