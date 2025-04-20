import openai
import os
import json
import numpy as np
import jsonpickle

class GPT4Helper:
    def __init__(self, model="gpt-4"):
        # Load API key from JSON in the parent directory
        key_file_path = os.path.expanduser("~") + "/openai_key.json"
        with open(key_file_path) as json_file:
            key = json.load(json_file)
        self.api_key = key["my_openai_api_key"]

        # Set API key for OpenAI
        openai.api_key = self.api_key
        self.model = model

    def query_llm(self, state, task):
        """
        Query GPT to provide a hint based on the full state and task, including token log probabilities.
        :param state: The full state as a dictionary.
        :param task: The task description as a string.
        :return: The LLM's hint, token probabilities, and uncertainty score.
        """
        # Format the state as a JSON-like string
        state_str = jsonpickle.encode(state)
        # print(state_str, task)

        # Construct the chat messages
        messages = [
            {"role": "system", "content": "You are an expert assistant helping an RL agent navigate a grid-based environment."},
            {"role": "user", "content": f"""
            The agent is in the following state:
            {state_str}

            The task is:
            {task}

            Based on this information, provide the most helpful hint for the agent. Your hint should be concise and actionable.
            """}
        ]
        print(messages)

        # Query GPT using the Chat Completion endpoint
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            max_tokens=100,
            temperature=0.7,
            logprobs=True,  # Include log probabilities in the response
            top_logprobs=5  # Retrieve the top 5 token probabilities
        )

        # TF IDF

        # Extract the hint
        hint = response["choices"][0]["message"]["content"].strip()

        # Extract token-level log probabilities
        token_logprobs = [token["logprob"] for token in response["choices"][0]["logprobs"]["content"]]
        
        # Convert log probabilities to raw probabilities
        token_probabilities = np.exp(token_logprobs)

        # Calculate average probability and uncertainty
        avg_probability = np.mean(token_probabilities)
        uncertainty = 1 - avg_probability  # Higher uncertainty for lower avg probability

        return hint, uncertainty


# Example Usage
if __name__ == "__main__":
    # Example state and task
    state = {'step': 5, 'agent': {'pos': (3, 7), 'room': 'K', 'dir': 3, 'carrying': None}, 'objects': [{'name': 'recycling bin', 'type': 'Storage', 'pos': (12, 10), 'room': 'D', 'state': 'closed', 'action': 'pedal', 'invisible': None, 'contains': []}, {'name': 'compost bin', 'type': 'Storage', 'pos': (11, 1), 'room': 'L', 'state': 'open', 'action': 'grasp', 'invisible': None, 'contains': []}, {'name': 'fruit', 'type': 'Pickable', 'pos': (12, 2), 'room': 'L', 'state': None, 'action': None, 'invisible': False, 'contains': None}, {'name': 'papers', 'type': 'Pickable', 'pos': (3, 10), 'room': 'K', 'state': None, 'action': None, 'invisible': False, 'contains': None}, {'name': 'plates', 'type': 'Pickable', 'pos': (9, 1), 'room': 'L', 'state': None, 'action': None, 'invisible': True, 'contains': None}, {'name': 'bottle', 'type': 'Pickable', 'pos': (10, 8), 'room': 'D', 'state': None, 'action': None, 'invisible': True, 'contains': None}], 'front_obj': None, 'unsafe': {'name': None, 'poss': {}, 'end': -1}}
    task = "move the fruit to the dining room"

    # Initialize the helper and query the LLM
    llm_helper = GPT4Helper(model="gpt-4o")
    hint, uncertainty = llm_helper.query_llm(state, task)

    # Print the results
    print(f"Hint: {hint}")
    print(f"Uncertainty: {uncertainty:.4f}")
