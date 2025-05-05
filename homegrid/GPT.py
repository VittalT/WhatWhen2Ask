# GPT.py

import openai
import os
import json
import numpy as np
import jsonpickle
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
import torch
from PIL import Image
from homegrid.utils import format_prompt, get_dummy_state, format_symbolic_state

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class GPT4Helper:
    def __init__(self, model="gpt-4o"):
        # Load API key from JSON in the parent directory
        key_file_path = os.path.expanduser("~") + "/openai_key.json"
        with open(key_file_path) as json_file:
            key = json.load(json_file)
        self.api_key = key["my_openai_api_key"]

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        self.model = model

        # Initialize sentence transformer for action mapping
        self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.action_space = [
            "left",
            "right",
            "up",
            "down",
            "pickup",
            "drop",
            "get",
            "pedal",
            "grasp",
            "lift",
        ]
        self.action_embeddings = self.encoder.encode(
            self.action_space, convert_to_tensor=True
        )
        if torch.cuda.is_available():
            self.action_embeddings = self.action_embeddings.cuda()
            self.encoder.to("cuda")

    def query_action(self, state, task):
        """
        Query GPT to get the next action based on the state and task.
        Args:
            state: Tuple of (observation, context) where observation is a PIL image
                and context is the symbolic state
            task: String describing the task
        Returns:
            action: Index of the selected action
            confidence: Confidence score for the selected action
        """
        observation, info = state
        context = format_symbolic_state(info["symbolic_state"])
        # observation.show()
        # print(state_str)

        # Convert PIL image to base64 for API
        import base64
        from io import BytesIO

        buffered = BytesIO()
        observation.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        prompt = f"""
You are an intelligent robot operating in a grid-based environment. Your goal is to complete the following task: **{task}**.

You can perform the following actions:  
left, right, up, down, pickup, drop, get, pedal, grasp, lift

### Important Rules:
- Moving in a direction (left, right, up, down) causes the agent to move one tile in that direction and then face that direction.
- You can only interact (pickup, drop, get, pedal, grasp, lift) objects if you are facing them.
- You can only carry one object at a time.
- There may be obstacles in the way that prevent you from moving in a direction.

Here is your current state and surroundings. Locations are given in the format (x, y), where x represents horizontal position (left to right) and y represents vertical position (top to bottom).
{context}

### Output Format:
Think step by step to reason through the current situation.  
Then, output the best next action.  
The first word of your response **must** be the chosen action (in lowercase), followed by a short explanation (on the next line).

Example:  
"left
The agent needs to get closer to the papers in order to pick them up. The papers at (3, 10) are to the left of the agent at (7, 10)"

Now, determine the best next action.
"""

        print(prompt)

        messages = [
            {
                "role": "system",
                "content": "You are an expert planner for a robot in a partially-observable grid environment.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_str}",
                            "detail": "high",
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=100,
            temperature=0.7,
            logprobs=True,
            top_logprobs=5,
        )

        # Get the generated action
        generated_text = response.choices[0].message.content.strip().lower()
        print(generated_text)

        # Extract token-level log probabilities
        token_logprobs = [
            token.logprob for token in response.choices[0].logprobs.content
        ]

        # Convert log probabilities to raw probabilities and get confidence
        token_probabilities = np.exp(token_logprobs)
        confidence = np.mean(token_probabilities)

        # Map to closest action using cosine similarity
        generated_action = generated_text.split()[0]
        gen_embedding = self.encoder.encode(generated_action, convert_to_tensor=True)
        if torch.cuda.is_available():
            gen_embedding = gen_embedding.cuda()

        cosines = util.cos_sim(gen_embedding, self.action_embeddings)[0]
        action = int(torch.argmax(cosines))

        return action, confidence

    def query_llm(self, task, obs, info):
        """
        Query GPT to provide a hint based on the full state and task, including token log probabilities.
        Args:
            state: Tuple of (observation, context) where observation is a PIL image and context is the symbolic state
            task: The task description as a string.
        Returns:
            hint: The LLM's generated hint
            confidence: Confidence score for the generated hint
        """
        observation = Image.fromarray(obs["image"])
        prompt = format_prompt(task, info)

        # Convert PIL image to base64 for API
        import base64
        from io import BytesIO

        buffered = BytesIO()
        observation.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Construct the chat messages
        messages = [
            {
                "role": "system",
                "content": "You are an expert assistant helping an RL agent navigate a grid-based environment.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_str}",
                            "detail": "high",
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ]
        # print(messages)

        # Query GPT using the Chat Completion endpoint with the new API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=100,
            temperature=0.7,
            logprobs=True,
            top_logprobs=5,
        )

        # Extract the hint from the new response format
        hint = response.choices[0].message.content.strip()

        # Extract token-level log probabilities
        token_logprobs = [
            token.logprob for token in response.choices[0].logprobs.content
        ]

        # Convert log probabilities to raw probabilities
        token_probabilities = np.exp(token_logprobs)
        confidence = np.mean(token_probabilities)

        return hint, confidence


# Example Usage
if __name__ == "__main__":
    state, task = get_dummy_state()

    # Initialize the helper and query the LLM
    gpt_agent = GPT4Helper(model="gpt-4o-mini")

    # Test query_actio
    action, action_confidence = gpt_agent.query_action(state, task)
    print(f"Action: {gpt_agent.action_space[action]}")
    print(f"Action Confidence: {action_confidence:.4f}")
