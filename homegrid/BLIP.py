# BLIP.py

from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch
import torch.nn.functional as F
import requests
from io import BytesIO
import json


class BLIP2Helper:
    def __init__(self, model_name="Salesforce/blip2-flan-t5-xl"):
        """
        Initialize the BLIP2 processor and model.
        """
        self.processor = Blip2Processor.from_pretrained(model_name, use_fast=True)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name)

    def query_llm(self, pil_image, context_str):
        """
        Query the LLM using a PIL image and a context string.

        Args:
            pil_image (PIL.Image): The input image. BLIP2's processor expects a PIL image to apply
                                   its internal transforms.
            context_str (str): A JSON-formatted string containing context information.

        Returns:
            hint (str): The generated text output.
            uncertainty (float): The average token-level entropy computed from the generated token scores,
                                 serving as a measure of uncertainty.
        """
        # Prepare inputs using the image and context string.
        prompt_text = f"""
        You are assisting a reinforcement learning agent navigating a grid-based house to complete tasks by interacting with objects.

        The agent uses a DQN that takes in its visual observation, state context, and your hint to decide which action to take. Available actions: left, right, up, down, pickup, drop, get, pedal, grasp, lift.

        Current state:
        {context_str}

        You are also provided with the agent’s current visual observation.

        Do not repeat the task or any obvious information already present in the context — the agent already has this.

        Instead, analyze the image and state to infer a new, helpful insight that will inform the agent's upcoming decisions — such as which direction to move, where an object is located, or what action might be effective.

        Provide one concise, specific, and actionable hint that can help the agent over the next several steps.
        """

        # pil_image.show()
        # print(prompt_text)

        inputs = self.processor(images=pil_image, text=prompt_text, return_tensors="pt")

        # Generate outputs with token scores (logits) needed to compute uncertainties.
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # Decode the generated tokens to produce the hint.
        generated_ids = outputs.sequences[0]
        hint = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Compute the token-level entropy (uncertainty) and average them.
        entropies = []
        for token_logits in outputs.scores:
            probs = torch.softmax(token_logits, dim=-1)
            log_probs = torch.log(probs + 1e-9)  # Add epsilon to avoid log(0)
            token_entropy = -torch.sum(
                probs * log_probs, dim=-1
            )  # shape: (batch_size,)
            entropies.append(token_entropy[0].item())

        avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0

        return hint, avg_entropy


def main():
    # Use a known working image URL (Unsplash image).
    image_url = "https://images.unsplash.com/photo-1516117172878-fd2c41f4a759?ixlib=rb-1.2.1&auto=format&fit=crop&w=634&q=80"

    # Fetch the image.
    response = requests.get(image_url)
    response.raise_for_status()  # Ensure the request was successful.
    pil_image = Image.open(BytesIO(response.content))

    # Create a dummy context as a JSON-formatted string.
    dummy_context = {
        "direction": "right",
        "carrying object": "none",
        "front object": "wall",
        "task": "navigate",
        "prior VLM output": "",
    }
    context_str = json.dumps(dummy_context)

    # Initialize BLIP2Helper and query the LLM.
    helper = BLIP2Helper()
    hint, uncertainty = helper.query_llm(pil_image, context_str)

    # Print the results.
    print("Generated Hint:", hint)
    print("Uncertainty:", uncertainty)


if __name__ == "__main__":
    main()
