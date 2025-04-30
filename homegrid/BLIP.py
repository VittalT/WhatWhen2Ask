# BLIP.py

from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch
import torch.nn.functional as F
import requests
from io import BytesIO
import json
from sentence_transformers import SentenceTransformer, util
from pprint import pprint
import os

# Add this after the imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class BLIP2Helper:
    def __init__(
        self,
        model_name: str = "Salesforce/blip2-flan-t5-xl",
        encoder_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Initialize BLIP-2 for generation and a sentence transformer for action embedding.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # BLIP-2
        self.processor = Blip2Processor.from_pretrained(model_name, use_fast=True)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name).to(
            self.device
        )
        # Encoder for cosine similarity
        self.encoder = SentenceTransformer(encoder_model_name).to(self.device)
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
        ).to(self.device)

        # for action in self.action_space:
        #     tokens = self.processor.tokenizer.encode(action, add_special_tokens=False)
        #     print(f"{action}: {len(tokens)} tokens -> {tokens}")

    def query_action(self, state, task, max_new_tokens=100):
        """
        Given a state tuple (image, symbolic_context), prompt BLIP-2 in Act style
        (only the action word), compute log-prob-based uncertainty, and map to
        the discrete action set via cosine similarity.
        """
        observation, context = state
        # observation.show()

        prompt = f"""
You are an expert planner for a robot in a partially‐observable grid. The robot's task is to {task}.
Actions: left, right, up, down, pickup, drop, get, pedal, grasp, lift.

Notes:
- Holds one object at a time.
- Bins must be opened before use.
- Explore to find hidden objects.

INPUT:
- image: current partial view
- symbolic_state: {context}

OUTPUT:
Think through the task and output the action the robot should take next."""
        # - If testing, output the action word alone, without quotes, punctuation, or extra text.
        # - Exactly one word from the action list above
        # - No quotes, punctuation or extra text

        # # Step 1: Encode with empty context to measure prompt base length
        # empty_context_prompt = base_prompt.format(context="")
        # base_tokens = self.processor.tokenizer.encode(
        #     empty_context_prompt, return_tensors="pt"
        # )
        # base_len = base_tokens.shape[1]

        # # Step 2: Get how many tokens are left for context
        # MAX_TOKENS = 480
        # remaining_tokens = MAX_TOKENS - base_len

        # # Step 3: Tokenize the context and truncate if needed
        # full_context = str(context)
        # context_tokens = self.processor.tokenizer.encode(
        #     full_context,
        #     truncation=True,
        #     max_length=remaining_tokens,
        # )
        # truncated_context = self.processor.tokenizer.decode(
        #     context_tokens, skip_special_tokens=True
        # )

        # # Step 4: Insert the truncated context into the full prompt
        # prompt = base_prompt.format(context=truncated_context)

        # prompt = f"""
        # You are an expert planner tasked with helping an embodied robot perform a task in a grid-based environment. The robot has only a partially observable view of its surroundings, so it must explore the environment as needed to complete the task.

        # The robot can perform the following actions: left, right, up, down, pickup, drop, get, pedal, grasp, lift.

        # ### Important Notes ###
        # - The robot can hold only one object at a time.
        # For example: If it is holding fruits, it cannot pick up another object until it puts the fruits down.
        # - Some objects, such as bins, may need to be opened before the robot can put objects inside.
        # - If an object is not visible, the robot must explore the environment to locate it.

        # ### INPUT FORMAT ###
        # You will receive:
        # - An image showing the robot’s partial observation of the environment at the current time step.
        # - A symbolic description of the environment, including the robot's location, any object it is holding, and the visible objects.

        # ### OUTPUT FORMAT ###
        # Based on the observation and symbolic state, output the **single best action**.

        # **IMPORTANT:**
        # - Output **ONLY one word**, with **no extra words** or explanations.
        # - The output must be **exactly one** of the following words:
        # "left", "right", "up", "down", "pickup", "drop", "get", "pedal", "grasp", "lift".
        # - **DO NOT** output anything like "move left", "go right", "pick up item", or any other variation.
        # - Output the action word alone, without quotes, punctuation, or extra text.

        # **IMPORTANT: DO NOT OUTPUT ANYTHING BEYOND THE SPECIFIED FORMAT.**

        # ### Inputs ###
        # 'symbolic_state': {context}
        # """

        # pil_image.show()
        # print(prompt_text)

        inputs = self.processor(images=observation, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate outputs with token scores (logits) needed to compute uncertainties.
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
        )
        print(prompt)

        # Decode the generated tokens to produce the hint.
        generated_ids = outputs.sequences[0]
        generated_text = (
            self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
            .strip()
            .lower()
        )
        print(generated_text)

        # Compute the token-level entropy (uncertainty) and average them.
        logits = torch.stack(outputs.scores, dim=0)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-9)
        token_entropies = -torch.sum(probs * log_probs, dim=-1)[:, 0]
        avg_entropy = (
            token_entropies.mean().item() if token_entropies.numel() > 0 else 0.0
        )

        # Normalize
        vocab_size = probs.shape[-1]
        max_entropy = torch.log(torch.tensor(vocab_size, dtype=torch.float32))
        uncertainty = avg_entropy / max_entropy.item()
        print(uncertainty)

        logits = torch.stack(
            outputs.scores, dim=0
        )  # (num_tokens, batch_size, vocab_size)
        probs = torch.softmax(logits, dim=-1)
        top_token_probs = torch.max(probs[:, 0, :], dim=-1).values  # (num_tokens,)
        confidence = top_token_probs.mean().item()
        print(confidence)

        # 6. Cosine similarity and choose best action
        gen_embedding = self.encoder.encode(generated_text, convert_to_tensor=True)
        cosines = util.cos_sim(gen_embedding, self.action_embeddings)[0]
        action = int(torch.argmax(cosines))

        print(self.action_space[action])

        return action, confidence

    def query_llm(self, pil_image, context_str):
        """
        Query the LLM using a PIL image and a context string.

        Args:
            pil_image (PIL.Image): The input image. BLIP2's processor expects a PIL image to apply its internal transforms.
            context_str (str): A JSON-formatted string containing context information.

        Returns:
            hint (str): The generated text output.
            uncertainty (float): The average token-level entropy computed from the generated token scores, serving as a measure of uncertainty.
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
