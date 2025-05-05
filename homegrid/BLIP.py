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
from homegrid.utils import format_prompt, get_dummy_state, format_symbolic_state

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
        observation, info = state
        # observation.show()
        context = format_symbolic_state(info["symbolic_state"])
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
        # - An image showing the robot's partial observation of the environment at the current time step.
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

        logits = torch.stack(
            outputs.scores, dim=0
        )  # (num_tokens, batch_size, vocab_size)
        probs = torch.softmax(logits, dim=-1)
        top_token_probs = torch.max(probs[:, 0, :], dim=-1).values  # (num_tokens,)
        confidence = top_token_probs.mean().item()

        # 6. Cosine similarity and choose best action
        gen_embedding = self.encoder.encode(generated_text, convert_to_tensor=True)
        cosines = util.cos_sim(gen_embedding, self.action_embeddings)[0]
        action = int(torch.argmax(cosines))

        return action, confidence

    def query_llm(self, task, obs, info):
        """
        Query the LLM using a PIL image and a context string.

        Args:
            pil_image (PIL.Image): The input image. BLIP2's processor expects a PIL image to apply its internal transforms.
            context_str (str): A JSON-formatted string containing context information.

        Returns:
            hint (str): The generated text output.
            confidence (float): The average probability of the top token at each position.
        """
        observation = Image.fromarray(obs["image"])
        prompt = format_prompt(task, info)

        inputs = self.processor(images=observation, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

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

        # Calculate confidence (top token probabilities mean)
        logits = torch.stack(
            outputs.scores, dim=0
        )  # (num_tokens, batch_size, vocab_size)
        probs = torch.softmax(logits, dim=-1)
        top_token_probs = torch.max(probs[:, 0, :], dim=-1).values  # (num_tokens,)
        confidence = top_token_probs.mean().item()

        # Return the hint and confidence (not the entropy)
        return hint, confidence


def main():
    state, task = get_dummy_state()
    # Initialize BLIP2Helper and query the LLM.
    blip_agent = BLIP2Helper()
    action, confidence = blip_agent.query_action(state, task)

    # Print the results.
    print(f"Action: {blip_agent.action_space[action]}")
    print("Confidence:", confidence)


if __name__ == "__main__":
    main()
