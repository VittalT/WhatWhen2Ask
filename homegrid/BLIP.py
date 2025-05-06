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
from homegrid.utils import format_prompt, format_action_prompt, get_dummy_state

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
        prompt = format_action_prompt(task, info)
        # print(prompt)
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
        # print(prompt)

        # Decode the generated tokens to produce the hint.
        generated_ids = outputs.sequences[0]
        generated_text = (
            self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
            .strip()
            .lower()
        )
        # print(generated_text)

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
        # observation.show()
        # print(prompt)

        inputs = self.processor(images=observation, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate outputs with token scores (logits) needed to compute uncertainties.
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
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
