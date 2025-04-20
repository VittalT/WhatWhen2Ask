import torch
from transformers import AutoTokenizer, Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import jsonpickle
import numpy as np

class LLaVAHelper:
    def __init__(self, model_name="LLaVA/LLaVA-7B", device=None):
        # Choose device automatically if not provided
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # For LLaVA, we use a BLIP2-style processor (often shared with the vision encoder) 
        # and a LLaVA checkpoint (this is a placeholder and should be replaced with the correct model ID).
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def query_vlm(self, image_path, task, state):
        """
        Query the LLaVA model with an image plus a text prompt (including task and JSON-encoded state).
        Returns the generated hint and an uncertainty measure computed from token probabilities.
        """
        # Load and preprocess the image.
        image = Image.open(image_path).convert("RGB")
        # Prepare the combined text prompt.
        input_text = f"Task: {task}\nState: {jsonpickle.encode(state)}\nProvide a concise hint for the agent:"
        # Process inputs using the shared processor.
        inputs = self.processor(images=image, text=input_text, return_tensors="pt").to(self.device)
        
        # Generate output with token scores.
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,
            output_scores=True,
            return_dict_in_generate=True
        )
        # Decode the output text.
        hint = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
        # Compute token-level log probabilities.
        token_logprobs = []
        for score, token_id in zip(outputs.scores, outputs.sequences[0][1:]):
            probs = torch.softmax(score[0], dim=-1)
            token_prob = probs[token_id].item()
            token_logprobs.append(np.log(token_prob))
        
        # Compute average probability and uncertainty.
        avg_probability = np.mean(np.exp(token_logprobs))
        uncertainty = 1 - avg_probability
        
        return hint, uncertainty

# Example usage for LLaVA:
if __name__ == "__main__":
    # Example state and task
    state = {
        'step': 5,
        'agent': {'pos': (3, 7), 'room': 'K', 'dir': 3, 'carrying': None},
        'objects': [{'name': 'fruit', 'type': 'Pickable', 'pos': (12, 2), 'room': 'L'}]
    }
    task = "move the fruit to the dining room"
    image_path = "path_to_your_image.jpg"  # Replace with your image file path

    llava_helper = LLaVAHelper()
    hint, uncertainty = llava_helper.query_vlm(image_path, task, state)
    print("=== LLaVA Output ===")
    print(f"Hint: {hint}")
    print(f"Uncertainty: {uncertainty:.4f}")
