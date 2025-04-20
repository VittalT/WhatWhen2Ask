import torch
from transformers import BlipForConditionalGeneration, BlipProcessor
from PIL import Image
import jsonpickle
import numpy as np

class BLIPHelper:
    def __init__(self, model_name="Salesforce/blip-image-captioning-base", device=None):
        # Choose device automatically if not provided
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Download and load the BLIP processor and model from Hugging Face.
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)

    def query_vlm(self, image_path, task, state):
        """
        Query the BLIP model with an image plus a text prompt (combining a task description and a JSON-encoded state).
        Returns the generated hint and an uncertainty measure based on token probabilities.
        """
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        # Combine the task and state information into a text prompt
        input_text = f"Task: {task}\nState: {jsonpickle.encode(state)}\nProvide a concise hint for the agent:"
        # Prepare inputs for the model
        inputs = self.processor(images=image, text=input_text, return_tensors="pt").to(self.device)
        
        # Generate output with token scores for uncertainty estimation.
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,
            output_scores=True,
            return_dict_in_generate=True
        )
        # Decode the output hint
        hint = self.processor.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
        # Compute token-level log probabilities
        token_logprobs = []
        # outputs.scores is a tuple (one per generated token) of tensors with shape (batch_size, vocab_size)
        # We align each generated token (skipping the first token if needed) with its score.
        for score, token_id in zip(outputs.scores, outputs.sequences[0][1:]):  
            probs = torch.softmax(score[0], dim=-1)  # Get probabilities from logits
            token_prob = probs[token_id].item()
            token_logprobs.append(np.log(token_prob))
        
        # Average token probability and compute uncertainty
        avg_probability = np.mean(np.exp(token_logprobs))
        uncertainty = 1 - avg_probability
        
        return hint, uncertainty

# Example usage for BLIP:
if __name__ == "__main__":
    # Example state and task (you may replace these with your actual data)
    state = {
        'step': 5,
        'agent': {'pos': (3, 7), 'room': 'K', 'dir': 3, 'carrying': None},
        'objects': [{'name': 'fruit', 'type': 'Pickable', 'pos': (12, 2), 'room': 'L'}]
    }
    task = "move the fruit to the dining room"
    image_path = "path_to_your_image.jpg"  # Replace with the actual path to your image

    blip_helper = BLIPHelper()
    hint, uncertainty = blip_helper.query_vlm(image_path, task, state)
    print("=== BLIP Output ===")
    print(f"Hint: {hint}")
    print(f"Uncertainty: {uncertainty:.4f}")
