from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os  # Added to handle path expansion

def main():
    # Load the model and tokenizer
    model_name = os.path.expanduser("../models/LLaVA")  # Use the local model directory
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Test the model with a simple query
    query = "What is the capital of France?"
    inputs = tokenizer(query, return_tensors="pt")
    
    # Compute logits
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Compute log probabilities
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Decode and print the response
    generated_ids = torch.argmax(logits, dim=-1)
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print("Model response:", response)
    print("Logits:", logits)
    print("Log probabilities:", log_probs)

if __name__ == "__main__":
    main()
