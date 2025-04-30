import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

print("=== PyTorch Check ===")
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
elif torch.backends.mps.is_available():
    print("Running on Apple MPS backend (Apple Silicon GPU)")
else:
    print("Running on CPU only")

print("\n=== Sentence-Transformers Check ===")
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    device = model.device
    print("Model loaded on:", device)
    embedding = model.encode("This is a GPU test.")
    print("Embedding shape:", embedding.shape)
except Exception as e:
    print("Error loading SentenceTransformer:", e)

print("\n=== Transformers Model Check ===")
try:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased").to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    input_ids = tokenizer("GPU test", return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        output = model(input_ids)
    print("Transformers model output shape:", output.last_hidden_state.shape)
except Exception as e:
    print("Error loading Transformers model:", e)
