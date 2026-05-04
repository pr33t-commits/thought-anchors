from transformer_lens.model_bridge import TransformerBridge
import os
from dotenv import load_dotenv
load_dotenv()
# HF_KEY = os.getenv("HF_KEY")

# Method 1: Simple loading (recommended)
bridge = TransformerBridge.boot_transformers(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    # token=HF_KEY,
    device="cpu",  # or "cpu"
    # dtype="float32",
    trust_remote_code=False,  # Usually not needed for official models
)

# Test basic forward pass
text = "Hello, how are you?"
logits = bridge(text, return_type="logits")
print(f"Logits shape: {logits.shape}")

# Test text generation
generated = bridge.generate("Once upon a time", max_new_tokens=50)
print(f"Generated text: {generated}")

# Test run_with_cache (for mechanistic interpretability)
logits, cache = bridge.run_with_cache("The meaning of life is")
# print(f"Cache keys: {cache.keys()}")