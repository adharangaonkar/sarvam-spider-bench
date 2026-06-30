# inspect_samples.py
import json
from lm_eval import simple_evaluate

results = simple_evaluate(
    model="gguf",
    model_args="base_url=http://127.0.0.1:8080",
    tasks=["gsm8k"],
    num_fewshot=5,
    limit=10,
    log_samples=True,
)

samples = results["samples"]["gsm8k"]

with open("gsm8k_samples.json", "w") as f:
    json.dump(samples, f, indent=2, default=str)

# print the bits that matter for extractor choice
for i, s in enumerate(samples):
    print(f"\n===== Q{i} =====")
    print("TARGET:", s["target"])
    print("MODEL OUTPUT:", s["resps"][0][0][:600])  # first 600 chars
    print("FILTERED (strict):", s.get("filtered_resps"))