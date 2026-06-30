# smoke_test.py
from lm_eval import simple_evaluate

results = simple_evaluate(
    model="gguf",
    model_args="base_url=http://127.0.0.1:8080",
    tasks=["gsm8k"],
    num_fewshot=5,
    limit=10,
)

print(results["results"])