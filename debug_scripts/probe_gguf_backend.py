# probe_gguf_backend.py
import inspect
from lm_eval.models.gguf import GGUFLM

print("init signature:")
print(inspect.signature(GGUFLM.__init__))
print("\nhas loglikelihood:", hasattr(GGUFLM, "loglikelihood"))
print("\n--- __init__ source ---")
print(inspect.getsource(GGUFLM.__init__))