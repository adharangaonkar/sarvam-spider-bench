# probe_backends.py
from lm_eval.api.registry import MODEL_REGISTRY
for n in sorted(MODEL_REGISTRY.keys()):
    print(n)