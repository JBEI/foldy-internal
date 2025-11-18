import logging
import sys
logging.basicConfig(level=logging.INFO)
print("Python:", sys.version)
try:
    import torch
    print("Torch:", torch.__version__)
    print("Torch CUDA:", torch.version.cuda if torch.version.cuda else "No CUDA")
except Exception as e:
    print("Torch error:", str(e))

try:
    import torchvision
    print("Torchvision:", torchvision.__version__)
except Exception as e:
    print("Torchvision error:", str(e))

try:
    import transformers
    print("Transformers:", transformers.__version__)
except Exception as e:
    print("Transformers error:", str(e))

try:
    from E1.modeling import E1ForMaskedLM
    logging.info("E1.modeling imported successfully")
except Exception as e:
    logging.warning("E1 unavailable (Python/version issue); skipping E1 features.")