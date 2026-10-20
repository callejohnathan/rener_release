import torch
import spacy
import transformers

print("=== Verificación del entorno ===")
print(f"PyTorch:         {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:             {torch.cuda.get_device_name(0)}")
    print(f"VRAM total:      {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"spaCy:           {spacy.__version__}")
print(f"Transformers:    {transformers.__version__}")

spacy.prefer_gpu()
nlp = spacy.blank("en")
print(f"spaCy GPU:       {spacy.prefer_gpu()}")
print("\n✓ Entorno listo" if torch.cuda.is_available() else "\n✗ GPU no detectada")