# Run from the repository root: python scripts/combinar_corpus.py
import json

total = 0
con_etiquetas = 0

with open("data/corpus_completo.jsonl", "w", encoding="utf-8") as fout:
    for line in open("data/dataset_doccano_v3.jsonl", encoding="utf-8"):
        item = json.loads(line.strip())
        if item.get("label"):
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            con_etiquetas += 1
        total += 1

print(f"Total: {total} | Con etiquetas: {con_etiquetas}")