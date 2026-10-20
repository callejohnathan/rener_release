# Run from the repository root: python scripts/convertir_corpus.py
import glob, json, spacy, random
from spacy.tokens import DocBin
from pathlib import Path

nlp = spacy.blank("en")

LABELS = {
    "ACTOR", "FUNCTION", "CONCEPT", "PROPERTY",
    "RESTRICTION", "EVENT_TRIGGER", "EVENT_ENTITY"
}

# Lote 9 como test (dominio no visto)
TEST_IDS = set()
with open("data/dataset_doccano_v3.jsonl", encoding="utf-8") as f:
    lines = f.readlines()

# Identifica el rango del lote 9 por texto característico
def es_lote9(text):
    markers = ["DAMS", "Zooniverse", "Cornell", "archivist",
               "spancat", "content file", "ringback"]
    return any(m in text for m in markers)

train_data, dev_data, test_data = [], [], []
random.seed(42)

all_items = []
with open("data/corpus_completo.jsonl", encoding="utf-8") as f:
    for line in f:
        all_items.append(json.loads(line.strip()))

test_items  = [x for x in all_items if es_lote9(x["text"])]
other_items = [x for x in all_items if not es_lote9(x["text"])]

random.shuffle(other_items)
split = int(len(other_items) * 0.85)
train_items = other_items[:split]
dev_items   = other_items[split:]

# --- Aumento sintetico dirigido a EVENT_TRIGGER/EVENT_ENTITY/RESTRICTION ---
# Se agrega SOLO a train (nunca a dev/test) para no inflar artificialmente
# las metricas de evaluacion. Defensa extra: se descarta cualquier item que
# haga match con los marcadores del dominio held-out (no deberia haber
# ninguno, pero se verifica igual porque dev/test dependen de esos marcadores).
synthetic_items = []
for fp in sorted(glob.glob("data/synthetic/synthetic_batch*.jsonl")):
    with open(fp, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            if es_lote9(item["text"]):
                print(f"  ! descartado de {fp} por coincidir con marcador held-out: {item['text'][:60]}")
                continue
            synthetic_items.append(item)

n_real_train = len(train_items)
train_items = train_items + synthetic_items

# --- Oversampling de clases raras (SOLO train) ---
# En vez de inventar mas texto (el lote 5 demostro que eso puede introducir
# ruido), se duplican historias YA existentes que contienen EVENT_TRIGGER/
# EVENT_ENTITY/RESTRICTION, para que el modelo vea mas señal de gradiente de
# esas clases por epoca sin tocar la diversidad de contenido. Nunca se aplica
# a dev/test.
OVERSAMPLE_EVENT = 3       # historias con EVENT_TRIGGER o EVENT_ENTITY -> x3
OVERSAMPLE_RESTRICTION = 2  # historias con RESTRICTION (y sin EVENT_*) -> x2

def oversample_factor(item):
    labs = {l for _, _, l in item.get("label", [])}
    if "EVENT_TRIGGER" in labs or "EVENT_ENTITY" in labs:
        return OVERSAMPLE_EVENT
    if "RESTRICTION" in labs:
        return OVERSAMPLE_RESTRICTION
    return 1

oversampled = []
n_extra = 0
for item in train_items:
    factor = oversample_factor(item)
    oversampled.append(item)
    for _ in range(factor - 1):
        oversampled.append(item)
        n_extra += 1
train_items = oversampled

print(f"Train: {n_real_train} real + {len(synthetic_items)} sintetico + {n_extra} oversampled "
      f"= {len(train_items)} | Dev: {len(dev_items)} (100% real) | Test: {len(test_items)} (100% real)")

def build_docbin(items, name):
    db = DocBin()
    skipped = 0
    for item in items:
        doc = nlp.make_doc(item["text"])
        spans = []
        for start, end, label in item.get("label", []):
            if label in LABELS:
                span = doc.char_span(start, end, label=label,
                                     alignment_mode="expand")
                if span is not None:
                    spans.append(span)
                else:
                    skipped += 1
        doc.spans["sc"] = spans
        db.add(doc)
    print(f"  {name}: {skipped} spans descartados por desalineación")
    return db

Path("data").mkdir(exist_ok=True)
build_docbin(train_items, "train").to_disk("data/train.spacy")
build_docbin(dev_items,   "dev"  ).to_disk("data/dev.spacy")
build_docbin(test_items,  "test" ).to_disk("data/test.spacy")

print("\nCorpus convertido en data/")