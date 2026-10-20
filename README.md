# RENER — Extended Entity Model for Semantic Completeness in User Stories

This repository accompanies the article *"Detecting Semantic Completeness in User Stories: A Span-Based Named Entity Recognition Approach"* by Johnathan Mauricio Calle Gallego (Universidad de Medellín) and Carlos Mario Zapata Jaramillo (Universidad Nacional de Colombia), submitted to *Automated Software Engineering* (Springer).

It contains the final annotated corpus, the synthetic augmentation data used for training, the spaCy training configuration, and the scripts needed to reproduce the reported model and results. It intentionally contains **only the final, validated artifacts**: exploratory intermediate models, ablations, and the two discarded augmentation batches are described in the article but not included here.

## Contents

```
config.cfg                       spaCy training config (RoBERTa-base transformer + spancat,
                                  n-gram range suggester 1–32 tokens)
data/
  dataset_doccano_v3.jsonl       Doccano export of all annotated user stories (1,666 lines;
                                  includes the EVENT_ENTITY and RESTRICTION annotation fixes
                                  described in the article)
  corpus_completo.jsonl          Same corpus filtered to stories with at least one label (1,663)
  synthetic/
    synthetic_batch1..4.jsonl    186 synthetic user stories used for training only
  LICENSE                        CC BY 4.0 notice for the data
scripts/
  combinar_corpus.py             dataset_doccano_v3.jsonl -> corpus_completo.jsonl
  convertir_corpus.py            Builds data/{train,dev,test}.spacy (held-out split, synthetic
                                  data merged into train only, class-aware oversampling)
  ensemble_eval.py               Evaluates the final 3-seed score-averaged ensemble
                                  (exact and partial match, dev and test)
  verificar_entorno.py           Environment check (CUDA / PyTorch / spaCy / Transformers)
results/
  ensemble_final_results.txt     Final reported metrics
CITATION.cff                     How to cite this repository
LICENSE                          MIT licence for the code
```

## Data

### Source and attribution

The user stories are taken from the public collection *Requirements data sets (user stories)* curated by Fabiano Dalpiaz (Utrecht University), distributed under the [Creative Commons Attribution 4.0 International licence](https://creativecommons.org/licenses/by/4.0/):

> Dalpiaz, F. (2018). *Requirements data sets (user stories)* (Version 1) [Data set]. Mendeley Data. https://doi.org/10.17632/7zbk8zsd8y.1

The original text of the stories is unchanged. This repository adds span-level semantic annotations; the annotations are released under the same CC BY 4.0 licence (see `data/LICENSE`).

### Format

Each line of the JSONL files is one user story in Doccano export format:

```json
{"text": "As a Data user, I want to have the 12-19-2017 deletions processed.",
 "label": [[5, 14, "ACTOR"], [26, 66, "FUNCTION"]]}
```

`label` is a list of `[start_char, end_char, LABEL]` triples. Spans may overlap (for example, the same noun phrase can be both `CONCEPT` and `EVENT_ENTITY`).

### Entity schema

| Label | Definition |
|---|---|
| `ACTOR` | Human role, system, or external agent interacting with the software |
| `FUNCTION` | Action, operation, or service an actor performs or the system provides |
| `CONCEPT` | Domain object, entity, or abstraction manipulated or referenced by a function |
| `PROPERTY` | Attribute, value, or characteristic associated with a concept or actor |
| `RESTRICTION` | Explicit constraint limiting or bounding a function or concept |
| `EVENT_TRIGGER` | Condition, occurrence, or stimulus that initiates system behavior |
| `EVENT_ENTITY` | Domain entity that participates in or is affected by an event |

Annotation was performed in [Doccano](https://github.com/doccano/doccano) with overlapping spans enabled.

### Corpus statistics (`corpus_completo.jsonl`)

| Label | Spans | Stories containing it |
|---|---:|---:|
| ACTOR | 1,652 | 1,652 |
| FUNCTION | 1,592 | 1,592 |
| CONCEPT | 1,579 | 907 |
| PROPERTY | 580 | 464 |
| RESTRICTION | 241 | 224 |
| EVENT_TRIGGER | 44 | 39 |
| EVENT_ENTITY | 41 | 38 |

### Splits

`scripts/convertir_corpus.py` builds the splits deterministically (`random.seed(42)`):

| Split | Stories | Notes |
|---|---:|---|
| Test (held out) | 155 | Digital-repository stories, selected by the domain markers listed in `es_lote9()` |
| Train (real) | 1,281 | 85 % of the remaining 1,508 stories |
| Dev | 227 | 15 % of the remaining 1,508 stories, real data only |
| Train + synthetic | 1,467 | Adds the 186 synthetic stories (train only) |
| Train after oversampling | 2,045 | Stories with an event label ×3, stories with RESTRICTION (no event label) ×2 |

### Synthetic data

The 186 synthetic stories (156 with `EVENT_TRIGGER`/`EVENT_ENTITY` spans, 51 with `RESTRICTION` spans) were drafted with Claude Opus 5 (Anthropic) in September 2026 and then reviewed and corrected by the authors, who validated every span annotation. They are used for training only and never appear in dev or test; any synthetic story matching a held-out domain marker is discarded by `convertir_corpus.py`.

## Reproducing the model

Tested on Windows 10 with an NVIDIA GeForce RTX 5070 Ti, PyTorch CUDA 12.8 wheels, CUDA 12.9 and CuPy 13.3.0, using spaCy 3 with `spacy-transformers`.

```bash
pip install spacy spacy-transformers torch   # choose the torch build that matches your CUDA version

python scripts/combinar_corpus.py     # -> data/corpus_completo.jsonl
python scripts/convertir_corpus.py    # -> data/{train,dev,test}.spacy

# Train three seeds; the article reports their score-averaged ensemble, not a single seed.
for s in 0 1 2; do
  python -m spacy train config.cfg --output ./output_seed$s \
      --paths.train ./data/train.spacy --paths.dev ./data/dev.spacy \
      --gpu-id 0 --system.seed $s
done

python scripts/ensemble_eval.py       # reproduces results/ensemble_final_results.txt
```

Trained weights are not included (RoBERTa-base checkpoints are large). Because some cuDNN kernels are non-deterministic, per-entity scores for the rare labels can vary between runs even with a fixed seed; this is why the article reports a three-seed ensemble.

## Final results

Held-out test set, three-seed ensemble, exact match:

| Entity | P | R | F1 |
|---|---:|---:|---:|
| Actor | 99.35 | 99.35 | 99.35 |
| Function | 100.00 | 98.69 | 99.34 |
| Concept | 67.74 | 58.88 | 63.00 |
| Property | 47.62 | 93.75 | 63.16 |
| Restriction | 60.00 | 33.33 | 42.86 |
| Event Entity | 100.00 | 33.33 | 50.00 |
| Event Trigger | 14.29 | 25.00 | 18.18 |
| **Overall** | **84.52** | **85.96** | **85.23** |

The test set contains only 18 Restriction, 4 Event Trigger and 3 Event Entity instances, so these three scores carry wide confidence intervals and should be read as indicative. Partial-match and validation-set results are in `results/ensemble_final_results.txt`.

## Licence

- Code (`scripts/`, `config.cfg`): MIT licence, see `LICENSE`.
- Data (`data/`): CC BY 4.0, see `data/LICENSE`. The user story texts remain attributed to Dalpiaz (2018).

## Citation

If you use this corpus or code, please cite the article (full reference to be added on publication) and the original user story collection. Machine-readable metadata is in `CITATION.cff`.

Earlier version of RENER:

> Calle Gallego, J. M., & Zapata Jaramillo, C. M. (2023). QUARE: towards a question-answering model for requirements elicitation. *Automated Software Engineering*, 30(2), 25. https://doi.org/10.1007/s10515-023-00386-w
