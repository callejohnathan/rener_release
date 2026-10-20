# -*- coding: utf-8 -*-
"""
Ensemble evaluation for the final RENER model.

Averages raw per-candidate spancat confidence scores across three
independently seeded training runs before applying the classification
threshold (0.5), rather than voting on each model's already-thresholded
output. This is more robust than any single seed for the low-frequency
entity types (RESTRICTION, EVENT_TRIGGER, EVENT_ENTITY), whose validation
and test support is small enough that a single run is not a reliable
basis for comparison (see the paper, section "Data Augmentation and
Multi-Seed Robustness").

Prerequisites:
  1. Build data/train.spacy, data/dev.spacy, data/test.spacy with
     scripts/convertir_corpus.py (which merges data/synthetic/*.jsonl into
     train only, and applies class-aware oversampling).
  2. Train three models with identical config.cfg but --system.seed 0, 1, 2,
     saving each to the paths listed in MODEL_PATHS below (or edit the list
     to match wherever you saved them):
       python -m spacy train config.cfg --output ./output_seed0 \
           --paths.train ./data/train.spacy --paths.dev ./data/dev.spacy \
           --gpu-id 0
       (repeat with --system.seed 1 --output ./output_seed1, etc.)

Evaluates both exact match and partial/token-overlap match, on dev and test.
"""
import spacy
import numpy as np
from spacy.tokens import DocBin

MODEL_PATHS = [
    "output_seed0/model-best",
    "output_seed1/model-best",
    "output_seed2/model-best",
]
LABELS_OF_INTEREST = ["ACTOR", "FUNCTION", "CONCEPT", "PROPERTY",
                       "RESTRICTION", "EVENT_TRIGGER", "EVENT_ENTITY"]


def load_models():
    return [spacy.load(p) for p in MODEL_PATHS]


def ensemble_predict(nlps, texts):
    """Returns, for each text, a dict label -> set of (start_char, end_char) predicted by the ensemble."""
    spancats = [nlp.get_pipe("spancat") for nlp in nlps]
    label_idx = {l: i for i, l in enumerate(spancats[0].labels)}
    assert all(sp.labels == spancats[0].labels for sp in spancats), "models have different label sets"

    all_docs = []
    all_scores = []
    for nlp, spancat in zip(nlps, spancats):
        # run the full pipeline up to (not including) spancat so the
        # transformer listener cache is populated before predict()
        docs = list(nlp.pipe(texts, disable=["spancat"]))
        indices, scores = spancat.predict(docs)
        scores = spancat.model.ops.to_numpy(scores)
        all_docs.append(docs)
        all_scores.append((indices, scores))

    n_docs = len(texts)
    results = []
    for i in range(n_docs):
        # candidate spans are identical across models: same suggester, same text
        idx0 = all_scores[0][0][i].dataXd
        idx0 = spancats[0].model.ops.to_numpy(idx0)
        n_cand = all_scores[0][0].lengths[i]

        offsets = [sum(sc[0].lengths[j] for j in range(i)) for sc in all_scores]
        score_blocks = [scores[offsets[m]:offsets[m] + n_cand]
                         for m, (indices, scores) in enumerate(all_scores)]
        avg_scores = np.mean(score_blocks, axis=0)  # (n_cand, n_labels)

        doc = all_docs[0][i]
        pred = {lbl: set() for lbl in LABELS_OF_INTEREST}
        for k in range(n_cand):
            tok_s, tok_e = idx0[k]
            span = doc[tok_s:tok_e]
            for lbl in LABELS_OF_INTEREST:
                j = label_idx[lbl]
                if avg_scores[k, j] >= 0.5:
                    pred[lbl].add((span.start_char, span.end_char))
        results.append(pred)
    return results


def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def score_exact(preds, golds, label):
    tp = fp = fn = 0
    for p, g in zip(preds, golds):
        pset, gset = p[label], g[label]
        tp += len(pset & gset)
        fp += len(pset - gset)
        fn += len(gset - pset)
    prec = tp / (tp + fp) * 100 if (tp + fp) else 0.0
    rec = tp / (tp + fn) * 100 if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return tp, fp, fn, prec, rec, f1


def score_partial(preds, golds, label):
    tp = fp = fn = 0
    for p, g in zip(preds, golds):
        pset, gset = list(p[label]), list(g[label])
        matched_g, matched_p = set(), set()
        for gi, gs in enumerate(gset):
            for pi, ps in enumerate(pset):
                if overlaps(gs, ps):
                    matched_g.add(gi)
                    matched_p.add(pi)
        tp += len(matched_g)
        fn += len(gset) - len(matched_g)
        fp += len(pset) - len(matched_p)
    prec = tp / (tp + fp) * 100 if (tp + fp) else 0.0
    rec = tp / (tp + fn) * 100 if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return tp, fp, fn, prec, rec, f1


def run(split_path, split_name):
    nlp_blank = spacy.blank("en")
    db = DocBin().from_disk(split_path)
    gold_docs = list(db.get_docs(nlp_blank.vocab))
    texts = [d.text for d in gold_docs]

    golds = []
    for gdoc in gold_docs:
        g = {lbl: set() for lbl in LABELS_OF_INTEREST}
        for s in gdoc.spans.get("sc", []):
            if s.label_ in g:
                g[s.label_].add((s.start_char, s.end_char))
        golds.append(g)

    nlps = load_models()
    preds = ensemble_predict(nlps, texts)

    for match_name, score_fn in [("EXACT MATCH", score_exact), ("PARTIAL/OVERLAP MATCH", score_partial)]:
        print(f"\n{'=' * 70}\nENSEMBLE - {split_name} - {match_name}\n{'=' * 70}")
        print(f"{'Label':15s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'P':>6s} {'R':>6s} {'F1':>6s}")
        overall_tp = overall_fp = overall_fn = 0
        for lbl in LABELS_OF_INTEREST:
            tp, fp, fn, p, r, f = score_fn(preds, golds, lbl)
            overall_tp += tp
            overall_fp += fp
            overall_fn += fn
            print(f"{lbl:15s} {tp:4d} {fp:4d} {fn:4d} {p:6.2f} {r:6.2f} {f:6.2f}")
        op = overall_tp / (overall_tp + overall_fp) * 100 if (overall_tp + overall_fp) else 0
        orc = overall_tp / (overall_tp + overall_fn) * 100 if (overall_tp + overall_fn) else 0
        of1 = 2 * op * orc / (op + orc) if (op + orc) else 0
        print(f"{'OVERALL':15s} {overall_tp:4d} {overall_fp:4d} {overall_fn:4d} {op:6.2f} {orc:6.2f} {of1:6.2f}")


if __name__ == "__main__":
    run("data/test.spacy", "TEST")
    run("data/dev.spacy", "DEV")
