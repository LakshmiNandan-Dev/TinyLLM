"""Build train / val splits on DISJOINT schema seeds (Spider-style).

Each seed produces a different synthetic schema, so disjoint seed ranges =>
disjoint schemas. Val therefore measures generalization to schemas never seen
in training -- the whole commercial premise.

Training pairs include paraphrases (more NL variety); val uses the canonical
question only, one per example, for a clean, stable metric.
"""

from __future__ import annotations

from .. import generate_example, serialize_schema

LEVELS = (1, 2, 3, 4, 5)
VAL_OFFSET = 1_000_000          # keep val seeds far from train -> disjoint schemas


def build_pairs(seeds, paraphrases=0, canonical_only=False, procedural=False):
    pairs: list[tuple[str, str, str]] = []
    for i, seed in enumerate(seeds):
        ex = generate_example(seed, level=LEVELS[i % len(LEVELS)],
                              n_paraphrases=paraphrases, procedural=procedural)
        schema_str = serialize_schema(ex.schema)
        if canonical_only:
            pairs.append((ex.question, schema_str, ex.sql))
        else:
            for question, sql in ex.training_pairs():
                pairs.append((question, schema_str, sql))
    return pairs


def make_split(n_train: int, n_val: int, paraphrases: int = 2, procedural: bool = False):
    train = build_pairs(range(n_train), paraphrases=paraphrases, procedural=procedural)
    val = build_pairs(range(VAL_OFFSET, VAL_OFFSET + n_val),
                      canonical_only=True, procedural=procedural)
    return train, val


def corpus_texts(pairs):
    """Flatten pairs into texts for tokenizer training (train split only)."""
    texts: list[str] = []
    for question, schema_str, sql in pairs:
        texts.extend((question, schema_str, sql))
    return texts
