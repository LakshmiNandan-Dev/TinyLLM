"""Tests for the data-engine slice: validity, graph-grounding, and the
inference-time graph gate (corrupted joins must be rejected)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tinyllm import generate_example  # noqa: E402
from tinyllm.schema_graph import SchemaGraph  # noqa: E402
from tinyllm.sql_sampler.ast import ColumnRef  # noqa: E402
from tinyllm.validate import validate_graph, validate_sqlglot  # noqa: E402

SEEDS = range(150)
LEVELS = (1, 2, 3, 4, 5)


@pytest.mark.parametrize("level", LEVELS)
def test_examples_are_valid(level):
    for seed in SEEDS:
        ex = generate_example(seed, level=level)
        assert ex.valid, (level, ex.sql, [v.issues for v in ex.validations])


@pytest.mark.parametrize("level", LEVELS)
def test_graph_validation_passes_by_construction(level):
    for seed in SEEDS:
        ex = generate_example(seed, level=level)
        res = validate_graph(ex.ast, SchemaGraph(ex.schema))
        assert res.ok, (level, res.issues)


@pytest.mark.parametrize("level", LEVELS)
def test_every_join_is_a_real_fk_edge(level):
    for seed in SEEDS:
        ex = generate_example(seed, level=level)
        graph = SchemaGraph(ex.schema)
        for join in ex.ast.joins:
            for left, right in join.on:
                assert graph.fk_between(left.table, right.table) is not None


@pytest.mark.skipif(
    importlib.util.find_spec("sqlglot") is None, reason="sqlglot not installed"
)
@pytest.mark.parametrize("level", LEVELS)
def test_sql_parses_in_oracle_dialect(level):
    for seed in SEEDS:
        ex = generate_example(seed, level=level)
        res = validate_sqlglot(ex.sql)
        assert res.ok, (level, ex.sql, res.issues)


@pytest.mark.parametrize("level", LEVELS)
def test_paraphrases_preserve_sql_and_vary(level):
    distinct: set[str] = set()
    for seed in range(60):
        ex = generate_example(seed, level=level, n_paraphrases=4)
        assert ex.paraphrases, (level, "no paraphrases produced")
        for p in ex.paraphrases:
            assert p and p == p.strip()
            assert "?" not in p[:-1], f"mid-sentence '?': {p!r}"
        # paraphrasing only varies surface form -- the gold SQL must not change
        assert {sql for _, sql in ex.training_pairs()} == {ex.sql}
        distinct.update(ex.paraphrases)
    assert len(distinct) > 60, f"too little variety at L{level}: {len(distinct)}"


@pytest.mark.parametrize("level", LEVELS)
def test_procedural_examples_are_valid(level):
    # opt-in procedural names must still yield valid, graph-grounded SQL
    for seed in range(60):
        ex = generate_example(seed, level=level, procedural=True)
        assert ex.valid, (level, ex.sql, [v.issues for v in ex.validations])
        assert validate_graph(ex.ast, SchemaGraph(ex.schema)).ok


def test_procedural_breaks_train_val_name_overlap():
    """The whole point: procedural names make train/val schemas share almost no
    table names, so the model must schema-link, not memorize a tiny vocabulary."""
    def names(seeds, procedural):
        out = set()
        for s in seeds:
            out.update(generate_example(s, level=2, procedural=procedural).schema.table_names)
        return out

    train, val = range(200), range(1_000_000, 1_000_200)
    overlap_default = len(names(train, False) & names(val, False))
    overlap_proc = len(names(train, True) & names(val, True))
    assert overlap_default > 50            # default: heavy shared vocabulary
    assert overlap_proc <= 5               # procedural: near-disjoint


def test_corrupted_join_is_rejected():
    """The graph gate must FAIL a join that doesn't follow the documented FK."""
    ex = next(generate_example(s) for s in SEEDS if generate_example(s).ast.joins)
    graph = SchemaGraph(ex.schema)
    assert validate_graph(ex.ast, graph).ok  # sanity: starts valid

    left, right = ex.ast.joins[0].on[0]
    right_table = ex.schema.table(right.table)
    wrong_col = next(c for c in right_table.columns if c.name != right.column.name)
    ex.ast.joins[0].on[0] = (left, ColumnRef(right.table, wrong_col))

    result = validate_graph(ex.ast, graph)
    assert result.ok is False
    assert any("FK" in issue for issue in result.issues)
