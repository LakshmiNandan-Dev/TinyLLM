# TinyLLM — NL → Oracle SQL for Oracle E-Business Suite

A from-scratch (no pre-trained models) encoder-decoder that translates natural
language into Oracle SQL over the EBS schema. Built to ship as an on-prem,
auditable **model factory**: customers run the pipeline on their own iron, their
schema and data never leave.

> **Status: a complete, working prototype — on synthetic data.** The full ML
> loop is built from scratch and measured end to end. It has **not yet touched a
> real EBS schema or a live Oracle database** — see
> [Prototype vs. product](#prototype-vs-product) for the honest boundary.

## Headline result

A **7.9M-parameter** from-scratch model (dev config: d256, 8 heads, 4+4 layers),
trained with a **cross-schema split** (it is evaluated on synthetic schemas it
never saw), scored by **execution accuracy** (do the predicted and gold SQL
return the same rows?):

| decode mode | execution accuracy | exact-match | un-runnable |
|---|---:|---:|---:|
| greedy | 0.825 | 0.812 | 12.5% |
| **graph-constrained (PICARD-style)** | **0.938** | 0.925 | **0.0%** |

So on schemas it has never seen, the model is correct **93.8%** of the time with
graph-constrained decoding, and emits **zero un-runnable queries**. (80 unseen
schemas; see `scripts/exec_eval.py`.) **50 tests pass.**

## The pipeline, built from scratch

```
schema (synthetic)
  → SchemaGraph        join paths via FK edges (graph owns "form")
  → QuerySampler       AST/IR over an L1–L5 complexity ladder (sampler owns intent)
  → render_oracle      AST → Oracle SQL (ANSI joins, EXTRACT, APPS-synonym targets)
  → render_question    AST → canonical NL + paraphrases (varied, meaning-preserving)
  → validate           graph (structural, dependency-free) + sqlglot (Oracle dialect)
  → BPE tokenizer      byte-level, trained on the corpus (no external tokenizer libs)
  → encoder-decoder    RMSNorm · RoPE · SwiGLU · 3-way tied embeddings · dropout
  → training           cross-schema split, warmup+cosine, best-by-exact checkpoint
  → decoding           graph-constrained beam search (model = intent, graph = form)
  → execution eval     translate to SQLite, run gold vs pred, compare result sets
```

The training triples `(schema, question, SQL)` are SQL-first, so queries are
correct **by construction**. EBS shapes are modeled directly: multi-org
`_ALL`/`org_id`, flexfield (KFF) `segmentN` with business labels, lookup-coded
columns, and header/lines 2-hop bridges (the `gl_code_combinations`-style join).

Complexity ladder: **L1** single-table · **L2** aggregate + GROUP BY ·
**L3** HAVING / top-N (`FETCH FIRST`) · **L4** nested subquery · **L5** window
ranking (`RANK() OVER`).

## The graph is used three times

One source of truth, used at generation, retrieval, and decoding:

- **Generation:** joins are sampled by walking real FK edges → correct by construction.
- **Retrieval:** at inference, `link_tables` matches the question's words to
  table/column labels and returns the FK-connected module they live in — turning
  a real EBS catalog (hundreds of tables, thousands of tokens) into the small,
  training-shaped view the encoder can take. On a 165-table catalog that's
  **5,638 → 106 encoder tokens** (the model's limit is 512), at **0.95 recall**.
- **Decoding:** the **incremental gate** (`SchemaPrefixGate`) prunes, *as the
  model types*, any token that would commit a non-existent table/column or a
  non-FK join key — so the model can only ever complete real identifiers.

Model owns intent · graph owns form · execution owns truth.

## Quick start

```bash
pip install -r requirements.txt          # sqlglot, torch, pytest

python3 scripts/generate_data.py --n 5 --level 2          # inspect examples
python3 scripts/generate_data.py --n 3 --paraphrases 4    # NL variants per query
python3 scripts/generate_data.py --n 5000 --quiet         # throughput + valid-rate
python3 scripts/generate_data.py --demo-repair            # graph rejects a fabricated join

python3 scripts/train_tokenizer.py --examples 3000        # from-scratch BPE tokenizer
python3 scripts/overfit.py --n 24 --steps 400             # prove the model learns

# train + eval on UNSEEN schemas (the current best recipe; ~16k pairs, dropout)
python3 scripts/train.py --train 4000 --val 300 --steps 3000 \
        --paraphrases 3 --dropout 0.1 --device cpu

python3 scripts/constrained_demo.py --n 80 --beam 5       # greedy vs verify vs picard
python3 scripts/exec_eval.py --n 80 --beam 5              # EXECUTION accuracy (the real metric)
python3 scripts/retrieve_demo.py --n 40 --model          # schema retrieval on a 165-table catalog
pytest                                                    # full test suite
```

The graph validator has **no dependencies** and always runs; sqlglot is the
Oracle-dialect gate and the eval's SQLite transpiler.

> On Apple-Silicon dev boxes use `--device cpu`: for this tiny model, CPU is
> faster and far more stable than MPS, which thrashes the unified memory.

## Pushing accuracy: data + regularization

Exact-match plateaued and **overfit** (val loss rose) at the first recipe. Adding
**dropout** and **~1.8× more diverse data** killed the overfitting (val loss kept
falling) and lifted unseen-schema execution accuracy from **0.850 → 0.938**.

An **opt-in procedural-name mode** (`--procedural`) goes further: it gives every
synthetic schema near-unique entity/doc names, dropping train∩val table-name
overlap from ~86% to ~0.5%, so the model **must** link the question to the
serialized schema instead of memorizing a small noun vocabulary. (Built and
tested; the deeper schema-linking run is the next experiment.)

## Layout

| Path | Role |
|---|---|
| `tinyllm/schema_graph/` | schema data model + `SchemaGraph` (join resolution) + synthetic generator |
| `tinyllm/sql_sampler/`  | SQL AST/IR + graph-walking sampler (L1–L5 ladder) |
| `tinyllm/render/`       | AST → Oracle SQL |
| `tinyllm/nl/`           | AST → question: canonical template + structure-aware paraphrase |
| `tinyllm/validate/`     | graph (structural) + sqlglot (dialect) validators |
| `tinyllm/tokenizer/`    | from-scratch byte-level BPE trained on the corpus |
| `tinyllm/model/`        | from-scratch encoder-decoder (`config`/`transformer`/`collate`) |
| `tinyllm/train/`        | cross-schema split, training loop, eval, checkpoints |
| `tinyllm/decode/`       | graph-constrained decoding: beam search + incremental `SchemaPrefixGate` |
| `tinyllm/retrieve/`     | inference-time schema retrieval: question → relevant FK-connected table subset |
| `tinyllm/extract/`      | EBS catalog → `Schema`: roles, flexfield/lookup meaning, FK inference from naming |
| `tinyllm/eval/`         | execution-accuracy harness (synthetic SQLite DB + result-set compare) |
| `tinyllm/pipeline.py`   | ties it together → `Example`; schema serialization for the encoder |
| `scripts/`              | data gen, tokenizer/model training, decode + execution demos |
| `tests/`                | validity, graph-grounding, gate, decode, and eval tests |

The `SchemaGraph`, `QuerySampler`, and semantic-mapping pieces sit behind clean
interfaces so they can later be accelerated (rustworkx/Cython) without touching
callers — performance only; the toolkit ships as auditable source.

## Prototype vs. product

Everything above is trained and evaluated **entirely on synthetic, EBS-shaped
schemas**. The 93.8% is on unseen *synthetic* schemas — a genuine
cross-schema-generalization result, but **not** real NL→SQL over a real EBS
instance. The bridge to real data is the next phase and is **not yet built**:

- **EBS catalog extractor** — the *mapping* is built and tested against a mock
  data dictionary (`tinyllm/extract/`): semantic roles, flexfield/lookup meaning,
  and the join graph **inferred from naming conventions** (real EBS declares
  almost no FKs). Only the live-Oracle adapter (`OracleCatalog`, the documented
  `ALL_*`/`FND_*` SQL) is unrun here — it needs a real instance.
- **Model transfer to novel vocabulary** — running the trained model on the
  extracted EBS schema shows it currently emits *synthetic* training names rather
  than copying the real ones from the encoder (the gate correctly rejects them).
  The 93.8% is partly inflated by train/val name overlap; true real-EBS transfer
  needs the **procedural-name run** (`--procedural`, built) so the model learns
  to schema-link, not memorize.
- **Real-Oracle gate** — `EXPLAIN PLAN` / read-only execute against a live DB
  (today the eval translates to SQLite).
- **Serving + UI** — no inference server (`serve/`) or web app (`web/`) yet.
- **Scale & coverage** — this is the 7.9M dev config (vs. the planned 55–180M);
  the SQL ladder stops at L5 (no set-ops / correlated subqueries).
- **Deployment** — the "model factory" packaging, split-learning (vendor base +
  customer-local fine-tune), and signed/reproducible artifacts.

## Status summary

- **Done & measured:** data engine · from-scratch tokenizer + encoder-decoder ·
  cross-schema training · incremental graph-constrained decoding · execution
  accuracy eval (93.8% on unseen synthetic schemas) · **schema retrieval**
  (165-table catalog → ~4 tables, 0.95 recall) · **EBS catalog extractor**
  (mock-tested; roles + flexfield/lookup + FK inference).
- **Next:** the **procedural-name training run** — the extractor demo shows it's
  the prerequisite for real-EBS transfer (the model must schema-link, not
  memorize). Then: live-Oracle adapter + `EXPLAIN` gate → serving.
