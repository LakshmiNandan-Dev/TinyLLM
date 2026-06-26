# TinyLLM — NL → Oracle SQL for Oracle E-Business Suite

A from-scratch (no pre-trained models) encoder-decoder that translates natural
language into Oracle SQL over the EBS schema. It ships as an on-prem, auditable
**model factory**: a customer runs the whole loop on their own iron —
**extract their catalog → train/fine-tune → serve** — and their schema and data
never leave the network.

> **Status: works end-to-end and transfers to a real EBS catalog.** Trained
> entirely from scratch on synthetic data, the model generalizes across unseen
> synthetic schemas **and** produces correct SQL on an *extracted* EBS catalog,
> which a quick customer-local fine-tune then specializes. The one piece that
> still needs a real instance to exercise is the **live Oracle adapter** (the
> read-only `oracledb` connection); its mapping/SQL are written and the logic is
> mock-tested. See [What's solid / what's left](#whats-solid--whats-left).

## The on-prem workflow (the product)

Two commands, run entirely on the customer's machine:

```bash
# 1. Extract their EBS catalog over a READ-ONLY connection -> schema.json
tinyllm extract --dsn 'readonly_user/pwd@host:1521/EBS' --out schema.json
#   roles, flexfield (segmentN) business labels, lookup domains, and the join
#   graph INFERRED from naming conventions (real EBS declares almost no FKs)

# 2. Fine-tune the shipped vendor base on THEIR schema (single GPU/CPU, minutes)
tinyllm train --schema schema.json --init base.pt --tok base_tok.json --out customer_model

# 3. Serve: NL question -> proposed SQL -> preview -> confirmed read-only run
tinyllm serve                          # FastAPI + a self-contained web UI at :8000
```

This is **split learning**: the vendor ships an opaque base trained on diverse
synthetic + EBS-realistic schemas; the customer specializes it locally on their
exact tables, lookup values, and flexfield meanings — no customer data ever
leaves, and the richest generation IP stays vendor-side.

## Results (from scratch, ~8M params, dev config d256/8h/4+4L)

**Cross-schema generalization** — trained and evaluated on *disjoint* synthetic
schemas, scored by **execution accuracy** (do predicted and gold SQL return the
same rows?), 80 unseen schemas:

| decode | execution acc | exact | un-runnable |
|---|---:|---:|---:|
| greedy | 0.825 | 0.812 | 12.5% |
| **graph-constrained** | **0.938** | 0.925 | **0.0%** |

**Real-EBS transfer** — the same model run on an *extracted* EBS catalog
(`ap_invoices_all`, `gl_code_combinations`, `vendor_id`, …):

| base | real-EBS execution acc |
|---|---|
| trained on generic/procedural names | 0 / 20 (garbled identifiers) |
| **trained on EBS-realistic names (v4)** | **10 / 20 zero-shot** |
| **+ ~2.5-min customer-local fine-tune** | **0.85 exact-match** on held-out queries |

The lesson that drove the design: real-EBS transfer is a *training-distribution*
problem, not a decoding one — once the synthetic generator emits real EBS naming
(`<module>_<entity>`, `_headers_all`/`_lines_all`, `segmentN`, `vendor_id`-style
keys), the model is in-distribution and transfers. **82 tests pass.**

## How it works

```
schema (synthetic for the base; EXTRACTED for the customer)
  → SchemaGraph        join paths via FK edges (graph owns "form")
  → QuerySampler       AST over an L1–L5 ladder (sampler owns intent)
  → render_oracle      AST → Oracle SQL (ANSI joins, EXTRACT, APPS-synonym targets)
  → render_question    AST → canonical NL + meaning-preserving paraphrases
  → validate           graph (dependency-free) + sqlglot (Oracle dialect)
  → BPE tokenizer      byte-level, from scratch (no external tokenizer libs)
  → encoder-decoder    RMSNorm · RoPE · SwiGLU · 3-way tied embeddings · dropout
  → training           cross-schema (base) or query-level (customer) split
  → retrieval+decode   link relevant tables → graph-constrained beam search
  → execution          translate to SQLite (or live Oracle) → compare result sets
```

EBS shapes are modeled directly: multi-org `_ALL`/`org_id`, flexfield (KFF)
`segmentN` with business labels, lookup-coded columns, and header/lines 2-hop
bridges (the `gl_code_combinations` join). Complexity ladder: **L1** single-table
· **L2** aggregate+GROUP BY · **L3** HAVING / top-N · **L4** nested subquery ·
**L5** window ranking.

### The schema graph is used three times (one source of truth)
- **Generation:** joins sampled by walking real FK edges → correct by construction.
- **Retrieval:** `link_tables` turns a real catalog (hundreds of tables, thousands
  of tokens) into the small training-shaped view the encoder takes — a 165-table
  catalog goes **5,638 → 106 tokens** (limit 512) at **0.95 recall**.
- **Decoding:** `SchemaPrefixGate` prunes, *as the model types*, any token that
  would commit a non-existent table/column or non-FK join key.

The runtime gate chain is the spec's safety design: model proposes → graph
validates form → `EXPLAIN` validates against the live DB → user confirms → a
**read-only** execute. The serving layer never auto-runs SQL.

Model owns intent · graph owns form · execution owns truth.

## Quick start (dev / synthetic)

```bash
pip install -e '.[model,serve]'        # sqlglot, torch, fastapi; add ',oracle' for live extract

python3 scripts/generate_data.py --n 5 --level 2          # inspect generated examples
python3 scripts/generate_data.py --n 5000 --quiet         # throughput + valid-rate
tinyllm generate --n 3 --schema                           # same via the CLI

# train the vendor base on EBS-realistic names (the recipe that transfers)
python3 scripts/train.py --train 4000 --val 300 --steps 3000 \
        --paraphrases 3 --dropout 0.1 --style ebs --device cpu

python3 scripts/exec_eval.py --n 80 --beam 5 --style ebs  # execution accuracy
python3 scripts/extract_demo.py --model                   # extract + run the model on real EBS
python3 scripts/retrieve_demo.py --n 40 --model           # retrieval on a 165-table catalog
tinyllm serve                                             # web UI + REST at :8000
pytest                                                    # full test suite
```

`--style` selects naming: `default` (generic pools) · `procedural` (near-unique,
forces schema-linking) · `ebs` (real EBS conventions — the one that transfers).

> On Apple-Silicon dev boxes use `--device cpu`: for this tiny model CPU is
> faster and far more stable than MPS, which thrashes the unified memory.

## Layout

| Path | Role |
|---|---|
| `tinyllm/schema_graph/` | schema model + `SchemaGraph` + synthetic generator (default/procedural/**ebs**) + JSON `serialize` |
| `tinyllm/sql_sampler/`  | SQL AST/IR + graph-walking sampler (L1–L5) |
| `tinyllm/render/`, `tinyllm/nl/` | AST → Oracle SQL · AST → question (template + paraphrase) |
| `tinyllm/validate/`     | graph (structural) + sqlglot (dialect) validators |
| `tinyllm/tokenizer/`, `tinyllm/model/` | from-scratch byte-level BPE · encoder-decoder |
| `tinyllm/train/`        | cross-schema + customer-local splits, training loop, checkpoints |
| `tinyllm/decode/`       | graph-constrained decoding (incremental gate + optional hard logit-mask) |
| `tinyllm/retrieve/`     | inference-time schema retrieval (question → relevant tables) |
| `tinyllm/extract/`      | EBS catalog → `Schema`: roles, flexfield/lookup meaning, FK inference; mock + `oracledb` |
| `tinyllm/eval/`         | execution-accuracy harness (SQLite stand-in DB + result-set compare) |
| `tinyllm/db/`           | runtime DB gate: `EXPLAIN`-validate + read-only execute (SqliteDb / OracleDb) |
| `tinyllm/serve/`        | `QueryService` + FastAPI (`/query`,`/execute`) + self-contained web UI |
| `tinyllm/cli.py`        | `tinyllm` console: `extract` · `train` · `serve` · `query` · `generate` |

Interfaces (`SchemaGraph`, `QuerySampler`, the catalog source, the DB connection)
are clean swap points so native/compiled or real-Oracle implementations drop in
without touching callers — the toolkit ships as auditable source.

## What's solid / what's left

**Solid (built + tested):** the from-scratch data engine, tokenizer, and model;
cross-schema training; retrieval; incremental graph-constrained decoding;
execution-accuracy eval; the EBS catalog extractor (mapping + FK inference);
**real-EBS transfer via EBS-realistic training**; the customer-local
extract→train→serve workflow with preview-confirm safety.

**Left:**
- **Live Oracle** — the `oracledb` read-only adapter + `EXPLAIN` gate are written
  and mock-tested but unexercised against a real instance.
- **Last-mile accuracy** — remaining errors are lookup-value / column-selection
  slips (not garbling); the customer fine-tune and value-constrained decoding
  close them.
- **Scale & coverage** — dev config (~8M; vs planned 55–180M); ladder stops at L5
  (no set-ops / correlated subqueries).
- **Packaging** — signed/reproducible artifacts and a third-party security audit
  for the shipped toolkit.

## License

**Proprietary — all rights reserved.** This source is public for evaluation and
reference only; it is **not** open source. Using, running, copying, modifying,
or redistributing it, or using it to build a competing product, requires a
separate written commercial license. See [LICENSE](LICENSE); for licensing
contact palla.nagendra@gmail.com.
