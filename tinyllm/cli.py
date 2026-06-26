"""`tinyllm` console entry point (installed via `pip install tinyllm`).

    tinyllm serve                       # run the NL->SQL service + web UI
    tinyllm query "top 5 vendors by amount" --schema ebs_ap_gl
    tinyllm generate --n 5 --level 2    # inspect synthetic training examples

The default schema registry is a demo (extracted mock EBS + one synthetic
schema) with SQLite stand-in databases. In a real deployment you would build the
registry from `EbsExtractor(OracleCatalog(cursor))` and wire `OracleDb(cursor)`.
"""

from __future__ import annotations

import argparse


def _demo_registry(with_dbs: bool = True):
    from . import generate_example
    from .extract import EbsExtractor, MockCatalog
    schemas = {
        "ebs_ap_gl": EbsExtractor(MockCatalog()).extract(),
        "synthetic_demo": generate_example(1_000_000, level=2).schema,
    }
    dbs = None
    if with_dbs:
        from .db import SqliteDb
        dbs = {sid: SqliteDb(s, seed=7) for sid, s in schemas.items()}
    return schemas, dbs


def _serve(args):
    from .serve import QueryService, build_app
    schemas, dbs = _demo_registry()
    service = QueryService.from_files(args.ckpt, args.tok, schemas, dbs=dbs)
    print(f"serving {service.schema_ids()} on http://{args.host}:{args.port}")
    import uvicorn
    uvicorn.run(build_app(service), host=args.host, port=args.port)


def _query(args):
    from .serve import QueryService
    schemas, dbs = _demo_registry(with_dbs=not args.no_execute)
    service = QueryService.from_files(args.ckpt, args.tok, schemas, dbs=dbs)
    res = service.query(args.question, args.schema)
    print(res.sql)
    print(f"# graph_valid={res.graph_valid} explain_ok={res.explain_ok} "
          f"tables={res.tables_used}")
    if not args.no_execute and res.graph_valid and res.explain_ok:
        rows = service.execute(args.schema, res.sql, max_rows=args.max_rows)
        print(f"# {len(rows)} row(s):")
        for r in rows:
            print("  ", r)


def _generate(args):
    from . import generate_example, serialize_schema
    for i in range(args.n):
        ex = generate_example(args.seed + i, level=args.level)
        print(f"\nQ: {ex.question}\nSQL: {' '.join(ex.sql.split())}")
        if args.schema:
            print(f"schema: {serialize_schema(ex.schema, ex.ast.tables)}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="tinyllm", description="NL -> Oracle EBS SQL toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the service + web UI")
    s.add_argument("--ckpt", default="artifacts/model_best.pt")
    s.add_argument("--tok", default="artifacts/tokenizer.json")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=_serve)

    q = sub.add_parser("query", help="one-shot NL -> SQL")
    q.add_argument("question")
    q.add_argument("--schema", default="ebs_ap_gl")
    q.add_argument("--ckpt", default="artifacts/model_best.pt")
    q.add_argument("--tok", default="artifacts/tokenizer.json")
    q.add_argument("--no-execute", action="store_true", help="propose only; do not run")
    q.add_argument("--max-rows", type=int, default=20)
    q.set_defaults(func=_query)

    g = sub.add_parser("generate", help="inspect synthetic examples")
    g.add_argument("--n", type=int, default=5)
    g.add_argument("--level", type=int, default=2)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--schema", action="store_true", help="also print the serialized schema")
    g.set_defaults(func=_generate)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
