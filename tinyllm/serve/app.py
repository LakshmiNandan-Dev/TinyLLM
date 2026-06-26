"""Thin FastAPI wrapper over QueryService (optional dependency).

    POST /query     {question, schema_id} -> proposed SQL (preview, not executed)
    GET  /schemas   available schema ids
    GET  /health

FastAPI is an optional extra; importing this module without it raises a clear
error. All real work is in `service.py`, which is server-free and tested there.
"""

from .service import QueryService


def build_app(service: QueryService):
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import HTMLResponse
        from pydantic import BaseModel
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("serving needs fastapi+pydantic: pip install 'tinyllm[serve]'") from e

    from pathlib import Path
    _INDEX = Path(__file__).resolve().parent / "static" / "index.html"

    class QueryRequest(BaseModel):
        question: str
        schema_id: str

    class ExecuteRequest(BaseModel):
        schema_id: str
        sql: str                      # the user-confirmed SQL from a prior /query
        max_rows: int = 50

    app = FastAPI(title="TinyLLM NL→Oracle SQL", version="0.1")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return _INDEX.read_text(encoding="utf-8")

    @app.get("/health")
    def health():
        return {"status": "ok", "schemas": service.schema_ids()}

    @app.get("/schemas")
    def schemas():
        return {"schemas": service.schema_ids()}

    @app.post("/query")
    def query(req: QueryRequest):
        """Propose SQL for confirmation -- does NOT execute."""
        try:
            return service.query(req.question, req.schema_id).to_dict()
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.post("/execute")
    def execute(req: ExecuteRequest):
        """Run the user-confirmed SELECT read-only against the live DB."""
        try:
            rows = service.execute(req.schema_id, req.sql, max_rows=req.max_rows)
            return {"schema_id": req.schema_id, "row_count": len(rows), "rows": rows}
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except (ValueError, RuntimeError) as e:
            raise HTTPException(status_code=400, detail=str(e))

    return app
