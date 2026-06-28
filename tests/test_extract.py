"""EBS catalog extraction: raw data-dictionary rows -> a semantic, join-aware
Schema. The hard part is inferring FKs (EBS declares none) and flexfield/lookup
meaning, and producing a Schema the rest of the pipeline consumes unchanged."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tinyllm.extract import EbsExtractor, MockCatalog  # noqa: E402
from tinyllm.schema_graph import SchemaGraph  # noqa: E402
from tinyllm.schema_graph.types import SemanticRole  # noqa: E402
from tinyllm.sql_sampler import QuerySampler  # noqa: E402
from tinyllm.validate import validate_graph  # noqa: E402


def _schema():
    return EbsExtractor(MockCatalog()).extract()


def test_tables_and_multi_org():
    s = _schema()
    assert set(s.table_names) == {
        "ap_suppliers", "ap_invoices_all", "ap_invoice_lines_all", "gl_code_combinations"
    }
    assert s.table("ap_invoices_all").is_multi_org is True       # _ALL + org_id
    assert s.table("ap_invoice_lines_all").is_multi_org is True
    assert s.table("ap_suppliers").is_multi_org is False
    assert s.table("gl_code_combinations").is_multi_org is False


def test_foreign_keys_inferred_from_conventions():
    s = _schema()
    edges = {(fk.from_table, fk.from_column, fk.to_table, fk.to_column)
             for fk in s.foreign_keys}
    # MockCatalog declares ZERO FKs; all of these are inferred by naming convention
    assert ("ap_invoices_all", "vendor_id", "ap_suppliers", "vendor_id") in edges
    assert ("ap_invoice_lines_all", "invoice_id", "ap_invoices_all", "invoice_id") in edges
    assert ("ap_invoice_lines_all", "code_combination_id",
            "gl_code_combinations", "code_combination_id") in edges
    # org_id is striping, NOT a foreign key
    assert not any(fk.from_column == "org_id" for fk in s.foreign_keys)


def test_semantic_roles():
    s = _schema()
    inv = s.table("ap_invoices_all")
    assert inv.column("invoice_id").role == SemanticRole.ID and inv.column("invoice_id").is_pk
    assert inv.column("invoice_amount").role == SemanticRole.AMOUNT
    assert inv.column("invoice_date").role == SemanticRole.DATE
    assert inv.column("org_id").role == SemanticRole.ORG_ID
    assert s.table("ap_suppliers").column("vendor_name").role == SemanticRole.NAME


def test_flexfield_and_lookup_metadata():
    s = _schema()
    seg2 = s.table("gl_code_combinations").column("segment2")
    assert seg2.role == SemanticRole.FLEXFIELD_SEGMENT
    assert seg2.business_label == "cost center"      # customer-specific mapping
    lk = s.table("ap_invoices_all").column("invoice_type_lookup_code")
    assert lk.role == SemanticRole.LOOKUP and lk.lookup_type == "INVOICE TYPE"
    assert "STANDARD" in lk.allowed_values


def test_extracted_schema_flows_through_pipeline():
    """The whole point: an extracted real-EBS-shaped schema is a drop-in for the
    synthetic one -- graph joins resolve and the sampler builds valid SQL."""
    s = _schema()
    g = SchemaGraph(s)
    # header -> lines -> code_combinations bridge resolves
    path = g.join_path("ap_invoice_lines_all", "gl_code_combinations")
    assert path is not None and len(path) == 1
    import random
    for level in (1, 2, 3):
        ast, _ = QuerySampler(g, random.Random(0)).sample(level)
        assert validate_graph(ast, g).ok


# -- live OracleCatalog (bulk reads, no Oracle): a fake cursor returns canned
#    set-based rows so the grouping + caching is proven without a database -------
_TABLES = [                                  # (synonym_name, table_owner, table_name)
    ("AP_INVOICES_ALL", "AP", "AP_INVOICES_ALL"),
    ("AP_SUPPLIERS", "AP", "PO_VENDORS"),    # synonym name != base table (APPS layer)
    ("GL_CODE_COMBINATIONS", "GL", "GL_CODE_COMBINATIONS"),
]
_COLS = [                                    # (synonym_name, column, type, nullable)
    ("AP_INVOICES_ALL", "INVOICE_ID", "NUMBER", "N"),
    ("AP_INVOICES_ALL", "VENDOR_ID", "NUMBER", "Y"),
    ("AP_INVOICES_ALL", "INVOICE_AMOUNT", "NUMBER", "Y"),
    ("AP_SUPPLIERS", "VENDOR_ID", "NUMBER", "N"),
    ("AP_SUPPLIERS", "VENDOR_NAME", "VARCHAR2", "Y"),
    ("GL_CODE_COMBINATIONS", "CODE_COMBINATION_ID", "NUMBER", "N"),
    ("GL_CODE_COMBINATIONS", "SEGMENT1", "VARCHAR2", "Y"),
    ("ZZ_VIEW_SYNONYM", "X", "NUMBER", "Y"),  # synonym not in the table list -> dropped
]
_PKS = [
    ("AP_INVOICES_ALL", "INVOICE_ID"),
    ("AP_SUPPLIERS", "VENDOR_ID"),
    ("GL_CODE_COMBINATIONS", "CODE_COMBINATION_ID"),
]


class _FakeCursor:
    """Routes each OracleCatalog bulk query to canned rows by a distinctive token."""

    def __init__(self):
        self._rows: list = []
        self.table_query_count = 0

    def execute(self, sql, bind=None):
        if "all_tab_columns" in sql:
            self._rows = _COLS
        elif "constraint_type = 'P'" in sql:
            self._rows = _PKS
        elif "constraint_type = 'R'" in sql:
            self._rows = []                  # EBS declares no FKs
        elif "all_tables t" in sql:          # the table-list query
            self.table_query_count += 1
            self._rows = _TABLES
        else:
            self._rows = []

    def fetchall(self):
        return self._rows


def test_oracle_catalog_bulk_grouping_and_caching():
    from tinyllm.extract.catalog import OracleCatalog
    cur = _FakeCursor()
    cat = OracleCatalog(cur)

    # synonym names are the canonical (lowercased) table names
    assert cat.tables() == ["ap_invoices_all", "ap_suppliers", "gl_code_combinations"]
    # columns grouped by synonym, lowercased, nullable decoded
    cols = {c.name: c for c in cat.columns("ap_invoices_all")}
    assert set(cols) == {"invoice_id", "vendor_id", "invoice_amount"}
    assert cols["invoice_id"].nullable is False and cols["vendor_id"].nullable is True
    assert cat.primary_key("gl_code_combinations") == ["code_combination_id"]
    # a synonym that appears in columns but not in the table list is ignored
    assert "zz_view_synonym" not in cat.tables()
    # bulk read happens ONCE and is cached -- not re-queried per accessor call
    cat.columns("ap_suppliers"); cat.primary_key("ap_invoices_all"); cat.tables()
    assert cur.table_query_count == 1


def test_oracle_catalog_flows_through_extractor():
    """End to end on the bulk adapter: extract() infers the join graph from the
    bulk-read metadata exactly as it does for the mock."""
    from tinyllm.extract.catalog import OracleCatalog
    s = EbsExtractor(OracleCatalog(_FakeCursor())).extract()
    assert set(s.table_names) == {"ap_invoices_all", "ap_suppliers", "gl_code_combinations"}
    edges = {(fk.from_table, fk.from_column, fk.to_table, fk.to_column)
             for fk in s.foreign_keys}
    assert ("ap_invoices_all", "vendor_id", "ap_suppliers", "vendor_id") in edges
