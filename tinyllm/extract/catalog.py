"""Catalog source: where raw EBS metadata comes from.

`CatalogSource` is the seam between *reading* an Oracle instance (read-only, via
the data dictionary) and the *mapping* logic in `extractor.py`. The real adapter
(`OracleCatalog`) issues the documented SQL below against `ALL_*`/`FND_*`; the
`MockCatalog` returns canned rows for the same shape, so the extractor's logic is
fully testable with no database. Nothing here leaves the customer network.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RawColumn:
    name: str
    data_type: str
    nullable: bool = True


@dataclass
class RawFk:
    from_table: str
    from_column: str
    to_table: str
    to_column: str


@dataclass
class RawFlex:
    table: str
    column: str
    business_label: str          # from FND_ID_FLEX_SEGMENTS (CUSTOMER-SPECIFIC)


@dataclass
class RawLookup:
    table: str
    column: str
    lookup_type: str
    values: tuple = ()


class CatalogSource:
    """Interface every catalog reader implements (real Oracle or mock)."""

    def tables(self) -> list[str]: raise NotImplementedError
    def columns(self, table: str) -> list[RawColumn]: raise NotImplementedError
    def primary_key(self, table: str) -> list[str]: raise NotImplementedError
    def foreign_keys(self) -> list[RawFk]: return []          # EBS often declares none
    def flex_segments(self) -> list[RawFlex]: return []
    def lookups(self) -> list[RawLookup]: return []


# -- the real adapter's SQL (documented; run by OracleCatalog) ----------------
# Read-only, returns no business data -- only the data dictionary + FND setup.
#
# Scope = ALL tables in ALL schemas that are LICENSED or SHARED-LICENSE enabled:
# every product whose fnd_product_installations.status is 'I' (Installed /
# licensed) or 'S' (Shared -- shared-license enabled), reached through the APPS
# synonym layer. Status 'N' (not installed) is excluded. This is a full-EBS
# extract (AP, AR, GL, PO, INV, ONT, HR/PER, FA, ...), not a per-module subset.
#
# Reads are BULK / set-based (a handful of queries for the whole instance, NOT
# one-per-table) so a real ~20k-table footprint is actually feasible to pull.
_LICENSED_OWNERS = """
    SELECT u.oracle_username
      FROM fnd_product_installations i
      JOIN fnd_oracle_userid u ON u.oracle_id = i.oracle_id
     WHERE i.status IN ('I', 'S')
"""

ORACLE_SQL = {
    # one row per base TABLE that APPS exposes, across every licensed/shared schema
    "tables": f"""
        SELECT s.synonym_name, s.table_owner, s.table_name
          FROM all_synonyms s
         WHERE s.owner = 'APPS'
           AND s.table_owner IN ({_LICENSED_OWNERS})
           AND EXISTS (SELECT 1 FROM all_tables t
                        WHERE t.owner = s.table_owner
                          AND t.table_name = s.table_name)
    """,
    # ALL columns for those tables in one shot, keyed by the APPS synonym name
    "columns": f"""
        SELECT s.synonym_name, c.column_name, c.data_type, c.nullable
          FROM all_synonyms s
          JOIN all_tab_columns c
            ON c.owner = s.table_owner AND c.table_name = s.table_name
         WHERE s.owner = 'APPS'
           AND s.table_owner IN ({_LICENSED_OWNERS})
         ORDER BY s.synonym_name, c.column_id
    """,
    # ALL primary-key columns in one shot, keyed by synonym name
    "primary_key": f"""
        SELECT s.synonym_name, cc.column_name
          FROM all_synonyms s
          JOIN all_constraints con
            ON con.owner = s.table_owner AND con.table_name = s.table_name
           AND con.constraint_type = 'P'
          JOIN all_cons_columns cc
            ON cc.owner = con.owner AND cc.constraint_name = con.constraint_name
         WHERE s.owner = 'APPS'
           AND s.table_owner IN ({_LICENSED_OWNERS})
    """,
    # declared FKs (rare in EBS); both ends mapped back to synonym names
    "foreign_keys": """
        SELECT sf.synonym_name, fcc.column_name, sp.synonym_name, pcc.column_name
          FROM all_constraints fc
          JOIN all_cons_columns fcc
            ON fcc.owner = fc.owner AND fcc.constraint_name = fc.constraint_name
          JOIN all_constraints pc
            ON pc.owner = fc.r_owner AND pc.constraint_name = fc.r_constraint_name
          JOIN all_cons_columns pcc
            ON pcc.owner = pc.owner AND pcc.constraint_name = pc.constraint_name
           AND pcc.position = fcc.position
          JOIN all_synonyms sf
            ON sf.owner = 'APPS' AND sf.table_owner = fc.owner AND sf.table_name = fc.table_name
          JOIN all_synonyms sp
            ON sp.owner = 'APPS' AND sp.table_owner = pc.owner AND sp.table_name = pc.table_name
         WHERE fc.constraint_type = 'R'
    """,
}


class OracleCatalog(CatalogSource):
    """Real adapter: ALL tables across every licensed ('I') + shared ('S') EBS
    schema, read through the APPS synonym layer with BULK set-based queries (so a
    full ~20k-table instance is feasible -- not one round-trip per table).

    Results are fetched once and cached. The grouping/caching logic is tested via
    a fake cursor in test_extract; the live SQL still needs a real read-only EBS
    account to confirm against a given instance's AOL/flexfield setup. (Flexfield
    and lookup enrichment for live extraction is a documented follow-up -- the
    extractor degrades gracefully when those are absent.)"""

    def __init__(self, cursor):
        self.cur = cursor
        self._tables: list[str] | None = None
        self._cols: dict[str, list[RawColumn]] = {}
        self._pk: dict[str, list[str]] = {}

    def _load(self) -> None:
        if self._tables is not None:
            return
        self.cur.execute(ORACLE_SQL["tables"])
        self._tables = sorted({r[0].lower() for r in self.cur.fetchall()})
        keep = set(self._tables)
        self.cur.execute(ORACLE_SQL["columns"])
        for syn, col, dtype, nullable in self.cur.fetchall():
            t = syn.lower()
            if t in keep:
                self._cols.setdefault(t, []).append(
                    RawColumn(col.lower(), dtype, nullable == "Y"))
        self.cur.execute(ORACLE_SQL["primary_key"])
        for syn, col in self.cur.fetchall():
            t = syn.lower()
            if t in keep:
                self._pk.setdefault(t, []).append(col.lower())

    def tables(self):
        self._load()
        return list(self._tables)

    def columns(self, table):
        self._load()
        return list(self._cols.get(table, []))

    def primary_key(self, table):
        self._load()
        return list(self._pk.get(table, []))

    def foreign_keys(self):
        self._load()
        try:                                  # declared FKs optional; extractor infers
            self.cur.execute(ORACLE_SQL["foreign_keys"])
            return [RawFk(a.lower(), b.lower(), c.lower(), d.lower())
                    for a, b, c, d in self.cur.fetchall()]
        except Exception:
            return []


# -- a small AP + GL mock instance (the spec's proposed starter modules) ------
class MockCatalog(CatalogSource):
    """Simulates a tiny EBS: AP invoices/suppliers + the GL accounting flexfield.
    Declares ZERO foreign keys (as real EBS usually does) -- the extractor must
    infer the join graph from naming conventions."""

    _COLUMNS: dict[str, list[RawColumn]] = {
        "ap_suppliers": [
            RawColumn("vendor_id", "NUMBER", False),
            RawColumn("vendor_name", "VARCHAR2"),
            RawColumn("vendor_number", "VARCHAR2"),
            RawColumn("creation_date", "DATE"),
        ],
        "ap_invoices_all": [
            RawColumn("invoice_id", "NUMBER", False),
            RawColumn("vendor_id", "NUMBER"),
            RawColumn("org_id", "NUMBER"),
            RawColumn("invoice_num", "VARCHAR2"),
            RawColumn("invoice_date", "DATE"),
            RawColumn("invoice_amount", "NUMBER"),
            RawColumn("invoice_type_lookup_code", "VARCHAR2"),
            RawColumn("payment_status_flag", "VARCHAR2"),
        ],
        "ap_invoice_lines_all": [
            RawColumn("invoice_line_id", "NUMBER", False),
            RawColumn("invoice_id", "NUMBER"),
            RawColumn("line_number", "NUMBER"),
            RawColumn("amount", "NUMBER"),
            RawColumn("code_combination_id", "NUMBER"),
            RawColumn("org_id", "NUMBER"),
        ],
        "gl_code_combinations": [
            RawColumn("code_combination_id", "NUMBER", False),
            RawColumn("segment1", "VARCHAR2"),
            RawColumn("segment2", "VARCHAR2"),
            RawColumn("segment3", "VARCHAR2"),
            RawColumn("segment4", "VARCHAR2"),
            RawColumn("segment5", "VARCHAR2"),
        ],
    }
    _PK = {
        "ap_suppliers": ["vendor_id"],
        "ap_invoices_all": ["invoice_id"],
        "ap_invoice_lines_all": ["invoice_line_id"],
        "gl_code_combinations": ["code_combination_id"],
    }
    _FLEX = [  # the customer's Accounting Flexfield segment labels
        RawFlex("gl_code_combinations", "segment1", "company"),
        RawFlex("gl_code_combinations", "segment2", "cost center"),
        RawFlex("gl_code_combinations", "segment3", "account"),
        RawFlex("gl_code_combinations", "segment4", "product"),
        RawFlex("gl_code_combinations", "segment5", "intercompany"),
    ]
    _LOOKUPS = [
        RawLookup("ap_invoices_all", "invoice_type_lookup_code", "INVOICE TYPE",
                  ("STANDARD", "CREDIT", "PREPAYMENT", "MIXED")),
        RawLookup("ap_invoices_all", "payment_status_flag", "PAYMENT STATUS",
                  ("Y", "N", "P")),
    ]

    def tables(self):
        return list(self._COLUMNS)

    def columns(self, table):
        return list(self._COLUMNS[table])

    def primary_key(self, table):
        return list(self._PK.get(table, []))

    def flex_segments(self):
        return list(self._FLEX)

    def lookups(self):
        return list(self._LOOKUPS)
