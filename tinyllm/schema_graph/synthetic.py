"""SyntheticSchemaGenerator -- diverse, EBS-shaped synthetic schemas.

For the vendor base we need MANY varied schemas (cross-schema generalization),
but they must carry real EBS shapes or the model never learns the idioms:
  - header / lines structure (1- and 2-hop joins)
  - a flexfield (KFF) table: code-combination with segmentN + business labels
  - multi-org (_ALL) striping via an org_id column
  - lookup-coded status columns with a fixed value domain
  - optional second dimension (extra join targets)

Names are drawn from large pools + a per-schema custom prefix (simulating `XX*`
custom schemas) + varied column naming, so train vs val seeds yield largely
DIFFERENT identifiers -- making cross-schema eval a real test of unseen
vocabulary, not just unseen combinations. Seeded for reproducibility.
"""

from __future__ import annotations

import random

from .types import Column, ColumnType, ForeignKey, Schema, SemanticRole, Table

_ENTITIES = [
    "supplier", "vendor", "customer", "party", "employee", "product", "item",
    "account", "project", "location", "organization", "bank", "carrier", "buyer",
    "requester", "department", "asset", "contract", "warehouse", "ledger",
    "currency", "resource", "operator", "payee",
]
_DOCS = [
    "invoice", "order", "voucher", "receipt", "payment", "shipment", "requisition",
    "journal", "transaction", "claim", "settlement", "adjustment", "transfer",
    "remittance", "disbursement", "accrual",
]
_AMOUNTS = [
    "amount", "total", "net_amount", "entered_amount", "line_amount", "gross_amount",
    "tax_amount", "func_amount", "base_amount", "paid_amount", "extended_amount",
    "accounted_amount",
]
_QTYS = ["quantity", "qty", "ordered_qty", "shipped_qty", "received_qty"]
_STATUSES = {
    "APPROVAL_STATUS": ("APPROVED", "PENDING", "REJECTED"),
    "PROCESS_STATUS": ("OPEN", "CLOSED", "ON_HOLD"),
    "PAYMENT_STATUS": ("PAID", "UNPAID", "PARTIAL"),
    "MATCH_STATUS": ("MATCHED", "UNMATCHED", "NEEDS_REVIEW"),
    "POSTING_STATUS": ("POSTED", "UNPOSTED", "ERROR"),
    "WORKFLOW_STATUS": ("STARTED", "COMPLETE", "SUSPENDED"),
}
_SEGMENT_LABELS = [
    "company", "cost center", "account", "department", "region", "project",
    "product", "line of business", "intercompany", "location", "division", "future use",
]
_NAME_COLS = ["{e}_name", "name", "description", "{e}_description", "title", "display_name"]
_DIM2 = ["category", "class", "type", "region", "status_group", "source", "channel"]
_PREFIXES = ["", "", "", "", "xx_", "xxen_", "cust_", "xxgl_"]

# Procedural roots (opt-in): near-unique pronounceable identifiers so train/val
# schemas DON'T share the core entity/doc nouns -> forces the model to link the
# question to the serialized schema instead of memorizing a tiny noun vocabulary.
_CONS = "bcdfgklmnprstvz"
_VOWS = "aeiou"


class SyntheticSchemaGenerator:
    def __init__(self, seed: int, procedural: bool = False):
        self.rng = random.Random(seed)
        self.seed = seed
        self.procedural = procedural

    def _root(self) -> str:
        rng = self.rng
        syls = []
        for _ in range(rng.randint(2, 3)):
            s = rng.choice(_CONS) + rng.choice(_VOWS)
            if rng.random() < 0.4:
                s += rng.choice(_CONS)
            syls.append(s)
        return "".join(syls)

    def generate(self) -> Schema:
        rng = self.rng
        prefix = rng.choice(_PREFIXES)
        entity = self._root() if self.procedural else rng.choice(_ENTITIES)
        doc = self._root() if self.procedural else rng.choice(_DOCS)

        def tn(name: str) -> str:
            return prefix + name

        tables: list[Table] = []
        fks: list[ForeignKey] = []

        # --- dimension: an entity table (1-hop target) ---------------------
        entity_pk = f"{entity}_id"
        entity_tbl = tn(entity)
        tables.append(Table(entity_tbl, [
            Column(entity_pk, ColumnType.NUMBER, is_pk=True, role=SemanticRole.ID),
            Column(rng.choice(_NAME_COLS).format(e=entity), ColumnType.VARCHAR2, role=SemanticRole.NAME),
            Column(f"{entity}_number", ColumnType.VARCHAR2, role=SemanticRole.CODE),
        ]))

        # --- optional second dimension -------------------------------------
        include_dim2 = rng.random() < 0.5
        d2_tbl = d2_pk = None
        if include_dim2:
            if self.procedural:
                d2 = self._root()
                while d2 == entity:
                    d2 = self._root()
            else:
                d2 = rng.choice([x for x in _DIM2 if x != entity])
            d2_pk, d2_tbl = f"{d2}_id", tn(d2)
            tables.append(Table(d2_tbl, [
                Column(d2_pk, ColumnType.NUMBER, is_pk=True, role=SemanticRole.ID),
                Column(f"{d2}_name", ColumnType.VARCHAR2, role=SemanticRole.NAME),
            ]))

        # --- optional flexfield (KFF) code-combination ---------------------
        include_flex = rng.random() < 0.6
        cc_tbl = None
        if include_flex:
            cc_tbl = tn("code_combinations")
            n_seg = rng.randint(2, 5)
            labels = rng.sample(_SEGMENT_LABELS, n_seg)
            seg = [
                Column(f"segment{i + 1}", ColumnType.VARCHAR2,
                       role=SemanticRole.FLEXFIELD_SEGMENT, business_label=labels[i])
                for i in range(n_seg)
            ]
            tables.append(Table(cc_tbl, [
                Column("code_combination_id", ColumnType.NUMBER, is_pk=True, role=SemanticRole.ID)
            ] + seg))

        # --- header (_ALL multi-org) document table ------------------------
        multi_org = rng.random() < 0.8
        status_type = rng.choice(list(_STATUSES))
        header = tn(f"{doc}_headers_all" if multi_org else f"{doc}_headers")
        header_pk = f"{doc}_id"
        cols = [
            Column(header_pk, ColumnType.NUMBER, is_pk=True, role=SemanticRole.ID),
            Column(entity_pk, ColumnType.NUMBER, role=SemanticRole.ID),
            Column(f"{doc}_date", ColumnType.DATE, role=SemanticRole.DATE),
            Column(rng.choice(_AMOUNTS), ColumnType.NUMBER, role=SemanticRole.AMOUNT),
            Column(status_type.lower() + "_code", ColumnType.VARCHAR2, role=SemanticRole.LOOKUP,
                   lookup_type=status_type, allowed_values=_STATUSES[status_type]),
        ]
        if include_dim2:
            cols.insert(2, Column(d2_pk, ColumnType.NUMBER, role=SemanticRole.ID))
        if multi_org:
            cols.insert(2, Column("org_id", ColumnType.NUMBER, role=SemanticRole.ORG_ID))
        tables.append(Table(header, cols, is_multi_org=multi_org))
        fks.append(ForeignKey(header, entity_pk, entity_tbl, entity_pk))
        if include_dim2:
            fks.append(ForeignKey(header, d2_pk, d2_tbl, d2_pk))

        # --- lines table (enables 2-hop join to the flexfield) -------------
        if include_flex or rng.random() < 0.5:
            lines = tn(f"{doc}_lines_all" if multi_org else f"{doc}_lines")
            lcols = [
                Column(f"{doc}_line_id", ColumnType.NUMBER, is_pk=True, role=SemanticRole.ID),
                Column(header_pk, ColumnType.NUMBER, role=SemanticRole.ID),
                Column(rng.choice(_AMOUNTS), ColumnType.NUMBER, role=SemanticRole.AMOUNT),
            ]
            if rng.random() < 0.5:
                lcols.append(Column(rng.choice(_QTYS), ColumnType.NUMBER, role=SemanticRole.QUANTITY))
            if include_flex:
                lcols.append(Column("code_combination_id", ColumnType.NUMBER, role=SemanticRole.ID))
            tables.append(Table(lines, lcols, is_multi_org=multi_org))
            fks.append(ForeignKey(lines, header_pk, header, header_pk))
            if include_flex:
                fks.append(ForeignKey(lines, "code_combination_id", cc_tbl, "code_combination_id"))

        return Schema(name=f"syn_{doc}_{self.seed}", tables=tables, foreign_keys=fks)
