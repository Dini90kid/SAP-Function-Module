
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fmtool.py — Lakehouse FM Agent CLI (robust imports for Streamlit/Jobs)

Recommended:
  python -m lakehouse_fm_agent.fmtool --help
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# --- Put repo root and package dir on sys.path (works regardless of CWD) ---
_THIS_FILE = Path(__file__).resolve()
_PKG_DIR   = _THIS_FILE.parent                  # .../lakehouse_fm_agent
_REPO_ROOT = _PKG_DIR.parent                    # .../

for p in (str(_REPO_ROOT), str(_PKG_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# --- Absolute imports only (no 'from runtime...' ) ---
from lakehouse_fm_agent.runtime.lineage import load_edges, topo_layers
from lakehouse_fm_agent.runtime.registry import resolve_handler, POINTERS
from lakehouse_fm_agent.runtime.manifest import Manifest


def _echo(msg: str) -> None:
    print(msg, flush=True)

def _fail(msg: str, exit_code: int = 2) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr, flush=True)
    sys.exit(exit_code)

# ----------------------------- Commands --------------------------------------

def cmd_graph(args: argparse.Namespace) -> None:
    lineage_path = Path(args.lineage); out_path = Path(args.out)
    if not lineage_path.exists():
        _fail(f"Lineage file not found: {lineage_path}")
    edges = load_edges(str(lineage_path))
    mmd = ["graph TD"]
    for p, c, k, _s in edges:
        mmd.append(f'  "{p}" --> "{c}"')
    out_path.write_text("
".join(mmd), encoding="utf-8")
    _echo(f"Mermaid saved: {out_path}")


def cmd_plan(args: argparse.Namespace) -> None:
    m = Manifest(plan_id=args.plan_id)
    # Minimal ref objects — adjust to your UC
    m.ensure_table("uc_catalog","ref","fx_rates","FX conversion reference",[])
    m.ensure_table("uc_catalog","ref","currency_decimals","Currency decimals",[])
    m.ensure_table("uc_catalog","ref","uom_factors","UoM base/factors",[])
    m.ensure_table("uc_catalog","ref","calendar","Enterprise calendar",[])
    m.ensure_table("uc_catalog","ref","logsys_map","Logical system map",[])
    Path(args.out).write_text(m.to_json(), encoding="utf-8")
    _echo(f"Manifest saved: {args.out}")


def cmd_scaffold(args: argparse.Namespace) -> None:
    fm = args.fm.upper(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out/"master.md").write_text(f"# {fm} — Master Doc

AS-IS, TO-BE, lineage, pointers, code.
", encoding="utf-8")
    (out/"design_lld.md").write_text(f"# {fm} — Design LLD

Components, contracts, code slices.
", encoding="utf-8")
    (out/"how_to_test.md").write_text(f"# {fm} — How to Test

Unit + integration tests.
", encoding="utf-8")
    handler_src = (
        "from lakehouse_fm_agent.runtime.context import Context


"
        "def handler(ctx: Context) -> None:
"
        '    """
'
        f"    Handler for {fm}. Implement per Master/LLD.
"
        "    - Read inputs (ctx.current_df or ctx.read_uc(...))
"
        "    - Apply Spark SQL / UDF / joins according to patterns
"
        "    - Write outputs or assign ctx.current_df
"
        '    """
'
        "    pass
"
    )
    (out/"handler.py").write_text(handler_src, encoding="utf-8")
    _echo(f"Scaffold created under: {out}")


def cmd_validate(args: argparse.Namespace) -> None:
    if args.lineage and not Path(args.lineage).exists():
        _fail(f"Provided lineage path not found: {args.lineage}")
    if args.config and not Path(args.config).exists():
        _fail(f"Provided config path not found: {args.config}")
    _echo("Validate: OK (stub).")


def cmd_test(args: argparse.Namespace) -> None:
    _echo(f"Run tests for FM={args.fm} (stub).")


def cmd_run(args: argparse.Namespace) -> None:
    lineage_path = Path(args.lineage)
    if not lineage_path.exists():
        _fail(f"Lineage file not found: {lineage_path}")
    edges = load_edges(str(lineage_path))
    layers = topo_layers(edges)
    ordered, seen = [], set()
    for layer in layers:
        for fm in layer:
            if fm not in seen:
                seen.add(fm); ordered.append(fm)
    _echo(f"Execution order (topological): {', '.join(ordered)}")
    for fm in ordered:
        fn = resolve_handler(fm)
        if fn:
            _echo(f"[RUN] {fm} -> handler")
            try:
                fn(None)  # In Databricks, pass Context(spark)
            except Exception as ex:
                _fail(f"Handler for {fm} raised an exception: {ex}")
        else:
            ptr = POINTERS.get(fm.upper())
            if ptr:
                _echo(f"[POINTER] {fm}: {ptr}")
            else:
                _echo(f"[SKIP] {fm}: no handler (BW pattern handled elsewhere)")

# ------------------------------- CLI -----------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("fmtool", description="Lakehouse FM Agent CLI")
    sp = ap.add_subparsers(dest="command")

    p = sp.add_parser("graph", help="Render Mermaid from LINEAGE.txt")
    p.add_argument("--lineage", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_graph)

    p = sp.add_parser("plan", help="Produce a PLANNED object manifest (JSON)")
    p.add_argument("--lineage", required=False)
    p.add_argument("--plan-id", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_plan)

    p = sp.add_parser("scaffold", help="Create docs + handler skeleton for an FM")
    p.add_argument("--fm", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_scaffold)

    p = sp.add_parser("validate", help="Validate config + lineage (stub)")
    p.add_argument("--config", required=False)
    p.add_argument("--lineage", required=False)
    p.set_defaults(fn=cmd_validate)

    p = sp.add_parser("test", help="Run unit/integration tests (stub)")
    p.add_argument("--fm", required=True)
    p.set_defaults(fn=cmd_test)

    p = sp.add_parser("run", help="Execute handlers in topological order")
    p.add_argument("--lineage", required=True)
    p.set_defaults(fn=cmd_run)
    return ap


def main(argv: list[str] | None = None) -> None:
    ap = build_parser()
    args = ap.parse_args(argv)
    if not hasattr(args, "fn"):
        ap.print_help(); sys.exit(1)
    args.fn(args)

if __name__ == "__main__":
    main()
