# streamlit_app/app.py
import io
import os
import sys
from pathlib import Path
from zipfile import ZipFile

import streamlit as st

# ------------------------------------------------------------------------------------
# Ensure repo root is on sys.path so package imports work regardless of CWD
# ------------------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

# Reuse the same regex used by the CLI/runtime → one source of truth
from lakehouse_fm_agent.runtime.lineage import EDGE_RE, SKIP_RE  # type: ignore

# ------------------------------------------------------------------------------------
# Branding / page config
# ------------------------------------------------------------------------------------
st.set_page_config(page_title="SAP FM Analyzer & Converter", layout="wide")
st.title("SAP FM Analyzer & Converter — Preview")

# ------------------------------------------------------------------------------------
# Helpers (parsing, classification, analytics)
# ------------------------------------------------------------------------------------
STD_PREFIXES = ("R", "RS", "RR", "DD", "ENQUEUE", "DEQUEUE", "TR", "S", "CL_", "SAP")

def is_custom(fm: str) -> bool:
    # Treat Y*, Z*, and namespaces like /OSP/*, /BIC/* as custom
    if not isinstance(fm, str):
        return False
    u = fm.upper().strip()
    return u.startswith(("Y", "Z", "/"))

def is_standard(fm: str) -> bool:
    u = fm.upper().strip()
    return u.startswith(STD_PREFIXES)

def category_for_fm(name: str) -> str:
    if not isinstance(name, str):
        return "Domain_Logic_Other"
    n = name.upper()
    if n.startswith(("RSD","RSR","RRSI","RSAU","RST","RSW","RSKC","RS_")):
        return "BW_Platform_API"
    if ("CURRENCY" in n) or ("CURR" in n) or ("UNIT_CONVERSION" in n) or ("UOM" in n) or ("BUOM" in n) or ("SSU" in n):
        return "UoM_Currency"
    if ("READ_0MATERIAL" in n) or ("READ_PRODFORM" in n) or ("READ_CUSTSALES" in n) or ("READ_ECLASS" in n) or ("READ_MASTER_DATA" in n):
        return "Masterdata_Readers"
    if ("DATE" in n) or ("MONTH" in n) or ("WEEK" in n) or ("PERIOD" in n) or n in {"SN_LAST_DAY_OF_MONTH","SLS_MISC_GET_LAST_DAY_OF_MONTH","/OSP/GET_DAYS_IN_MONTH","Y_PCM_LAST_DAY_OF_MONTH"}:
        return "Calendar_Time"
    if ("CONVERSION_EXIT_ALPHA" in n) or ("NUMERIC_CHECK" in n) or ("CHAVL_CHECK" in n) or ("REPLACE_STRANGE_CHARS" in n):
        return "Data_Cleansing_Validation"
    if "HIER" in n:
        return "Hierarchy"
    if "PAYTERMDAYS" in n:
        return "Payment_Terms"
    if "SID" in n or "TEXTS" in n:
        return "BW_SID_Texts"
    return "Domain_Logic_Other"

def parse_edges_text(text: str):
    """
    Parse a LINEAGE.txt-like string into:
      list of tuples (parent, child, kind, skipped)
    """
    rows = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m_skip = SKIP_RE.search(line)
        if m_skip:
            p = m_skip.group("p").strip()
            c = m_skip.group("c").strip()
            rows.append((p, c, "FM", True))
            continue
        m = EDGE_RE.search(line.replace("  ", " ").replace(" -", "-"))
        if m:
            p = m.group("p").strip()
            c = m.group("c").strip()
            k = m.group("k").strip().upper()
            rows.append((p, c, k, False))
    return rows

def build_mermaid(edges):
    mmd_lines = ["graph TD"] + [f'  "{p}" --> "{c}"' for p, c, k, s in edges]
    return "```mermaid\n" + "\n".join(mmd_lines) + "\n```"

def summarize_edges(edges):
    nodes = set()
    for p, c, k, s in edges:
        nodes.add(p); nodes.add(c)
    return {"edges": len(edges), "nodes": len(nodes)}

def deg_metrics(edges):
    """
    Return:
      - in_deg: dict[fm] = count
      - out_deg: dict[fm] = count
      - nodes: set of all fm
    """
    in_deg, out_deg, nodes = {}, {}, set()
    for p, c, k, s in edges:
        nodes.add(p); nodes.add(c)
        out_deg[p] = out_deg.get(p, 0) + 1
        in_deg[c] = in_deg.get(c, 0) + 1
        in_deg.setdefault(p, in_deg.get(p, 0))
        out_deg.setdefault(c, out_deg.get(c, 0))
    return in_deg, out_deg, nodes

def criticality_score(fm, in_deg, out_deg):
    # simple, interpretable: emphasize shared utilities (fan-in), then fan-out; custom +1
    score = in_deg.get(fm,0) * 2 + out_deg.get(fm,0)
    if is_custom(fm):
        score += 1
    return score

def bfs_subgraph(edges, root_fm, depth=2, direction="out"):
    """
    Build a subgraph (edge list) around a root FM, up to 'depth'.
    direction: "out" = root -> children; "in" = parents -> root; "both" = both ways
    """
    adj_out, adj_in = {}, {}
    for p,c,k,s in edges:
        adj_out.setdefault(p, set()).add(c)
        adj_in.setdefault(c, set()).add(p)

    frontier = {root_fm}
    seen = {root_fm}
    out_edges = []

    for _ in range(depth):
        next_frontier = set()
        for fm in frontier:
            if direction in ("out","both"):
                for ch in adj_out.get(fm, set()):
                    out_edges.append((fm, ch, "FM", False))
                    if ch not in seen:
                        seen.add(ch); next_frontier.add(ch)
            if direction in ("in","both"):
                for pa in adj_in.get(fm, set()):
                    out_edges.append((pa, fm, "FM", False))
                    if pa not in seen:
                        seen.add(pa); next_frontier.add(pa)
        frontier = next_frontier
        if not frontier:
            break
    return out_edges

def to_csv(rows, header):
    buf = io.StringIO()
    buf.write(",".join(header) + "\n")
    for r in rows:
        buf.write(",".join([str(x) for x in r]) + "\n")
    buf.seek(0)
    return buf.getvalue()

# ------------------------------------------------------------------------------------
# Upload + parsing: accept both ZIP (many runs) and TXT (single run)
# ------------------------------------------------------------------------------------
uploaded = st.file_uploader(
    "Upload ZIP (multiple FM run folders) or a single LINEAGE.txt",
    type=["zip", "txt"],
)

if not uploaded:
    st.info(
        "Upload a **.zip** containing multiple FM runs (each with `LINEAGE.txt`) "
        "or a single **LINEAGE.txt** to see analytics, subgraph, and docs."
    )
    st.stop()

runs = {}  # { run_key: [(p,c,k,s), ...] }

if uploaded.name.lower().endswith(".zip"):
    z = ZipFile(io.BytesIO(uploaded.getvalue()))
    lineage_members = [n for n in z.namelist() if n.upper().endswith("LINEAGE.TXT")]
    if not lineage_members:
        st.error("No `LINEAGE.txt` files found in the ZIP.")
        st.stop()
    for member in lineage_members:
        try:
            text = z.read(member).decode("utf-8", errors="ignore")
        except Exception:
            text = z.read(member).decode("latin-1", errors="ignore")
        parts = [p for p in member.split("/") if p]
        run_key = parts[0] if parts else member
        edges = parse_edges_text(text)
        if edges:
            runs.setdefault(run_key, []).extend(edges)

else:  # .txt
    text = uploaded.getvalue().decode("utf-8", errors="ignore")
    runs["SINGLE_RUN"] = parse_edges_text(text)

if not any(runs.values()):
    st.warning("Uploaded file(s) did not yield any edges. Check the input format.")
    st.stop()

# Combined
combined = []
for v in runs.values():
    combined.extend(v)

# ------------------------------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------------------------------
view_choice = st.sidebar.selectbox("Select run to visualize", ["Combined"] + list(runs.keys()))
if view_choice == "Combined":
    edges_view = combined
else:
    edges_view = runs[view_choice]

# Search/focus controls
st.sidebar.markdown("**Focus FM (subgraph)**")
all_nodes = sorted({x for e in edges_view for x in (e[0], e[1])})
focus_fm = st.sidebar.selectbox("FM", options=["<none>"] + all_nodes, index=0)
depth = st.sidebar.slider("Depth", min_value=1, max_value=5, value=2)
direction = st.sidebar.selectbox("Direction", options=["out","in","both"], index=0)

# ------------------------------------------------------------------------------------
# Top-level summary + analytics
# ------------------------------------------------------------------------------------
stats = summarize_edges(edges_view)
st.write(f"**Runs parsed:** {len(runs)}  |  **Nodes:** {stats['nodes']}  |  **Edges:** {stats['edges']}")

# Build node attributes for analytics
in_deg, out_deg, nodes = deg_metrics(edges_view)

rows_summary = []
cat_counts = {}
cust_cnt = std_cnt = 0
for n in nodes:
    cat = category_for_fm(n)
    cat_counts[cat] = cat_counts.get(cat, 0) + 1
    if is_custom(n): cust_cnt += 1
    elif is_standard(n): std_cnt += 1
    rows_summary.append((
        n,
        in_deg.get(n,0),
        out_deg.get(n,0),
        "Y" if is_custom(n) else "N",
        cat,
        criticality_score(n, in_deg, out_deg)
    ))

# Sort by criticality
rows_summary.sort(key=lambda r: r[-1], reverse=True)

# Show top 20 critical FMs
st.subheader("Top 20 critical FMs (by fan-in/out + custom bonus)")
st.dataframe(rows_summary[:20], use_container_width=True, hide_index=True)

# Category & custom/standard counts
with st.expander("Category distribution & custom/standard split"):
    st.write("**Custom (Y/Z/namespace)**:", cust_cnt, " | **Standard (SAP/BW)**:", std_cnt)
    # Render a small table for categories
    st.table(sorted([(k,v) for k,v in cat_counts.items()], key=lambda x: x[1], reverse=True))

# ------------------------------------------------------------------------------------
# Subgraph around a focused FM
# ------------------------------------------------------------------------------------
st.subheader("Focused subgraph")
if focus_fm != "<none>":
    sub_edges = bfs_subgraph(edges_view, focus_fm, depth=depth, direction=direction)
    if not sub_edges:
        st.info("No neighbors found with current settings.")
    else:
        st.markdown(build_mermaid(sub_edges))
        with st.expander("Subgraph edges table"):
            st.dataframe(sub_edges, use_container_width=True, hide_index=True)
else:
    st.info("Pick a **Focus FM** on the left to see a subgraph (with depth & direction).")

# ------------------------------------------------------------------------------------
# Full edges table + downloads
# ------------------------------------------------------------------------------------
with st.expander("Preview edges table (full view selection)"):
    st.dataframe(edges_view, use_container_width=True, hide_index=True)

left, right = st.columns(2)
with left:
    csv = to_csv(edges_view, header=["parent","child","kind","skipped"])
    st.download_button("Download edges (CSV)", data=csv, file_name=f"{view_choice}_edges.csv", mime="text/csv")
with right:
    # Export the Top‑N list as CSV
    top_csv = to_csv(rows_summary[:50], header=["fm","in_degree","out_degree","is_custom","category","criticality"])
    st.download_button("Download Top‑50 critical (CSV)", data=top_csv, file_name=f"{view_choice}_top50.csv", mime="text/csv")

# ------------------------------------------------------------------------------------
# Doc seeds (Master / LLD / How-to-Test) for the focused FM
# ------------------------------------------------------------------------------------
st.subheader("Generate documentation seeds (focused FM)")

if focus_fm == "<none>":
    st.info("Select a **Focus FM** to generate doc seeds.")
else:
    # Compose doc stubs in Markdown
    fm_cat = category_for_fm(focus_fm)
    fm_is_cust = "Yes" if is_custom(focus_fm) else "No"
    ptr = ""
    if fm_cat == "BW_Platform_API":
        ptr = (
            "- **Pointer**: Replace BW mechanics with Lakehouse patterns:\n"
            "  - DSO writes → **Delta MERGE**\n"
            "  - SID lookups → **joins by business keys**\n"
            "  - DDIC/metadata → control tables\n"
        )

    master_md = f"""# {focus_fm} — Master Doc

## AS‑IS (inferred)
- Category: **{fm_cat}** | Custom: **{fm_is_cust}**
- Role in graph: in_degree={in_deg.get(focus_fm,0)}, out_degree={out_deg.get(focus_fm,0)}, criticality={criticality_score(focus_fm,in_deg,out_deg)}

## Lineage (local view)
Use the "Focused subgraph" above to visualize N‑hop context.

## TO‑BE (Databricks)
- Follow category guidance:
  - UoM/Currency → **table‑driven** (FX, currency decimals, UoM factors) + Spark UDFs
  - Masterdata readers → **joins** to curated dims in Unity Catalog
  - Calendar/Date → Spark functions + enterprise calendar table
  - Cleansing/Alpha → Spark SQL expressions/UDFs
  - BW Platform → **no 1:1**; use **MERGE/joins/control tables**
{ptr}
"""

    lld_md = f"""# {focus_fm} — Design LLD

## Components
- Handler module: `core/` or `bw_replace/` per category
- Inputs/Outputs: define schemas (DECIMAL scales for currency/UoM)

## Contracts
- Tables: FX, currency_decimals, uom_factors, calendar, logsys_map (as applicable)
- Config: rounding policy, scale

## Pseudocode
1) Read required reference tables
2) Transform using Spark SQL/UDFs
3) MERGE/WRITE results or return DataFrame

## Error handling
- Null/invalid checks, logging, metrics
"""

    test_md = f"""# {focus_fm} — How to Test

## Unit tests
- Golden cases for conversion/rounding (currency/UoM)
- Cleansing/alpha behaviors

## Integration tests
- Replay a short path from lineage (parents/children around {focus_fm})
- Validate outputs and row counts
"""

    c1, c2, c3 = st.columns(3)
    c1.download_button("Download Master.md", master_md, file_name=f"{focus_fm}_MASTER.md")
    c2.download_button("Download Design_LLD.md", lld_md, file_name=f"{focus_fm}_LLD.md")
    c3.download_button("Download How_to_Test.md", test_md, file_name=f"{focus_fm}_TEST.md")
