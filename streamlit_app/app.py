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

# Reuse the same regex used by the CLI/runtime → single source of truth
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
    if not isinstance(fm, str):
        return False
    u = fm.upper().strip()
    return u.startswith(("Y", "Z", "/"))  # Y*, Z*, or namespaces (/OSP/*, /BIC/*)

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
    if ("DATE" in n) or ("MONTH" in n) or ("WEEK" in n) or ("PERIOD" in n) or n in {
        "SN_LAST_DAY_OF_MONTH","SLS_MISC_GET_LAST_DAY_OF_MONTH","/OSP/GET_DAYS_IN_MONTH","Y_PCM_LAST_DAY_OF_MONTH"
    }:
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

# Workaround/pattern pointers (show in the summary for standard FMs)
POINTERS = {
    "BW_Platform_API":
        "- Replace BW mechanics with Lakehouse patterns:\n"
        "  - DSO writes → **Delta MERGE**\n"
        "  - SID lookups → **joins by business keys**\n"
        "  - DDIC/metadata → **control tables**",
    "UoM_Currency":
        "- Implement **table-driven** conversion (FX, currency decimals, UoM factors) + **Spark UDFs**",
    "Masterdata_Readers":
        "- Replace with **joins** to curated dimensions in Unity Catalog",
    "Calendar_Time":
        "- Use **Spark date functions** + enterprise **calendar** table",
    "Data_Cleansing_Validation":
        "- **Lightweight Spark SQL** expressions/UDFs (e.g., lpad, regex checks)",
    "Hierarchy":
        "- Maintain a **parent-child** table; resolve via recursive/self-joins",
    "Payment_Terms":
        "- Model as **table-driven rules**; compute via PySpark functions",
    "BW_SID_Texts":
        "- No 1:1; maintain **attributes/texts** in Delta + **MERGE**",
    "Domain_Logic_Other":
        "- Assess; rebuild as PySpark transforms with small reference tables"
}

def classify_action(fm: str):
    """
    Returns (action, rationale, category, is_custom_bool)
      action ∈ {REBUILD, PATTERN, LIGHTWEIGHT, JOIN}
    """
    cat = category_for_fm(fm)
    custom = is_custom(fm)
    if custom:
        if cat in {"UoM_Currency", "Masterdata_Readers", "Hierarchy", "Payment_Terms", "Domain_Logic_Other"}:
            return ("REBUILD", POINTERS.get(cat, POINTERS["Domain_Logic_Other"]), cat, True)
        if cat == "Data_Cleansing_Validation":
            return ("LIGHTWEIGHT", POINTERS[cat], cat, True)
        if cat == "Calendar_Time":
            return ("LIGHTWEIGHT", POINTERS[cat], cat, True)
        if cat in {"BW_Platform_API","BW_SID_Texts"}:
            return ("PATTERN", POINTERS[cat], cat, True)
        return ("REBUILD", POINTERS["Domain_Logic_Other"], cat, True)
    else:
        # Standard SAP/BW → Pattern/Lightweight/Join
        if cat in {"BW_Platform_API","BW_SID_Texts"}:
            return ("PATTERN", POINTERS[cat], cat, False)
        if cat in {"UoM_Currency","Data_Cleansing_Validation","Calendar_Time"}:
            return ("LIGHTWEIGHT", POINTERS[cat], cat, False)
        if cat == "Masterdata_Readers":
            return ("JOIN", POINTERS[cat], cat, False)
        return ("PATTERN", POINTERS["Domain_Logic_Other"], cat, False)

def parse_edges_text(text: str):
    """
    Parse a LINEAGE.txt-like string into: list[(parent, child, kind, skipped)]
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
    in_deg, out_deg, nodes = {}, {}, set()
    for p, c, k, s in edges:
        nodes.add(p); nodes.add(c)
        out_deg[p] = out_deg.get(p, 0) + 1
        in_deg[c]  = in_deg.get(c, 0) + 1
        in_deg.setdefault(p, in_deg.get(p, 0))
        out_deg.setdefault(c, out_deg.get(c, 0))
    return in_deg, out_deg, nodes

def bfs_collect(edges, root, mode="out", max_depth=5):
    """Return set of nodes reachable from root up to depth (excluding root)."""
    adj_out, adj_in = {}, {}
    for p,c,k,s in edges:
        adj_out.setdefault(p, set()).add(c)
        adj_in.setdefault(c, set()).add(p)

    frontier, seen = {root}, {root}
    out_nodes = set()
    for _ in range(max_depth):
        nxt = set()
        for fm in frontier:
            if mode in ("out","both"):
                for ch in adj_out.get(fm, set()):
                    if ch not in seen:
                        seen.add(ch); nxt.add(ch); out_nodes.add(ch)
            if mode in ("in","both"):
                for pa in adj_in.get(fm, set()):
                    if pa not in seen:
                        seen.add(pa); nxt.add(pa); out_nodes.add(pa)
        frontier = nxt
        if not frontier:
            break
    return out_nodes

def fm_summary(focus_fm, edges, in_deg, out_deg, nested_depth=3):
    """
    Build a rich summary for a single FM:
      - direct children
      - nested callees (up to nested_depth)
      - inbound callers
      - actions (REBUILD, PATTERN, LIGHTWEIGHT, JOIN) with rationale
      - 'create list': custom FMs (direct + nested) that must be implemented
      - 'pattern list': standard FMs we handle via patterns
    """
    # Direct children
    direct = []
    for p,c,k,s in edges:
        if p == focus_fm and not s:
            act, why, cat, is_c = classify_action(c)
            direct.append((c, cat, "Y" if is_c else "N", act, why))

    # Nested callees
    nested_nodes = bfs_collect(edges, focus_fm, mode="out", max_depth=nested_depth)
    nested = []
    for n in sorted(nested_nodes):
        if n == focus_fm:
            continue
        act, why, cat, is_c = classify_action(n)
        nested.append((n, cat, "Y" if is_c else "N", act, why))

    # Inbound callers (parents)
    parents = sorted({p for p,c,k,s in edges if c == focus_fm})

    # Build 'create' vs 'pattern' suggestion lists
    create_list = sorted({name for (name, cat, y, act, why) in (direct + nested) if act == "REBUILD"})
    pattern_list = sorted({name for (name, cat, y, act, why) in (direct + nested) if act in {"PATTERN","LIGHTWEIGHT","JOIN"}})

    metrics = {
        "in_degree": in_deg.get(focus_fm, 0),
        "out_degree": out_deg.get(focus_fm, 0),
        "criticality": (in_deg.get(focus_fm,0) * 2 + out_deg.get(focus_fm,0) + (1 if is_custom(focus_fm) else 0))
    }

    return {
        "direct": direct,
        "nested": nested,
        "parents": parents,
        "create_list": create_list,
        "pattern_list": pattern_list,
        "metrics": metrics
    }

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
        "or a single **LINEAGE.txt** to see analytics, summaries, and build lists."
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
        # Use the first path segment as run key (N4MSALES/LINEAGE.txt → N4MSALES)
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
edges_view = combined if view_choice == "Combined" else runs[view_choice]

# Build degree metrics once
in_deg, out_deg, nodes = deg_metrics(edges_view)
all_nodes = sorted(nodes)

st.sidebar.markdown("**Focus FM (summary & subgraph)**")
focus_fm = st.sidebar.selectbox("FM", options=["<none>"] + all_nodes, index=0)
depth = st.sidebar.slider("Nested depth (calls)", min_value=1, max_value=8, value=3)


# ------------------------------------------------------------------------------------
# Top-level summary + analytics per view
# ------------------------------------------------------------------------------------
stats = summarize_edges(edges_view)
st.write(f"**Runs parsed:** {len(runs)}  |  **Nodes:** {stats['nodes']}  |  **Edges:** {stats['edges']}")

# Build overall create list (custom FMs across the current view)
overall_create = sorted({fm for fm in nodes if is_custom(fm)})

st.subheader("Overall build list (custom FMs to implement as handlers)")
st.write(f"Total custom FMs: **{len(overall_create)}**")
st.dataframe(overall_create, use_container_width=True)
st.download_button(
    "Download overall build list (TXT)",
    data="\n".join(overall_create),
    file_name=f"{view_choice}_build_list.txt",
    mime="text/plain"
)


# ------------------------------------------------------------------------------------
# Per‑FM summary (what it calls, what to convert vs. pattern, workaround)
# ------------------------------------------------------------------------------------
st.subheader("Per‑FM summary (what it calls, recommendations, workarounds)")
if focus_fm == "<none>":
    st.info("Select a **Focus FM** on the left to generate its summary.")
else:
    summary = fm_summary(focus_fm, edges_view, in_deg, out_deg, nested_depth=depth)

    m = summary["metrics"]
    st.markdown(
        f"- **FM:** `{focus_fm}`  |  **in_degree:** {m['in_degree']}  |  **out_degree:** {m['out_degree']}  |  **criticality:** {m['criticality']}"
    )

    # Direct children table
    with st.expander("Direct children (one hop)"):
        st.dataframe(
            summary["direct"],
            use_container_width=True,
            hide_index=True
        )
        st.download_button(
            "Download (CSV)",
            data=to_csv(summary["direct"], header=["child","category","is_custom","action","rationale"]),
            file_name=f"{focus_fm}_direct_children.csv",
            mime="text/csv"
        )

    # Nested callees table
    with st.expander(f"Nested callees (≤ {depth} hops)"):
        st.dataframe(
            summary["nested"],
            use_container_width=True,
            hide_index=True
        )
        st.download_button(
            "Download (CSV)",
            data=to_csv(summary["nested"], header=["callee","category","is_custom","action","rationale"]),
            file_name=f"{focus_fm}_nested_callees.csv",
            mime="text/csv"
        )

    # Parents
    with st.expander("Inbound callers (parents)"):
        st.dataframe(summary["parents"], use_container_width=True)

    # Create vs Pattern lists
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Create in Python (handlers)**")
        st.dataframe(summary["create_list"], use_container_width=True)
        st.download_button(
            "Download create list (TXT)",
            data="\n".join(summary["create_list"]),
            file_name=f"{focus_fm}_create_list.txt",
            mime="text/plain"
        )
    with c2:
        st.markdown("**Replace via pattern / lightweight / join**")
        st.dataframe(summary["pattern_list"], use_container_width=True)
        st.download_button(
            "Download pattern list (TXT)",
            data="\n".join(summary["pattern_list"]),
            file_name=f"{focus_fm}_pattern_list.txt",
            mime="text/plain"
        )

    # Doc seed (one-click)
    master_md = f"""# {focus_fm} — Master Doc

## AS‑IS (inferred)
- in_degree={m['in_degree']}, out_degree={m['out_degree']}, criticality={m['criticality']}
- Parents (who call this FM): {", ".join(summary["parents"]) if summary["parents"] else "—"}

## TO‑BE (Databricks)
- Rebuild list (handlers): {", ".join(summary["create_list"]) if summary["create_list"] else "—"}
- Patterns/lightweight/join: {", ".join(summary["pattern_list"]) if summary["pattern_list"] else "—"}

## Rationale & Workarounds (by category)
{os.linesep.join([f"- {k}: {v}" for k,v in POINTERS.items()])}
"""
    st.download_button(
        "Download Master doc seed (MD)",
        data=master_md,
        file_name=f"{focus_fm}_MASTER.md",
        mime="text/markdown"
    )


# ------------------------------------------------------------------------------------
# Full edges table + downloads for the current view
# ------------------------------------------------------------------------------------
with st.expander("Edges table (current view)"):
    st.dataframe(edges_view, use_container_width=True, hide_index=True)

left, right = st.columns(2)
with left:
    st.download_button(
        "Download edges (CSV)",
        data=to_csv(edges_view, header=["parent","child","kind","skipped"]),
        file_name=f"{view_choice}_edges.csv",
        mime="text/csv"
    )
with right:
    # Also export a 2‑column "FM,action" plan for the current view
    action_rows = []
    for n in sorted(nodes):
        act, why, cat, is_c = classify_action(n)
        action_rows.append((n, cat, act, "Y" if is_c else "N"))
    st.download_button(
        "Download action plan (CSV)",
        data=to_csv(action_rows, header=["fm","category","action","is_custom"]),
        file_name=f"{view_choice}_action_plan.csv",
        mime="text/csv"
    )

# ------------------------------------------------------------------------------------
# Lineage graph (mermaid) for the current view or focused subgraph
# ------------------------------------------------------------------------------------
st.subheader("Lineage graph (current view)")
st.markdown(build_mermaid(edges_view))
