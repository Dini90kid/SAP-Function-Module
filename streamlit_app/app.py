# streamlit_app/app.py
import io
import os
import sys
from pathlib import Path
from zipfile import ZipFile

import streamlit as st

# Ensure repo root is on sys.path so package imports work regardless of CWD
BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

# We reuse the same regex the runtime uses (keeps one source of truth)
from lakehouse_fm_agent.runtime.lineage import EDGE_RE, SKIP_RE  # type: ignore

# ----------------------- Helpers -----------------------
def parse_edges_text(text: str):
    """Parse a LINEAGE.txt-like string into [(parent, child, kind, skipped), ...]."""
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
    """Return a mermaid code block as a single string."""
    mmd_lines = ["graph TD"] + [f'  "{p}" --> "{c}"' for p, c, k, s in edges]
    return "```mermaid\n" + "\n".join(mmd_lines) + "\n```"

def summarize_edges(edges):
    """Compute simple stats."""
    nodes = set()
    for p, c, k, s in edges:
        nodes.add(p); nodes.add(c)
    return {"edges": len(edges), "nodes": len(nodes)}

# ----------------------- UI -----------------------
st.set_page_config(page_title="SAP FM Analyzer & Converter", layout="wide")
st.title("SAP FM Analyzer & Converter — Preview")

uploaded = st.file_uploader(
    "Upload ZIP (multiple FM run folders) or a single LINEAGE.txt",
    type=["zip", "txt"],
)

if not uploaded:
    st.info("Upload a **.zip** containing multiple FM runs (each with `LINEAGE.txt`) "
            "or a single **LINEAGE.txt** to preview combined and per-run lineage graphs.")
    st.stop()

# Storage for parsed results: {run_key: [edges]}
runs = {}

if uploaded.name.lower().endswith(".zip"):
    # Read ZIP in-memory
    z = ZipFile(io.BytesIO(uploaded.getvalue()))

    # Collect all lineage files
    lineage_members = [n for n in z.namelist() if n.upper().endswith("LINEAGE.TXT")]

    if not lineage_members:
        st.error("No `LINEAGE.txt` files found in the uploaded ZIP. "
                 "Please upload the dump produced by your ABAP extractor.")
        st.stop()

    for member in lineage_members:
        try:
            text = z.read(member).decode("utf-8", errors="ignore")
        except Exception:
            # Try latin-1 fallback (rare but happens)
            text = z.read(member).decode("latin-1", errors="ignore")

        # Use the first path segment (top folder) as the run key
        # e.g., "N4MSALES/LINEAGE.txt" -> "N4MSALES"
        parts = [p for p in member.split("/") if p]
        run_key = parts[0] if parts else member

        edges = parse_edges_text(text)
        if edges:
            runs.setdefault(run_key, []).extend(edges)

elif uploaded.name.lower().endswith(".txt"):
    # Single LINEAGE.txt path
    text = uploaded.getvalue().decode("utf-8", errors="ignore")
    runs["SINGLE_RUN"] = parse_edges_text(text)

# If nothing parsed, inform user
if not any(runs.values()):
    st.warning("Uploaded file(s) did not yield any edges. "
               "Check the format of `LINEAGE.txt` (lines like `PARENT -> CHILD [ FM ]`).")
    st.stop()

# Combined edges across all runs
combined = []
for edges in runs.values():
    combined.extend(edges)

# Sidebar: Run selector
choices = ["Combined"] + list(runs.keys())
pick = st.sidebar.selectbox("Select run to visualize", choices, index=0)

# Show stats
if pick == "Combined":
    stats = summarize_edges(combined)
else:
    stats = summarize_edges(runs[pick])

st.write(
    f"**Runs parsed:** {len(runs)}  |  **Nodes:** {stats['nodes']}  |  **Edges:** {stats['edges']}"
)

# Data preview
with st.expander("Preview edges table"):
    if pick == "Combined":
        st.dataframe(combined)
    else:
        st.dataframe(runs[pick])

# Mermaid graph
st.subheader("Lineage graph")
if pick == "Combined":
    st.markdown(build_mermaid(combined))
else:
    st.markdown(build_mermaid(runs[pick]))
