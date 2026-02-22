# streamlit_app/app.py
import os, sys
from pathlib import Path
import streamlit as st

# Ensure repo root is on sys.path so package imports work regardless of CWD
BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from lakehouse_fm_agent.runtime.lineage import load_edges

st.set_page_config(page_title="Lakehouse FM Agent", layout="wide")
st.title("Lakehouse FM Agent — Preview")

up = st.file_uploader("Upload LINEAGE.txt", type=["txt"])

if up is not None:
    # Save uploaded file to a temp path
    txt = up.getvalue().decode("utf-8", errors="ignore")
    tmp = Path("uploaded_LINEAGE.txt")
    tmp.write_text(txt, encoding="utf-8")

    # Parse edges
    edges = load_edges(tmp)
    st.success(f"Parsed edges: {len(edges)}")

    with st.expander("Preview edges"):
        # edges is a list of tuples (parent, child, kind, skipped)
        st.dataframe(edges)

    # --- Mermaid graph: build full block first (no unterminated strings) ---
    mmd_lines = ["graph TD"] + [f'  "{p}" --> "{c}"' for p, c, k, s in edges]
    mermaid_block = "```mermaid\n" + "\n".join(mmd_lines) + "\n```"
    st.markdown(mermaid_block)

else:
    st.info("Upload LINEAGE.txt to preview edges and Mermaid graph.")
