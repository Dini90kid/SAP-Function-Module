# streamlit_app/app.py
import io
import os
import sys
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import streamlit as st

# ---------------- Excel writer helper (auto-select engine) ----------------
def workbook_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    """
    Build an Excel workbook in-memory from dataframes.
    Prefers XlsxWriter; falls back to openpyxl; otherwise shows a helpful message.
    """
    import io
    bio = io.BytesIO()

    # Prefer XlsxWriter (great for writing); fall back to openpyxl if not available
    engine = None
    try:
        import xlsxwriter  # noqa: F401
        engine = "xlsxwriter"
    except Exception:
        try:
            import openpyxl  # noqa: F401
            engine = "openpyxl"
        except Exception:
            st.error(
                "Neither **XlsxWriter** nor **openpyxl** is available. "
                "Add one of them to requirements.txt and redeploy."
            )
            return b""

    with pd.ExcelWriter(bio, engine=engine) as xw:
        for name, df in sheets.items():
            # Excel sheet names are max 31 chars
            df.to_excel(xw, sheet_name=name[:31], index=False)

    bio.seek(0)
    st.caption(f"Excel writer engine used: **{engine}**")
    return bio.read()
# ---------------------------------------------------------------------
# Make repo root importable
# ---------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

# Use the same regex as runtime to parse lineage lines
from lakehouse_fm_agent.runtime.lineage import EDGE_RE, SKIP_RE  # type: ignore

APP_TITLE = "SAP FM Analyzer & Converter — Preview"
st.set_page_config(page_title="SAP FM Analyzer & Converter", layout="wide")
st.title(APP_TITLE)

# ---------------------------------------------------------------------
# Helpers: classification, lineage parsing, BFS, workbook writer
# ---------------------------------------------------------------------
STD_PREFIXES = ("R", "RS", "RR", "DD", "ENQUEUE", "DEQUEUE", "TR", "S", "CL_", "SAP")

def is_custom(fm: str) -> bool:
    if not isinstance(fm, str): return False
    u = fm.upper().strip()
    return u.startswith(("Y", "Z", "/"))

def category_for_fm(name: str) -> str:
    if not isinstance(name, str): return "Domain_Logic_Other"
    n = name.upper()
    if n.startswith(("RSD","RSR","RRSI","RSAU","RST","RSW","RSKC","RS_")): return "BW_Platform_API"
    if ("CURRENCY" in n) or ("CURR" in n) or ("UNIT_CONVERSION" in n) or ("UOM" in n) or ("BUOM" in n) or ("SSU" in n): return "UoM_Currency"
    if ("READ_0MATERIAL" in n) or ("READ_PRODFORM" in n) or ("READ_CUSTSALES" in n) or ("READ_ECLASS" in n) or ("READ_MASTER_DATA" in n): return "Masterdata_Readers"
    if ("DATE" in n) or ("MONTH" in n) or ("WEEK" in n) or ("PERIOD" in n) or n in {"SN_LAST_DAY_OF_MONTH","SLS_MISC_GET_LAST_DAY_OF_MONTH","/OSP/GET_DAYS_IN_MONTH","Y_PCM_LAST_DAY_OF_MONTH"}: return "Calendar_Time"
    if ("CONVERSION_EXIT_ALPHA" in n) or ("NUMERIC_CHECK" in n) or ("CHAVL_CHECK" in n) or ("REPLACE_STRANGE_CHARS" in n): return "Data_Cleansing_Validation"
    if "HIER" in n: return "Hierarchy"
    if "PAYTERMDAYS" in n: return "Payment_Terms"
    if "SID" in n or "TEXTS" in n: return "BW_SID_Texts"
    return "Domain_Logic_Other"

POINTERS = {
    "BW_Platform_API": "Replace BW mechanics with Lakehouse patterns: DSO → Delta MERGE; SID → joins by business keys; DDIC → control tables",
    "UoM_Currency": "Table‑driven conversion (FX, currency decimals, UoM factors) + Spark UDFs",
    "Masterdata_Readers": "Join curated dimensions in Unity Catalog; no generic BW readers",
    "Calendar_Time": "Spark date functions + enterprise calendar table",
    "Data_Cleansing_Validation": "Use Spark SQL expressions/UDFs (lpad, regex) instead of exits",
    "Hierarchy": "Parent‑child table + recursive/self‑joins",
    "Payment_Terms": "Model rules in a table; compute with PySpark",
    "BW_SID_Texts": "Keep attributes/texts in Delta and MERGE; no BW writers",
    "Domain_Logic_Other": "Assess & rebuild in PySpark with small reference tables"
}

def action_for_fm(fm: str) -> tuple[str, str]:
    cat = category_for_fm(fm)
    custom = is_custom(fm)
    if custom and cat in {"UoM_Currency","Masterdata_Readers","Hierarchy","Payment_Terms","Domain_Logic_Other"}:
        return "REBUILD", POINTERS.get(cat, POINTERS["Domain_Logic_Other"])
    if cat in {"BW_Platform_API","BW_SID_Texts"}:
        return "PATTERN", POINTERS[cat]
    if cat in {"UoM_Currency","Data_Cleansing_Validation","Calendar_Time"}:
        return "LIGHTWEIGHT", POINTERS[cat]
    if cat == "Masterdata_Readers":
        return "JOIN", POINTERS[cat]
    return ("REBUILD" if custom else "PATTERN"), POINTERS.get(cat, POINTERS["Domain_Logic_Other"])

def parse_lineage_text(text: str):
    edges = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line: continue
        m_skip = SKIP_RE.search(line)
        if m_skip:
            p = m_skip.group("p").strip(); c = m_skip.group("c").strip()
            edges.append((p, c))
            continue
        m = EDGE_RE.search(line.replace("  "," ").replace(" -","-"))
        if m:
            p = m.group("p").strip(); c = m.group("c").strip()
            edges.append((p, c))
    # dedupe (ordered)
    seen, uniq = set(), []
    for e in edges:
        if e not in seen:
            seen.add(e); uniq.append(e)
    return uniq

def edges_from_csv(df: pd.DataFrame):
    cols = {c: str(c).strip().lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=cols)
    for need in ["main_fm","subfm","level2","level_3","level_4"]:
        if need not in df.columns: df[need] = None
    edges = []
    for _, r in df.iterrows():
        chain = [r.get("main_fm"), r.get("subfm"), r.get("level2"), r.get("level_3"), r.get("level_4")]
        chain = [x for x in chain if isinstance(x, str) and x.strip()!='']
        for a,b in zip(chain, chain[1:]):
            edges.append((a.strip(), b.strip()))
    # dedupe
    seen, uniq = set(), []
    for e in edges:
        if e not in seen:
            seen.add(e); uniq.append(e)
    return df, uniq

def degrees(edges):
    from collections import defaultdict
    in_deg, out_deg, nodes = defaultdict(int), defaultdict(int), set()
    for p,c in edges:
        nodes.add(p); nodes.add(c)
        out_deg[p]+=1; in_deg[c]+=1
        in_deg.setdefault(p, in_deg.get(p,0)); out_deg.setdefault(c, out_deg.get(c,0))
    return in_deg, out_deg, nodes

def workbook_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
    bio.seek(0)
    return bio.read()

# ---------------------------------------------------------------------
# Upload: ZIP/TXT/CSV/XLSX
# ---------------------------------------------------------------------
uploaded = st.file_uploader(
    "Upload ZIP (multiple FM run folders with `LINEAGE.txt`), or a single LINEAGE.txt, or your CSV/XLSX",
    type=["zip","txt","csv","xlsx"],
)

if not uploaded:
    st.info("Upload a **.zip** / **.txt** / **.csv** / **.xlsx** to view the full analysis, download the workbook, and use the prompt box.")
    st.stop()

runs_edges = {}          # run_key -> list[(parent, child)]
your_format_df = None    # hold your original table if provided
zip_lineage_paths = []   # diagnostics

if uploaded.name.lower().endswith(".zip"):
    z = ZipFile(io.BytesIO(uploaded.getvalue()))
    # pull ANY file that ends with LINEAGE.TXT (any nesting)
    lineage_members = [n for n in z.namelist() if n.upper().endswith("LINEAGE.TXT")]
    zip_lineage_paths = lineage_members[:]
    if not lineage_members:
        st.error("No `LINEAGE.txt` found in the ZIP.")
        st.stop()
    # parse all lineage files
    for member in lineage_members:
        try:
            text = z.read(member).decode("utf-8", errors="ignore")
        except Exception:
            text = z.read(member).decode("latin-1", errors="ignore")
        parts = [p for p in member.split("/") if p]
        run_key = parts[0] if parts else member
        edges = parse_lineage_text(text)
        if edges:
            runs_edges.setdefault(run_key, []).extend(edges)

elif uploaded.name.lower().endswith(".txt"):
    text = uploaded.getvalue().decode("utf-8", errors="ignore")
    runs_edges["SINGLE_RUN"] = parse_lineage_text(text)

elif uploaded.name.lower().endswith(".csv"):
    df_in = pd.read_csv(uploaded)
    your_format_df, edges = edges_from_csv(df_in)
    runs_edges["FROM_CSV"] = edges

else:  # .xlsx
    x = pd.ExcelFile(uploaded)
    if "Interdependency_Edges" in x.sheet_names:
        ed = x.parse("Interdependency_Edges")
        if {"Parent","Child"}.issubset(ed.columns):
            edges = list(zip(ed["Parent"].astype(str), ed["Child"].astype(str)))
            runs_edges["FROM_XLSX"] = edges
    # keep first sheet as Your_Format if present
    try:
        your_format_df = x.parse("Your_Format")
    except Exception:
        your_format_df = x.parse(x.sheet_names[0])

# diagnostics for ZIP members
if zip_lineage_paths:
    st.caption(f"Found LINEAGE files in ZIP: {len(zip_lineage_paths)}")
    with st.expander("Show first 10 LINEAGE paths found"):
        st.code("\n".join(zip_lineage_paths[:10]))

# Combine edges across runs
combined_edges = []
for e in runs_edges.values():
    combined_edges.extend(e)
# de-dupe combined
seen, uniq = set(), []
for e in combined_edges:
    if e not in seen:
        seen.add(e); uniq.append(e)
combined_edges = uniq

# If still no edges, stop gracefully with guidance
if not combined_edges:
    st.error("No edges parsed from the uploaded file(s). "
             "If you uploaded a ZIP, please ensure each run contains a `LINEAGE.txt` with lines like `PARENT -> CHILD [ FM ]`. "
             "If you uploaded your CSV, ensure columns like `Main FM, SubFM, Level2, Level 3, Level 4` are present.")
    st.stop()

# ---------------------------------------------------------------------
# Build analysis tables (Your Format + rich analysis)
# ---------------------------------------------------------------------
in_deg, out_deg, nodes = degrees(combined_edges)

# 1) Your_Format
def make_your_format(df_source: pd.DataFrame | None, edges: list[tuple[str,str]]):
    cols = ["Main FM","SubFM","Level2","Level 3","Level 4","Type","Remarks","Action Plan","Download Document"]
    if df_source is not None:
        df = df_source.copy()
        # ensure columns exist
        for c in cols:
            if c not in df.columns: df[c] = ""
        # back-fill Type/Action/Remarks
        order = ["Level 4","Level 3","Level2","SubFM","Main FM"]
        def best_name(row):
            for pos in order:
                val = row.get(pos, "")
                if isinstance(val, str) and val.strip(): return val.strip()
            return ""
        for i,row in df.iterrows():
            name = best_name(row)
            if name:
                act, why = action_for_fm(name)
                if not str(df.at[i,"Type"]).strip():
                    df.at[i,"Type"] = "Custom" if is_custom(name) else "Standard"
                if not str(df.at[i,"Action Plan"]).strip():
                    df.at[i,"Action Plan"] = act
                if not str(df.at[i,"Remarks"]).strip():
                    df.at[i,"Remarks"] = why
        return df[cols]
    else:
        # synthesize one from edges (2-level ladder per parent)
        from collections import defaultdict
        kids = defaultdict(set)
        for p,c in edges: kids[p].add(c)
        rows = []
        for p in sorted(kids.keys()):
            for c in sorted(kids[p]):
                # try grandchildren
                gks = sorted(kids.get(c, []))
                if gks:
                    for g in gks:
                        rows.append([p, c, g, "", "", "Custom" if is_custom(g) else "Standard",
                                     POINTERS.get(category_for_fm(g), ""),
                                     action_for_fm(g)[0], ""])
                else:
                    rows.append([p, c, "", "", "", "Custom" if is_custom(c) else "Standard",
                                 POINTERS.get(category_for_fm(c), ""),
                                 action_for_fm(c)[0], ""])
        return pd.DataFrame(rows, columns=cols)

your_format = make_your_format(your_format_df, combined_edges)

# 2) FM_Catalog_ActionPlan (guard for empty)
cat_rows = []
for fm in sorted(nodes):
    act, why = action_for_fm(fm)
    cat_rows.append({
        "FM": fm,
        "Category": category_for_fm(fm),
        "Is_Custom": "Y" if is_custom(fm) else "N",
        "Action": act,
        "Rationale": why,
        "In_Degree": in_deg.get(fm,0),
        "Out_Degree": out_deg.get(fm,0),
        "Criticality": in_deg.get(fm,0)*2 + out_deg.get(fm,0) + (1 if is_custom(fm) else 0),
    })

catalog_cols = ["FM","Category","Is_Custom","Action","Rationale","In_Degree","Out_Degree","Criticality"]
catalog = pd.DataFrame(cat_rows, columns=catalog_cols)
if len(catalog) > 0:
    catalog = catalog.sort_values(["Criticality","In_Degree","Out_Degree"], ascending=False)

# 3) Interdependency_Edges
edges_df = pd.DataFrame(combined_edges, columns=["Parent","Child"])
edges_df["Parent_Category"] = edges_df["Parent"].apply(category_for_fm)
edges_df["Child_Category"]  = edges_df["Child"].apply(category_for_fm)
edges_df["Child_Is_Custom"] = edges_df["Child"].apply(lambda x: "Y" if is_custom(x) else "N")

# 4) Per_Main_Summary (from Your_Format if present)
if your_format_df is not None and "Main FM" in your_format_df.columns:
    mains = [m for m in your_format_df["Main FM"].dropna().unique().tolist() if isinstance(m, str) and m.strip()]
else:
    mains = sorted(set(p for p,_ in combined_edges))

from collections import defaultdict
adj = defaultdict(set)
for p,c in combined_edges: adj[p].add(c)

summary_rows = []
for main in mains:
    direct = sorted(adj.get(main, []))
    # nested up to 3 hops
    seen, frontier, nested = {main}, {main}, set()
    for _ in range(3):
        nxt = set()
        for x in frontier:
            for ch in adj.get(x, set()):
                if ch not in seen:
                    seen.add(ch); nxt.add(ch); nested.add(ch)
        frontier = nxt
        if not frontier: break
    create_list  = sorted(x for x in nested if is_custom(x))
    pattern_list = sorted(x for x in nested if not is_custom(x))
    summary_rows.append({
        "Main_FM": main,
        "Direct_Calls_Count": len(direct),
        "Nested_Upto3_Count": len(nested),
        "Create_Handlers_Count": len(create_list),
        "Pattern_Replace_Count": len(pattern_list),
        "Create_Handlers_List": ", ".join(create_list),
        "Pattern_Replace_List": ", ".join(pattern_list),
    })
main_summary = pd.DataFrame(summary_rows)

# 5) Overall_Build_List
overall_build = catalog.loc[catalog.get("Is_Custom","N")=="Y", ["FM","Category","Action","Criticality"]] \
                       if len(catalog)>0 else pd.DataFrame(columns=["FM","Category","Action","Criticality"])
if len(overall_build)>0:
    overall_build = overall_build.sort_values(["Action","Criticality"], ascending=[True, False])

# 6) Interdependency_Matrix
matrix = pd.crosstab(edges_df["Parent"], edges_df["Child"]) if len(edges_df)>0 else pd.DataFrame()

# ---------------------------------------------------------------------
# Prompt input (embedded in workbook)
# ---------------------------------------------------------------------
st.subheader("Give a prompt for the next step")
default_prompt = ("Generate Master/LLD/Test docs and handler.py for the top 5 custom FMs by criticality. "
                  "Limit nested analysis to depth=2 and include UoM & FX contracts.")
user_prompt = st.text_area("Your prompt", value=default_prompt, height=120)

# ---------------------------------------------------------------------
# Show everything (tabs) and provide Excel download
# ---------------------------------------------------------------------
tabs = st.tabs([
    "Your Format", "FM Catalog / Action Plan", "Per Main Summary",
    "Edges", "Matrix", "Overall Build List"
])

with tabs[0]:
    st.dataframe(your_format, use_container_width=True, hide_index=True)

with tabs[1]:
    st.dataframe(catalog, use_container_width=True, hide_index=True)

with tabs[2]:
    st.dataframe(main_summary, use_container_width=True, hide_index=True)

with tabs[3]:
    st.dataframe(edges_df, use_container_width=True, hide_index=True)

with tabs[4]:
    st.dataframe(matrix, use_container_width=True)

with tabs[5]:
    st.dataframe(overall_build, use_container_width=True, hide_index=True)

# Compose workbook bytes (includes Prompt_Box)
wb_sheets = {
    "Your_Format": your_format,
    "FM_Catalog_ActionPlan": catalog,
    "Per_Main_Summary": main_summary,
    "Interdependency_Edges": edges_df,
    "Interdependency_Matrix": matrix.reset_index() if len(matrix)>0 else pd.DataFrame(),
    "Overall_Build_List": overall_build,
    "Prompt_Box": pd.DataFrame({"Your Prompt Here:":[user_prompt]}),
}
wb_bytes = (lambda sheets: (io.BytesIO() if True else None))  # placeholder for closure
def _wb_bytes(sheets):
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
    bio.seek(0); return bio.read()
wb_bytes = _wb_bytes(wb_sheets)

st.download_button(
    "⬇️ Download analysis workbook (Excel)",
    data=wb_bytes,
    file_name="SAP_FM_Analysis.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# Optional: Doc-seed generator for a selected FM
st.subheader("Doc seeds for a selected FM (optional)")
pick = st.selectbox("Choose FM", options=["<none>"] + sorted(nodes))
if pick != "<none>":
    act, why = action_for_fm(pick)
    md = f"""# {pick} — Master Doc (seed)
## AS-IS
- in_degree={in_deg.get(pick,0)}, out_degree={out_deg.get(pick,0)}
## TO-BE
- Action: **{act}**
- Rationale: {why}
"""
    st.download_button("Download Master.md", data=md, file_name=f"{pick}_MASTER.md", mime="text/markdown")
