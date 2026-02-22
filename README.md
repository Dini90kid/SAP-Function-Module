
#SAP FM Analyzer & Converter (GitHub-ready repo)

A minimal, robust starter that:
- Parses ABAP extractor outputs (e.g., `LINEAGE.txt`).
- Builds Mermaid graphs, a PLANNED object manifest, and runs handlers in topological order.
- Works both as **CLI/Jobs** and a **Streamlit UI** (upload LINEAGE and preview graph).

## Repo layout
```
<repo-root>/
  lakehouse_fm_agent/                  # Python package (must be this exact name)
    __init__.py
    fmtool.py                          # CLI entry (not Streamlit)
    runtime/
      __init__.py  lineage.py  registry.py  manifest.py  context.py
    core/
      __init__.py  alpha.py  currency.py  uom.py  dates.py  cleansing.py
    dims/
      __init__.py  material.py
    bw_replace/
      __init__.py  dso.py  logsys.py
    conf/
      tables.yml  rules.yml
    samples/
      sample_LINEAGE.txt
  streamlit_app/
    app.py                             # Streamlit Main file
  requirements.txt
  .gitignore
```

## Quick start (CLI)
```bash
python -m lakehouse_fm_agent.fmtool --help
python -m lakehouse_fm_agent.fmtool graph --lineage lakehouse_fm_agent/samples/sample_LINEAGE.txt --out graph.mmd
```

## Quick start (Streamlit)
```bash
pip install -r requirements.txt
streamlit run streamlit_app/app.py
```
Then upload a `LINEAGE.txt` to preview.

## Notes
- The CLI uses **absolute imports** and bootstraps `sys.path` so it works regardless of working directory.
- Streamlit main file is **streamlit_app/app.py**, not `fmtool.py`.
