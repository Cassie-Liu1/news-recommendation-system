import json
from pathlib import Path


MOUNT_REPLACEMENT = """from pathlib import Path

# Place the source CSV files in the repository-local data/ directory.
Path("outputs").mkdir(exist_ok=True)
"""

PATH_REPLACEMENT = """data_path = "data/"  # Repository-local source data directory
save_path = "outputs/"  # Repository-local generated-artifact directory
"""


for notebook in Path("notebooks").glob("*.ipynb"):
    payload = json.loads(notebook.read_text())
    replacements = 0
    for cell in payload["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "drive.mount('/content/drive')" in source:
            cell["source"] = MOUNT_REPLACEMENT.splitlines(keepends=True)
            replacements += 1
        elif "/content/drive/My Drive/" in source:
            rest = source[source.find("\n\n# 做召回评估"):]
            cell["source"] = (PATH_REPLACEMENT + rest).splitlines(keepends=True)
            replacements += 1
        cell["outputs"] = []
        cell["execution_count"] = None
    if not replacements:
        raise RuntimeError(f"No personal path was replaced in {notebook}")
    notebook.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    print(f"Sanitized {notebook}")
