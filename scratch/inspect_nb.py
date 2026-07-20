import json
from pathlib import Path

nb_path = Path("Notebooks/Panel 1D_Reglas_Asociacion.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    cell_type = cell["cell_type"]
    source_lines = cell["source"]
    first_lines = "".join(source_lines[:3]).strip().replace("\n", " ")
    if len(first_lines) > 80:
        first_lines = first_lines[:77] + "..."
    print(f"[{i:02d}] ({cell_type:8s}): {first_lines}")
