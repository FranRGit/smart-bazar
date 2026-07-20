import json
from pathlib import Path

nb_path = Path("Notebooks/Panel 1D_Reglas_Asociacion.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

with open("scratch/cells_dump.txt", "w", encoding="utf-8") as out:
    for i, cell in enumerate(nb["cells"]):
        out.write(f"=== CELL {i} ({cell['cell_type']}) ===\n")
        out.write("".join(cell["source"]))
        out.write("\n\n")
