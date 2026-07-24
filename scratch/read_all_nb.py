import json
import os

notebook_path = r"c:\Users\RootAccess\Documents\Proyecto_Mineria\smart-bazar\Notebooks\Panel 3_ Series temporales.ipynb"
output_path = r"c:\Users\RootAccess\Documents\Proyecto_Mineria\smart-bazar\scratch\all_cells.txt"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open(output_path, 'w', encoding='utf-8') as out:
    for idx, cell in enumerate(nb.get('cells', [])):
        cell_type = cell.get('cell_type')
        source = cell.get('source', [])
        source_str = "".join(source)
        out.write(f"\n======================================\n")
        out.write(f"Cell {idx} ({cell_type})\n")
        out.write(f"======================================\n")
        out.write(source_str)
        out.write("\n")
