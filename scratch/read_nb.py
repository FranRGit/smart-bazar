import json
import os

notebook_path = r"c:\Users\RootAccess\Documents\Proyecto_Mineria\smart-bazar\Notebooks\Panel 3_ Series temporales.ipynb"

if not os.path.exists(notebook_path):
    print("Notebook path does not exist.")
    exit(1)

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Number of cells: {len(nb.get('cells', []))}")
for idx, cell in enumerate(nb.get('cells', [])):
    cell_type = cell.get('cell_type')
    source = cell.get('source', [])
    source_str = "".join(source)
    
    # Let's print summary of each cell
    print(f"\n--- Cell {idx} ({cell_type}) ---")
    if cell_type == 'code':
        # Print the first few lines of code
        lines = source_str.splitlines()
        print("\n".join(lines[:10]))
        if len(lines) > 10:
            print("...")
    elif cell_type == 'markdown':
        lines = source_str.splitlines()
        print("\n".join(lines[:3]))
        if len(lines) > 3:
            print("...")
