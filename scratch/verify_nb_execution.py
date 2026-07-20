import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from IPython.display import display
except ImportError:
    def display(*args, **kwargs):
        for arg in args:
            print(arg)

nb_path = Path("Notebooks/Panel 1D_Reglas_Asociacion.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Verificando {len(nb['cells'])} celdas...")

global_namespace = {"display": display}

for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        code = "".join(cell["source"])
        # Reemplazar plt.show(), fig.show() y !pip por pass
        code_test = code.replace("plt.show()", "pass").replace("fig.show()", "pass")
        code_test = code_test.replace("!pip install mlxtend", "pass")
        print(f"Ejecutando celda de codigo {idx}...")
        try:
            exec(code_test, global_namespace)
            print(f"[OK] Celda {idx} ejecutada con exito.")
        except Exception as e:
            print(f"[ERROR] Error en celda {idx}: {e}")
            raise e

print("\n[EXITO] Todas las celdas de codigo del Notebook Panel 1D se ejecutaron y verificaron sin ningun error!")
