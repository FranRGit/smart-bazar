import json
import os

nb_path = r'c:\Users\Francis Ramos\Complementos\UNMSM\smart-bazar\Notebooks\Panel 1A_ Auditoria_y_Limpieza_Datos.ipynb'
out_path = r'c:\Users\Francis Ramos\Complementos\UNMSM\smart-bazar\scratch\nb_1a_summary.txt'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

lines = []
for idx, cell in enumerate(nb.get('cells', [])):
    ctype = cell.get('cell_type', '')
    source = "".join(cell.get('source', []))
    lines.append(f"=== CELL {idx} ({ctype}) ===")
    lines.append(source)
    
    if ctype == 'code':
        for out in cell.get('outputs', []):
            otype = out.get('output_type', '')
            if otype == 'stream':
                text = "".join(out.get('text', []))
                if len(text) > 1000:
                    text = text[:1000] + "... [TRUNCATED]"
                lines.append(f"--- OUTPUT (stream) ---\n{text}")
            elif otype in ('execute_result', 'display_data'):
                data = out.get('data', {})
                if 'text/plain' in data:
                    text = "".join(data['text/plain'])
                    if len(text) > 1000:
                        text = text[:1000] + "... [TRUNCATED]"
                    lines.append(f"--- OUTPUT (text/plain) ---\n{text}")

os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(lines))
print("Done writing summary.")
