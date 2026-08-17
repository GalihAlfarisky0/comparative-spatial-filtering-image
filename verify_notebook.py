import json
from pathlib import Path

p = Path("c:/Users/Galihmuhhamad/OneDrive/Documents/SEMESTER 5/Pengolah  sinyal data/FINAL PROJECT/Comparative_Spatial_Filtering_Image2.ipynb")
nb = json.loads(p.read_text(encoding="utf-8"))
ns = {}
for i, cell in enumerate(nb["cells"], 1):
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", []))
    try:
        exec(compile(src, f"<cell {i}>", "exec"), ns)
        print(f"Cell {i}: OK")
    except Exception as e:
        print(f"Cell {i}: ERROR {type(e).__name__}: {e}")
        raise
print("OVERALL: PASS")
