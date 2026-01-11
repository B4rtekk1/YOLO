import json

notebook_path = r'c:\Users\Bartosz Kasyna\OneDrive - Wydział Edukacji Urząd Miasta Gdańska\Pulpit\YOLO\notebook135b90e9c8.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb.get('cells', [])):
    if cell.get('cell_type') == 'code':
        outputs = cell.get('outputs', [])
        for output in outputs:
            if output.get('output_type') == 'error':
                print(f"Cell {i} failed with error:")
                print(output.get('ename'), ":", output.get('evalue'))
                for line in output.get('traceback', []):
                    print(line)
                print("-" * 40)
            elif 'text' in output:
                text = output['text']
                if 'Error' in text or 'Traceback' in text:
                    print(f"Cell {i} has potential error in stdout/stderr:")
                    print(text)
                    print("-" * 40)
