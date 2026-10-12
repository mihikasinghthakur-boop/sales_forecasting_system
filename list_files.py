from pathlib import Path

root = Path('.')
SKIP = {'venv', '__pycache__', '.git', 'notebooks', '.agents'}

for p in sorted(root.rglob('*')):
    if any(part in SKIP for part in p.parts):
        continue
    indent = '  ' * (len(p.relative_to(root).parts) - 1)
    if p.is_dir():
        print(f'{indent}[DIR]  {p.name}/')
    else:
        size = p.stat().st_size
        if size > 1_000_000:
            sz = f'{size/1e6:.1f} MB'
        elif size > 1000:
            sz = f'{size/1e3:.1f} KB'
        else:
            sz = f'{size} B'
        print(f'{indent}[FILE] {p.name}  ({sz})')
