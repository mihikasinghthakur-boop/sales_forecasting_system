from pathlib import Path
import joblib

p = Path('models/best_model.joblib')
print('File exists :', p.exists())
if p.exists():
    size_mb = p.stat().st_size / 1e6
    print(f'File size   : {p.stat().st_size:,} bytes ({size_mb:.2f} MB)')
    model = joblib.load(p)
    print('Model type  :', type(model).__name__)
    step = model.named_steps.get('model')
    print('Estimator   :', type(step).__name__ if step else 'N/A')
    print('Status      : READY FOR PREDICTIONS')
