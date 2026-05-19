import subprocess
from pathlib import Path


def test_e2e_run(tmp_path):
    out = tmp_path / 'out'
    cmd = [
        'python', 'experiments/run_models_classicos.py',
        '--data-path', 'fatigueset-lib/data/sample/sample_df.csv',
        '--output-dir', str(out)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert (out / 'results_models.csv').exists()
    assert (out / 'cv_scores.pkl').exists()
