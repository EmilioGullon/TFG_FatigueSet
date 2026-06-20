import os
import sys
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path

def run_notebook(nb_path: Path):
    print(f"Loading notebook: {nb_path}")
    with open(nb_path, encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Configure path
    jupyter_dir = str(nb_path.parent)
    
    # Add fatigueset-lib to python path for the kernel execution
    lib_path = str(nb_path.parent.parent / "fatigueset-lib")
    os.environ["PYTHONPATH"] = lib_path + os.pathsep + os.environ.get("PYTHONPATH", "")
    
    print("Executing notebook...")
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': jupyter_dir}})
    
    print("Saving notebook in-place...")
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    print("[OK] Notebook executed successfully.")

if __name__ == "__main__":
    notebook_file = Path(r"c:\Users\egull\OneDrive\Documentos\Proyectos\tfg\Jupyters\06_transformer.ipynb")
    run_notebook(notebook_file)
