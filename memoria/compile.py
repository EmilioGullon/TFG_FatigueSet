#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de compilación automatizada para la memoria del TFG en LaTeX (ETSIIT - UGR).
Ejecuta la secuencia: pdflatex -> bibtex -> pdflatex -> pdflatex
y reporta el estado y número de páginas resultantes.
"""

import os
import sys
import subprocess
import re

def run_cmd(cmd, desc):
    print(f"\n[Compilación] >>> {desc} ({' '.join(cmd)})...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    result = subprocess.run(cmd, cwd=script_dir, capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        print(f"[Error en {desc}]: Retorno {result.returncode}")
        print(result.stdout[-2000:] if result.stdout else "")
        print(result.stderr[-2000:] if result.stderr else "")
        return False
    else:
        print(f"[Éxito] {desc} completado correctamente.")
        return True

def main():
    tex_file = "main"
    
    # Paso 1: Primer paso pdflatex
    if not run_cmd(["pdflatex", "-interaction=nonstopmode", f"{tex_file}.tex"], "Primer paso pdflatex"):
        sys.exit(1)
        
    # Paso 2: BibTeX
    if not run_cmd(["bibtex", tex_file], "Compilación de bibliografía bibtex"):
        print("[Aviso] Continuando tras paso bibtex...")
        
    # Paso 3: Segundo paso pdflatex
    if not run_cmd(["pdflatex", "-interaction=nonstopmode", f"{tex_file}.tex"], "Segundo paso pdflatex"):
        sys.exit(1)
        
    # Paso 4: Tercer paso pdflatex (resolución final de referencias)
    if not run_cmd(["pdflatex", "-interaction=nonstopmode", f"{tex_file}.tex"], "Paso final pdflatex"):
        sys.exit(1)

    # Extraer número de páginas del archivo .log
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, f"{tex_file}.log")
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            log_content = f.read()
            match = re.search(r"Output written on .*?\((\d+)\s+pages", log_content)
            if match:
                pages = match.group(1)
                print(f"\n=======================================================")
                print(f" ¡DOCUMENTO COMPILADO CON ÉXITO!")
                print(f" Archivo generado: memoria/{tex_file}.pdf")
                print(f" Extensión final: {pages} páginas")
                print(f"=======================================================\n")
            else:
                print(f"\nDocumento generado: memoria/{tex_file}.pdf\n")

if __name__ == "__main__":
    main()
