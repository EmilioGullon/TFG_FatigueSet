@echo off
cd /d "%~dp0"
echo =======================================================
echo Compilando memoria de TFG (ETSIIT - Universidad de Granada)...
echo =======================================================

echo [1/4] Paso 1: pdflatex inicial...
pdflatex -interaction=batchmode --miktex-disable-installer main.tex

echo [2/4] Paso 2: bibtex (procesamiento de bibliografia)...
bibtex --miktex-disable-installer main

echo [3/4] Paso 3: pdflatex intermedio...
pdflatex -interaction=batchmode --miktex-disable-installer main.tex

echo [4/4] Paso 4: pdflatex final (resolucion de referencias e indices)...
pdflatex -interaction=batchmode --miktex-disable-installer main.tex

echo =======================================================
echo Compilacion completada con exito.
echo Archivo PDF generado: main.pdf
echo =======================================================
pause
