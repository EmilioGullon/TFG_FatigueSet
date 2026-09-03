# Análisis Predictivo de Fatiga Mediante Datos Biométricos Multimodales

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Optuna](https://img.shields.io/badge/Optuna-Bayesian%20HPO-blueviolet.svg?logo=optuna&logoColor=white)](https://optuna.org/)
[![LaTeX](https://img.shields.io/badge/LaTeX-MiKTeX%20%7C%20TeXLive-green.svg?logo=latex&logoColor=white)](https://www.latex-project.org/)
[![Status](https://img.shields.io/badge/Status-Completed%20%28120%20pages%29-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

**Trabajo de Fin de Grado en Ingeniería Informática**  
*Mención en Computación y Sistemas Inteligentes*  
**Escuela Técnica Superior de Ingenierías Informática y de Telecomunicación (ETSIIT)**  
**Universidad de Granada (UGR)**

---

**Autor:** Emilio Gullón López  
**Director:** D. Diego Jesús García Gil  
**Departamento:** Lenguajes y Sistemas Informáticos (LSI)  
**Curso Académico:** 2025 / 2026  

</div>

---

## 📋 Resumen Ejecutivo

La fatiga humana, tanto en su manifestación física como cognitiva o mental, constituye uno de los factores de riesgo más críticos en la degradación del rendimiento laboral, los siniestros laborales y los accidentes de transporte. 

Este Trabajo de Fin de Grado aborda el **modelado y predicción continua y no invasiva del nivel de fatiga** a partir de señales fisiológicas y biomecánicas multimodales procedentes de dispositivos portables (*wearables*), utilizando el conjunto de datos de referencia **FatigueSet** (12 sujetos bajo protocolos experimentales controlados).

A lo largo del proyecto se implementa y valida un pipeline completo de Ciencia de Datos e Inteligencia Artificial:
1. **Sincronización temporal multimodal y armonización de frecuencias dispares** (desde 1 Hz hasta 256 Hz) mediante interpolación cúbica segmentada y ventaneo deslizante (*sliding window*) con prevención estricta de fuga de datos (*data leakage*).
2. **Desarrollo de la librería `fatigueset-lib`:** Framework en Python modular, desacoplado y orientado a objetos que implementa cargadores de datos, transformaciones fisiológicas, arquitecturas neuronales y rutinas de evaluación estandarizadas.
3. **Benchmarking experimental exhaustivo** comparando tres paradigmas computacionales:
   - **Modelos Clásicos de Machine Learning:** Random Forest, SVM, KNN, Regresión Lineal regularizada (Ridge, Lasso, ElasticNet).
   - **Modelos de Aprendizaje Profundo Nativos:** LSTM, GRU, CNN-LSTM, Redes Convolucionales Temporales (TCN), Transformers, PatchTST y arquitecturas recurrentes de última generación (**xLSTM / sLSTM**).
   - **Modelos Fundacionales para Series Temporales (*Time Series Foundation Models*):** Chronos-T5 (Amazon), TimesFM 2.5 (Google) y MOMENT-1-large (Carnegie Mellon / Auton Lab), evaluados bajo paradigmas *zero-shot* y *fine-tuning*.
4. **Optimización Hiperparamétrica Bayesiana con Optuna (TPE):** Búsqueda multivariable de hiperparámetros estructurales y optimizadores adaptativos (Adam, AdamW, RMSprop, SGD).
5. **Memoria Académica Oficial:** Documento formal de 120 páginas maquetado en LaTeX según la normativa de la ETSIIT-UGR.

---

## 🔬 Principales Contribuciones

* **Prevención rigurosa de Data Leakage:** Formulación de esquemas de validación cruzada por bloques temporales (*Temporal Purged Block K-Fold*) que impiden que ventanas solapadas contaminen los conjuntos de prueba.
* **Librería modular `fatigueset-lib`:** Código empaquetado bajo estándares PEP8, con tipado estático, cobertura de pruebas unitarias (`pytest`) y desacoplamiento siguiendo los principios SOLID.
* **Evaluación del estado del arte:** Primer estudio comparativo que enfrenta arquitecturas recurrentes avanzadas (xLSTM) y modelos fundacionales preentrenados en series fisiológicas multicanal continuas.
* **Compromiso precisión-eficiencia:** Caracterización cuantitativa del coste computacional (latencia de inferencia, parámetros entrenables y huella en memoria) frente a la precisión predictiva para dispositivos de recursos limitados (*Edge Computing*).

---

## 📊 Resultados Experimentales Destacados

Todos los modelos se evaluaron mediante validación cruzada estratificada temporal de 5 pliegues (*5-Fold Time-Series CV*). Las métricas principales reportadas son el Error Absoluto Medio (MAE) y la Raíz del Error Cuadrático Medio (RMSE) sobre escala normalizada [0, 100].

### 1. Modelos de Aprendizaje Profundo Nativos (Optimizados con Optuna)

| Arquitectura | Target | MAE Medio ± Std | RMSE Medio ± Std | Optimizador | Nº Parámetros |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Custom GRU** | **Fatiga Física** | **13.94 ± 3.71** | **16.86 ± 3.54** | Adam | 568K |
| **Custom xLSTM** | **Fatiga Física** | **14.46 ± 3.41** | **17.54 ± 3.59** | RMSprop | 625K |
| **Custom TCN** | **Fatiga Física** | 14.79 ± 3.45 | 18.27 ± 3.91 | RMSprop | 175K |
| **Custom LSTM** | **Fatiga Física** | 15.06 ± 4.58 | 17.71 ± 4.22 | SGD | 887K |
| **Custom CNN-LSTM** | **Fatiga Física** | 15.10 ± 4.17 | 17.91 ± 3.90 | AdamW | 84K |
| **Custom PatchTST** | **Fatiga Física** | 15.68 ± 5.51 | 17.87 ± 5.36 | AdamW | 77K |
| **Custom Transformer** | **Fatiga Física** | 15.79 ± 5.32 | 18.31 ± 4.91 | SGD | 396K |
| **Custom LSTM** | **Fatiga Mental** | **19.93 ± 7.94** | **23.19 ± 8.84** | Adam | 120K |
| **Custom CNN-LSTM** | **Fatiga Mental** | 20.02 ± 7.87 | 23.32 ± 8.83 | AdamW | 807K |
| **Custom GRU** | **Fatiga Mental** | 20.04 ± 8.14 | 23.41 ± 8.96 | Adam | 58K |
| **Custom xLSTM** | **Fatiga Mental** | 20.19 ± 7.96 | 23.45 ± 8.89 | SGD | 287K |
| **Custom TCN** | **Fatiga Mental** | 20.60 ± 6.27 | 24.77 ± 7.28 | Adam | 134K |
| **Custom Transformer** | **Fatiga Mental** | 20.70 ± 8.02 | 23.89 ± 8.88 | RMSprop | 130K |
| **Custom PatchTST** | **Fatiga Mental** | 23.19 ± 8.55 | 26.88 ± 9.04 | SGD | 655K |

### 2. Modelos Fundacionales para Series Temporales

| Modelo | Paradigma | MAE Física | RMSE Física | CRPS Física | MAE Mental | RMSE Mental | CRPS Mental | Cobertura 90% |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Chronos-T5-base** | Zero-shot + Linear | **14.11** | **17.03** | **10.17** | **18.95** | **22.22** | **13.69** | **78.9%** |
| **TimesFM 2.5 (200M)** | Zero-shot + Linear | 16.39 | 20.15 | 12.53 | 22.38 | 26.71 | 17.30 | 68.7% |
| **MOMENT-1-large** | Fine-tuning Cabeza | 24.64 | 28.98 | — | 36.68 | 41.37 | — | — |

### 💡 Conclusiones Experimentales Principales:
1. **Supremacía recurrente compacta:** Las arquitecturas recurrentes optimizadas (**Custom GRU** y **Custom xLSTM**) alcanzan el mejor equilibrio entre rendimiento predictivo y eficiencia computacional, superando a los Transformers densos en series fisiológicas continuas.
2. **Capacidad predictiva de los Foundation Models:** **Chronos-T5** demostró una notable transferencia *zero-shot* (MAE = 14.11 en fatiga física y 18.95 en mental), con la ventaja añadida de cuantificar la incertidumbre predictiva mediante distribuciones de probabilidad (CRPS = 10.17). Sin embargo, su coste computacional es ~20x superior al de los modelos nativos.
3. **Limitación de los modelos tabulares clásicos:** Las técnicas clásicas sobre características agregadas no logran capturar la dinámica transitoria temporal ni los patrones micro-fisiológicos de variabilidad cardíaca o respuesta galvánica.

---

## 🗂️ Estructura del Repositorio

```text
├── memoria/                    # Documentación oficial de la memoria académica (LaTeX)
│   ├── main.tex                # Documento maestro
│   ├── main.pdf                # Memoria completa compilada (120 páginas, apta para entrega)
│   ├── chapters/               # Capítulos modulares (00_frontmatter a 08_conclusiones)
│   ├── appendices/             # Apéndices A a E (licencias, manuales, SLURM, glosario)
│   ├── bib/references.bib      # Base de datos bibliográfica en BibTeX (>80 referencias)
│   ├── config/preamble.tex     # Paquetes, tipografía y estilos visuales
│   ├── figures/                # Diagramas vectoriales y figuras de resultados a 300 DPI
│   ├── compile.py              # Script automatizado de compilación LaTeX en Python
│   └── compile.bat             # Script de compilación para entornos Windows
│
├── fatigueset-lib/             # Librería Python desarrollada para el proyecto
│   ├── pyproject.toml          # Configuración del paquete estándar
│   ├── setup.cfg               # Metadatos del instalador
│   ├── fatigueset/             # Código fuente del módulo
│   │   ├── models/             # Implementaciones PyTorch (LSTM, GRU, CNN-LSTM, TCN, xLSTM, etc.)
│   │   ├── preprocessing/      # Filtros (Butterworth), normalización y ventaneo
│   │   ├── dataset/            # PyTorch Dataset y DataLoaders multicanal
│   │   └── evaluation/         # Métricas de regresión e intervalos probabilísticos
│   └── tests/                  # Batería de pruebas unitarias automatizadas
│
├── fatigueset/                 # Conjunto de datos experimental FatigueSet
│   ├── 01/ ... 12/             # Registros por sujeto y sesión (eSense, Muse, Zephyr, Empatica)
│   ├── metadata.csv            # Orden de intensidades y sesiones
│   └── *.xlsx                  # Cuestionarios pre-tarea y demográficos
│
├── Jupyters/                   # Cuadernos interactivos y experimentación reproducible
│   ├── 1.Preprocesado/         # Sincronización, alineación temporal y feature engineering
│   ├── 01_random_forest.ipynb  # Modelos clásicos tabulares
│   ├── 02_lstm.ipynb ... 08    # Cuadernos individuales por arquitectura deep learning
│   ├── 09_foundation_*.ipynb   # Evaluación de Chronos-T5, MOMENT y TimesFM
│   ├── optuna_v2.db            # Base de datos SQLite con los trials de Optuna
│   └── *.sh                    # Scripts para el clúster GPU mediante SLURM
│
├── models/                     # Modelos y pesos entrenados para inferencia directa
│   ├── classicos/              # Modelos scikit-learn (.pkl)
│   ├── deep_learning/          # Checkpoints PyTorch de las arquitecturas nativas (.pt)
│   └── rnn/                    # Checkpoints del baseline recurrente
│
├── output/                     # Resultados consolidados y artefactos de visualización
│   ├── optuna_v2/              # Curvas de aprendizaje, convergencia bayesiana y radar charts
│   ├── full_run/               # Resultados consolidados de validación cruzada
│   └── *.csv, *.json, *.png    # Tablas de resultados definitivas citadas en la memoria
│
├── docs/                       # Documentación complementaria y propuesta oficial
└── scripts/                    # Scripts de generación de gráficos para la memoria
```

---

## 🛠️ Instalación y Uso

### 1. Clonar el Repositorio
```bash
git clone https://github.com/EmilioGullon/TFG_FatigueSet.git
cd TFG_FatigueSet
```

### 2. Crear y Activar el Entorno Virtual
```bash
python -m venv .venv

# En Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# En Linux/macOS:
source .venv/bin/activate
```

### 3. Instalar la Librería `fatigueset-lib`
La librería del proyecto puede instalarse en modo editable para desarrollo:
```bash
pip install --upgrade pip
pip install -e fatigueset-lib
```

Para verificar que la instalación es correcta, ejecute los tests unitarios:
```bash
pytest fatigueset-lib/tests
```

### 4. Ejemplo Rápido de Uso de la Librería
```python
import torch
from fatigueset.models import CustomGRU, CustomxLSTM

# Instanciación de una arquitectura recurrente multicanal (23 señales a 64 Hz)
model = CustomGRU(
    input_size=23,
    hidden_size=128,
    num_layers=2,
    dropout=0.2,
    output_size=1
)

# Tensor de entrada: [batch_size=16, seq_len=640, channels=23]
x = torch.randn(16, 640, 23)
prediccion_fatiga = model(x)
print("Forma de la predicción:", prediccion_fatiga.shape)  # [16, 1]
```

---

## 📄 Compilación de la Memoria (LaTeX)

La memoria académica oficial se encuentra lista para su compilación en [memoria/](file:///c:/Users/egull/OneDrive/Documentos/Proyectos/tfg/memoria).

### Requisitos:
* Distribución de LaTeX instalada (**MiKTeX** o **TeX Live**).
* Compilador `pdflatex` y utilidades de bibliografía (`bibtex`).

### Compilar el PDF Oficial:
Puede compilarse con el script auxiliar multiplataforma:
```bash
python memoria/compile.py
```
O directamente con `pdflatex`:
```bash
cd memoria
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```
El documento generado [main.pdf](file:///c:/Users/egull/OneDrive/Documentos/Proyectos/tfg/memoria/main.pdf) (120 páginas) cumple todas las normativas formales de la Universidad de Granada.

---

## 📖 Estructura de la Memoria Oficial

1. **Frontmatter:** Portada reglamentaria ETSIIT, créditos y licencias, visto bueno del director, autorizaciones de biblioteca y resúmenes estructurados en español e inglés.
2. **Capítulo 1: Introducción y Motivación:** Planteamiento del problema, taxonomía de paradigmas y formulación de hipótesis científicas.
3. **Capítulo 2: Estado del Arte:** Fundamentos biomédicos de la fatiga, procesamiento de series temporales y evolución arquitectónica (RNNs a Foundation Models).
4. **Capítulo 3: Conjunto de Datos y Preprocesamiento Fisiológico:** Armonización temporal, filtrado digital de artefactos, ventaneo y estrategias contra el *data leakage*.
5. **Capítulo 4: Arquitectura y Diseño del Framework fatigueset-lib:** Principios SOLID, patrones de diseño de software y catálogo de modelos.
6. **Capítulo 5: Experimentación y Benchmarking:** Protocolo experimental, infraestructura computacional en clúster GPU y optimización bayesiana con Optuna.
7. **Capítulo 6: Resultados y Discusión:** Análisis cuantitativo de modelos, métricas probabilísticas, trade-offs de eficiencia y validación de hipótesis.
8. **Capítulo 7: Planificación Temporal y Presupuesto Económico:** Metodología ágil, cronograma Gantt, presupuesto desglosado y análisis de amortización.
9. **Capítulo 8: Conclusiones y Trabajos Futuros:** Síntesis de logros, implicaciones éticas, privacidad biométrica y líneas futuras de investigación.
10. **Apéndices:** Licencias de software, manual técnico de `fatigueset-lib`, hiperparámetros óptimos, scripts SLURM y glosario de acrónimos.

---

## 📜 Cita Académica

Si utilizas el código, los modelos o los resultados de este Trabajo de Fin de Grado en tu investigación, por favor cita:

```bibtex
@mastersthesis{gullon2026fatiga,
  author       = {Emilio Gull{\'o}n L{\'o}pez},
  title        = {An{\'a}lisis predictivo de fatiga mediante datos biom{\'e}tricos multimodales},
  school       = {Escuela T{\'e}cnica Superior de Ingenier{\'i}as Inform{\'a}tica y de Telecomunicaci{\'o}n, Universidad de Granada},
  year         = {2026},
  type         = {Trabajo de Fin de Grado en Ingenier{\'i}a Inform{\'a}tica},
  note         = {Dirigido por D. Diego Jes{\'u}s Garc{\'i}a Gil}
}
```

---

## ⚖️ Licencia y Reconocimientos

* **Memoria Académica y Documentación:** Publicada bajo licencia [Creative Commons Reconocimiento-NoComercial-CompartirIgual 4.0 Internacional (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.es).
* **Código Fuente y Librería (`fatigueset-lib`):** Distribuido bajo [Licencia MIT](https://opensource.org/licenses/MIT).
* **Dataset FatigueSet:** El conjunto de datos FatigueSet fue recopilado y publicado por Nokia Bell Labs (Kalanadhabhatta et al., *Pervasive Health 2021*).
