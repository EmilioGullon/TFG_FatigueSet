# Plan de Implementación: Desarrollo y Comparación de Modelos de PyTorch para Regresión de Fatiga

Este plan detalla el diseño, la literatura de respaldo, la estructura de archivos y las métricas para implementar una suite de modelos avanzados de Machine Learning y Deep Learning en PyTorch para la regresión de series temporales del dataset FatigueSet, siguiendo la filosofía de modularidad establecida.

---

## 1. Alcance y Decisiones del Diseño

> [!IMPORTANT]
> - **Tipo de Tarea:** Regresión pura. Se mantendrán las variables continuas `fatiga_fisica` y `fatiga_mental` como objetivos de predicción de dos dimensiones `(2,)`.
> - **Métricas de Rendimiento:**
>   - **R² (Coeficiente de Determinación):** Capacidad explicativa de los modelos.
>   - **MAE (Error Absoluto Medio):** Desviación media en escala real de fatiga.
>   - **RMSE (Raíz del Error Cuadrático Medio):** Sensibilidad ante errores grandes.
>   - **Tiempo de entrenamiento:** Eficiencia computacional por época/fold.
>   - **Número de parámetros:** Complejidad y huella del modelo.
> - **Optimización de Hiperparámetros:** Uso de **Optuna** para búsqueda bayesiana.
> - **Verificación:** Un único script de *sanity check* en `experiments/verify_models_sanity.py` que ejecutará una prueba rápida de 1 época por modelo para verificar dimensiones y procesamiento antes del entrenamiento real.

---

## 2. Literatura Científica y Referencias Recomendadas

Para cada modelo se incluirá su paper de referencia en los encabezados del código y notebooks:
1. **Random Forest:** *Breiman, L. (2001). "Random Forests". Machine Learning.* (Excelente baseline no lineal para regresión tabular).
2. **LSTM:** *Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory". Neural Computation.* (Base de dependencias a largo plazo).
3. **GRU:** *Cho, K. et al. (2014). "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation". arXiv.* (Alternativa más ligera con menos parámetros).
4. **CNN + LSTM:** *Shi, X. et al. (2015). "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting". NeurIPS.* (Extracción de patrones locales y posterior secuencia temporal).
5. **TCN (Temporal Convolutional Networks):** *Bai, S., Kolter, J. Z., & Koltun, V. (2018). "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling". arXiv.* (Convoluciones dilatadas y causales eficientes para regresión).
6. **Time Series Transformer:** *Vaswani, A. et al. (2017). "Attention Is All You Need". NeurIPS.* (Adaptado para regresión de series de tiempo con codificación posicional).
7. **PatchTST:** *Nie, Y. et al. (2022). "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers". arXiv.* (Segmentación por parches y canales independientes para predicciones robustas).
8. **xLSTM:** *Beck, M. et al. (2024). "xLSTM: Extended Long Short-Term Memory". arXiv.* (Revisión de LSTM clásica con mecanismos de escala exponencial y matrices de almacenamiento).

---

## 3. Estructura de Directorios Propuesta

```text
fatigueset-lib/
└── fatigueset/
    └── models/
        ├── __init__.py         # Exposición pública de todas las clases de modelos
        ├── rnn.py              # RNN clásica existente
        ├── lstm.py             # Nueva clase LSTM
        ├── gru.py              # Nueva clase GRU
        ├── cnn_lstm.py         # Nueva arquitectura combinada CNN-LSTM
        ├── tcn.py              # Nueva clase TCN (dilated causal convolutions)
        ├── transformer.py      # Nueva arquitectura Transformer
        ├── patchtst.py         # Nueva arquitectura PatchTST
        ├── xlstm.py            # Arquitectura xLSTM (si PyTorch/dependencias lo soportan)
        └── engine.py           # Motor de entrenamiento/evaluación común
experiments/
└── verify_models_sanity.py    # Script de prueba rápida para todos los modelos
Jupyters/
├── 01_random_forest.ipynb      # Notebook explicativo y experimental de Random Forest
├── 02_lstm.ipynb
├── 03_gru.ipynb
├── 04_cnn_lstm.ipynb
├── 05_tcn.ipynb
├── 06_transformer.ipynb
├── 07_patchtst.ipynb
├── 08_xlstm.ipynb
└── experimento_comparativo_optuna.ipynb # Notebook comparativo con optimizador Optuna
```

---

## 4. Interfaces y Arquitectura de Código

### A. Motor Común de Entrenamiento (`engine.py`)
Contendrá funciones desacopladas del modelo:
- `train_step(model, dataloader, loss_fn, optimizer, device)`
- `val_step(model, dataloader, loss_fn, device)`
- `train_model(model, train_loader, val_loader, loss_fn, optimizer, epochs, patience, device)` -> Retorna métricas por época y el mejor modelo.
- `train_kfold_cv(...)` -> Ejecuta validación cruzada GroupKFold por participante y retorna MAE, R², **RMSE** (para fatiga física y mental), **número de parámetros entrenables del modelo**, tiempo de ejecución y métricas detalladas en el archivo JSON.

### B. Especificaciones de los Modelos (PyTorch `nn.Module`)

- **LSTM (`LSTMRegressor`):**
  - Entradas: `(batch_size, seq_len, input_size)`.
  - Parámetros: `hidden_size`, `num_layers`, `dropout`.
  - Salida: `(batch_size, 2)` (a través del estado oculto final `h_n[-1]`).

- **GRU (`GRURegressor`):**
  - Estructura análoga a la LSTM pero con celdas GRU.

- **CNN + LSTM (`CNNLSTMRegressor`):**
  - Capas `nn.Conv1d` para extraer características temporales locales (reducen dimensionalidad temporal).
  - Capa intermedia `nn.LSTM` para capturar la secuencia de features convolucionales.
  - Capa lineal final para la regresión.

- **TCN (`TCNRegressor`):**
  - Capas residuales de convolución 1D causal con dilatación exponencial (dilated convolutions).
  - Permite capturar un campo receptivo amplio sin sufrir desvanecimiento de gradiente.

- **Time Series Transformer (`TSTransformerRegressor`):**
  - Capa de embedding lineal de características.
  - Codificación posicional para conservar el orden secuencial.
  - Bloques de `nn.TransformerEncoderLayer` con atención multicabeza.
  - Pooling temporal y capa lineal final.

- **PatchTST (`PatchTSTRegressor`):**
  - Agrupación del canal temporal en parches disjuntos o solapados (*patching*).
  - Procesamiento independiente de canales (*channel independence*).
  - Auto-atención a nivel de parches.

- **xLSTM (`CustomxLSTMRegressor`):**
  - Variante sLSTM (Stabilized LSTM) con puertas de olvido, entrada y salida exponenciales.
  - Mecanismo de estabilización numérica mediante el seguimiento del máximo $m_t$ y el normalizador $n_t$.
  - Mapeo lineal del último estado oculto $h_t$ a la salida bidimensional `(2,)`.

## User Review Required

> [!IMPORTANT]
> - **sLSTM (Stabilized LSTM) para xLSTM:** Se implementará la variante sLSTM en [xlstm.py](file:///c:/Users/egull/OneDrive/Documentos/Proyectos/tfg/fatigueset-lib/fatigueset/models/xlstm.py) usando puertas exponenciales y estabilización numérica mediante log-normalizador ($m_t$ y $n_t$).
> - **Métricas y Parámetros en engine.py:** Se actualizará [engine.py](file:///c:/Users/egull/OneDrive/Documentos/Proyectos/tfg/fatigueset-lib/fatigueset/models/engine.py) para calcular y registrar explícitamente RMSE de fatiga física y mental, y contar los parámetros entrenables del modelo.
> - **Sanity Check script:** Se desarrollará [verify_models_sanity.py](file:///c:/Users/egull/OneDrive/Documentos/Proyectos/tfg/experiments/verify_models_sanity.py) para testear secuencialmente todos los regresores (incluyendo la nueva xLSTM).
> - **Optuna:** Se instalará `optuna` y se implementará [experimento_comparativo_optuna.ipynb](file:///c:/Users/egull/OneDrive/Documentos/Proyectos/tfg/Jupyters/experimento_comparativo_optuna.ipynb) para búsqueda bayesiana de hiperparámetros.

---

## 5. Plan de Verificación

1. **Pruebas Unitarias de xLSTM:**
   - Crear y ejecutar [test_xlstm.py](file:///c:/Users/egull/OneDrive/Documentos/Proyectos/tfg/tests/test_xlstm.py) para asegurar que las celdas y el regresor de xLSTM realicen pasadas forward/backward sin errores.
2. **Sanity Check Centralizado:**
   - Implementar [verify_models_sanity.py](file:///c:/Users/egull/OneDrive/Documentos/Proyectos/tfg/experiments/verify_models_sanity.py) y ejecutarlo para verificar que todas las clases de modelos instancien, entrenen por 1 época y predigan sin problemas de dimensiones.
3. **Notebooks Individuales y Comparativos:**
   - Implementar [08_xlstm.ipynb](file:///c:/Users/egull/OneDrive/Documentos/Proyectos/tfg/Jupyters/08_xlstm.ipynb) con explicaciones teóricas y entrenamiento rápido de xLSTM.
   - Implementar [experimento_comparativo_optuna.ipynb](file:///c:/Users/egull/OneDrive/Documentos/Proyectos/tfg/Jupyters/experimento_comparativo_optuna.ipynb) realizando una búsqueda con Optuna y tabulando los resultados finales (MAE, R², RMSE, número de parámetros y tiempos).
