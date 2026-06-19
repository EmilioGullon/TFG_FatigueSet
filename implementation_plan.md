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
- `train_kfold_cv(model_class, model_kwargs, pipeline, seq_len, step, n_splits, opt_lr, epochs, device)` -> Ejecuta validación cruzada GroupKFold por participante y retorna MAE, R², RMSE, tiempo e hiperparámetros.

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

- **xLSTM (`xLSTMRegressor`):**
  - Implementación adaptada de sLSTM (stabilized LSTM) o mLSTM (matrix LSTM) en PyTorch puro para regresión.

---

## 5. Plan de Verificación

1. **Sanity Check:**
   - Ejecutar `verify_models_sanity.py` en el entorno virtual para verificar que todos los modelos se instancien correctamente y realicen un paso forward/backward sin errores de dimensiones en tensores.
2. **Notebooks Individuales:**
   - Validar que cada notebook explique teóricamente el modelo (con citas científicas), muestre su diagrama de flujo, y se ejecute correctamente en un mini-entrenamiento.
3. **Optimización con Optuna:**
   - Validar que el notebook comparativo ejecute la búsqueda bayesiana sobre un subconjunto de datos y guarde la tabla final con las métricas de comparación (MAE, R², RMSE, parámetros, tiempos) para todos los modelos.
