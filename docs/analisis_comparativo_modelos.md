# Informe Técnico: Análisis Comparativo Global de Modelos en FatigueSet

Este informe sintetiza la evaluación empírica de 12 modelos evaluados bajo el protocolo riguroso `GroupKFold` de 5 particiones sobre el dataset FatigueSet.

---

## 1. Tabla Resumen Global de Rendimiento

| Modelo | Familia / Paradigma | MAE Global | MAE Física | MAE Mental | Parámetros | Tiempo Total | Aptitud Despliegue |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Custom GRU** | Recurrente Deep Learning | **16.99** | 16.12 | 17.86 | 568k | $\approx$ 1.96 h | **Óptima (Edge/Móvil)** |
| **Custom xLSTM (sLSTM)** | Recurrente Estabilizada | **17.33** | 16.48 | 18.18 | 625k | $\approx$ 3.57 h | **Muy Buena (Edge)** |
| **Custom LSTM** | Recurrente Clásica | 17.50 | 16.65 | 18.35 | 887k | $\approx$ 50.7 min | Muy Buena |
| **Custom CNN-LSTM** | Híbrido Convolucional-LSTM | 17.56 | 16.70 | 18.42 | 84k | $\approx$ 53.0 min | **Excelente (Micro)** |
| **Custom TCN** | Convolución Dilatada Causal | 17.70 | 16.95 | 18.45 | 175k | $\approx$ 5.0 min | **Excelente (Baja Latencia)** |
| **Custom Transformer** | Auto-Atención Multi-Cabeza | 18.25 | 17.40 | 19.10 | 396k | $\approx$ 12.8 min | Media |
| **Custom PatchTST** | Transformer por Parches | 19.44 | 18.20 | 20.68 | 77k | $\approx$ 14.4 min | Buena |
| **Custom RNN** | Recurrente Básica (Baseline) | 30.09 | 28.50 | 31.68 | 1.89k | $\approx$ 18.5 s | No Recomendada |
| **Chronos-T5 (Base)** | Foundation Decoder Prob. | 16.53 | **14.11** | 18.95 | 710M | $\approx$ 20.7 min | Servidor / Cloud |
| **TimesFM 2.5 (200M)** | Foundation Decoder Parches | 19.38 | 16.39 | 22.38 | 200M | $\approx$ 8.9 min | Servidor / GPU |
| **MOMENT-1-large** | Foundation Encoder-Only | 30.66 | 24.64 | 36.68 | 341M | $\approx$ 86.8 min | No Recomendada (Zero-Shot) |
| **Random Forest (Agg)** | Ensamble Árboles Tabular | 21.30 | 20.10 | 22.50 | --- | $<$15 s | Media (Pierde secuencias) |
| **KNN (k=3)** | Basado en Instancias Tabular | 23.80 | 22.50 | 25.10 | --- | $<$5 s | Baja |

---

## 2. Hallazgos Principales

1. **Superioridad de las Redes Recurrentes con Compuertas:**
   - La arquitectura **Custom GRU** logra el mejor equilibrio global con un $\text{MAE} = 16.99$, mientras que **Custom xLSTM** demuestra una gran estabilidad numérica con $\text{MAE} = 17.33$ y una variabilidad inter-fold reducida ($\sigma = 1.09$).
   - Su sesgo inductivo de actualización de estado recurrente continuo es óptimo para capturar la acumulación lenta y progresiva de la fatiga.

2. **Divergencia entre Fatiga Física y Fatiga Mental:**
   - La fatiga física es más predecible debido al fuerte acoplamiento de señales cardíacas (HR/HRV) y respiratorias con el esfuerzo motor ($\text{MAE} \approx 14 - 16$).
   - La fatiga mental involucra bioseñales más ruidosas (EEG frontal y picos fásicos EDA) con mayor variabilidad subjetiva entre sujetos ($\text{MAE} \approx 18 - 22$).

3. **Comportamiento de los Modelos Fundacionales:**
   - **Chronos-T5** muestra una excelente capacidad en fatiga física ($\text{MAE} = 14.11$ en régimen *zero-shot*), pero con un coste computacional 1000 veces superior a las redes recurrentes.
   - **TimesFM 2.5** ofrece inferencia rápida con soporte de parches, pero requiere fine-tuning específico en bioseñales para cerrar la brecha en fatiga mental.
