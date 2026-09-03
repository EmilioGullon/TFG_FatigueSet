# Guía y Documentación Exhaustiva del Preprocesamiento de Datos en FatigueSet

Esta guía técnica detalla paso a paso la arquitectura de preprocesamiento, sincronización y extracción de características fisiológicas implementada en `fatigueset-lib` y desarrollada en los cuadernos interactivos de la carpeta `Jupyters/1.Preprocesado/`.

---

## 1. Fuentes de Datos y Heterogeneidad Multimodal

El dataset FatigueSet recopila 23 canales fisiológicos heterogéneos provenientes de cuatro dispositivos comerciales:

| Dispositivo | Sensor / Archivo | Magnitud Física | $f_s$ Original | Dominio |
| :--- | :--- | :--- | :--- | :--- |
| **Nokia eSense** | `ear_acc_left/right.csv`, `ear_gyro_left/right.csv` | Aceleración y velocidad angular bilateral | 100 Hz | Inercial |
| **Nokia eSense** | `ear_ppg_left/right.csv` | Fotopletismografía de canal auditivo (Verde, IR, Rojo) | 100 Hz | Óptico |
| **Muse S** | `forehead_eeg_raw.csv` | EEG 4 electrodos (TP9, AF7, AF8, TP10) | 256 Hz | Neuroeléctrico |
| **Muse S** | `forehead_eeg_[band]_abs.csv` | Potencia absoluta de bandas $\delta, \theta, \alpha, \beta, \gamma$ | 10 Hz | Espectral |
| **Zephyr BioHarness 3.0** | `chest_raw_ecg.csv` | Electrocardiograma continuo (12 bits) | 250 Hz | Cardíaco |
| **Zephyr BioHarness 3.0** | `chest_raw_breathing.csv` | Onda mecánica de respiración | 25 Hz | Respiratorio |
| **Zephyr BioHarness 3.0** | `chest_physiology_summary.csv` | Resumen HR, BR, HRV-SDNN | 1 Hz | Fisiológico |
| **Empatica E4** | `wrist_bvp.csv` | Fotopletismografía de muñeca (BVP) | 64 Hz | Óptico |
| **Empatica E4** | `wrist_eda.csv` | Actividad Electrodérmica (GSR) | 4 Hz | Conductancia |
| **Empatica E4** | `wrist_skin_temperature.csv` | Temperatura dérmica por infrarrojos | 4 Hz | Térmico |
| **Empatica E4** | `wrist_acc.csv` | Acelerómetro triaxial de muñeca | 32 Hz | Inercial |

---

## 2. Etapas del Pipeline de Preprocesamiento

### 2.1. Ingestión y Manejo de Valores Ausentes
- Los archivos CSV contienen marcas de tiempo UNIX sincronizadas en el inicio de la sesión.
- Se detectan valores anómalos o de saturación de sensor (p. ej., valor 4095 en Zephyr o desajuste de contacto `muse_device_fit > 1`).
- Los valores atípicos puntuales se imputan mediante interpolación monótona lineal local sobre ventanas cortas ($<500\text{ ms}$) o se descartan si el contacto se perdió de forma prolongada.

### 2.2. Remuestreo Polifásico Anti-Aliasing a 64 Hz
Para sincronizar canales que oscilan entre 1 Hz y 256 Hz sin generar artefactos armónicos de aliasing:
- Se utiliza `scipy.signal.resample_poly`.
- Factores de interpolación $P$ y diezmado $Q$ con filtro paso-bajo FIR diseñado con ventana Kaiser.
- Frecuencia temporal unificada de referencia: $f_{\text{target}} = 64\text{ Hz}$.

### 2.3. Filtrado Digital por Modalidad
Se aplican filtros IIR Butterworth de fase cero (`scipy.signal.filtfilt`):
1. **ECG (Zephyr):** Pasa-banda 4º orden ($0.5 - 40.0\text{ Hz}$) para eliminar derivas basales respiratorias y ruidos electromiográficos.
2. **EDA (Empatica E4):** Pasa-bajos 2º orden ($f_c = 1.0\text{ Hz}$) para suavizar la señal tónica y fásica.
3. **EEG Frontal (Muse S):** Filtro Notch IIR a $50\text{ Hz}$ ($Q=30$) + Pasa-banda ($0.5 - 45.0\text{ Hz}$).
4. **BVP (Empatica E4):** Pasa-banda 3º orden ($0.5 - 5.0\text{ Hz}$) centrado en la pulsación arterial.

---

## 3. Segmentación en Ventanas y Prevención de Fuga de Datos (Data Leakage)

- **Longitud de Ventana:** $W = 60\text{ s}$ ($L = 3840$ muestras a 64 Hz).
- **Desplazamiento / Solapamiento:** $\Delta = 30\text{ s}$ (50% de overlap).
- **Protocolo de Validación:** Validación cruzada estratificada por participante (`GroupKFold` con $K=5$).
  - **Importancia Crítica:** Nunca se mezclan ventanas del mismo sujeto entre conjuntos de entrenamiento y prueba para evitar que el modelo memorice la firma biológica del usuario.

---

## 4. Catálogo de Características Matemáticas y Fisiológicas

Para cada ventana de 60 segundos, se extraen descriptores en tres dominios:

### A. Dominio Temporal y Estadístico
- Media ($\mu$), Varianza ($\sigma^2$), Desviación estándar ($\sigma$).
- Asimetría (*Skewness*, $\gamma_1$) y Curtosis (*Kurtosis*, $\gamma_2$).
- Valor Cuadrático Medio (*RMS*), Amplitud Pico a Pico (*PtP*).
- Tasa de cruces por cero (*Zero-Crossing Rate*), Media del valor absoluto (*MAV*).

### B. Dominio Frecuencial / Espectral (Método de Welch)
- Densidad Espectral de Potencia $\hat{S}_{xx}(f)$ con ventana Hanning y 50% de solapamiento.
- **Bandas Cardíacas (HRV):**
  - $\text{VLF} < 0.04\text{ Hz}$
  - $\text{LF}: 0.04 - 0.15\text{ Hz}$ (modulación simpática y barorrefleja)
  - $\text{HF}: 0.15 - 0.40\text{ Hz}$ (tono vagal / parasimpático)
  - $\text{Ratio LF/HF}$ (equilibrio autonómico simpático/vagal)
- **Bandas Cerebrales (EEG Frontal AF7/AF8/TP9/TP10):**
  - Delta ($\delta: 0.5 - 4\text{ Hz}$)
  - Theta ($\theta: 4 - 8\text{ Hz}$)
  - Alfa ($\alpha: 8 - 12\text{ Hz}$)
  - Beta ($\beta: 12 - 30\text{ Hz}$)
  - Gamma ($\gamma > 30\text{ Hz}$)
  - Ratios cognitivos: $\theta/\beta$ e $(\theta + \alpha)/(\alpha + \beta)$

### C. Dominio No Lineal y Dinámica Caótica
- **Entropía Muestral ($\text{SampEn}$):** Dimensión de incrustación $m=2$, tolerancia $r = 0.2 \cdot \sigma$.
- **Geometría de Poincaré (HRV):** Descriptores $SD_1$, $SD_2$ y ratio $SD_1/SD_2$.

### D. Normalización Robusta
- `RobustScaler`: $\tilde{x} = \frac{x - \text{mediana}(x)}{Q_{75}(x) - Q_{25}(x)}$ para neutralizar outliers de sensores vestibles.
