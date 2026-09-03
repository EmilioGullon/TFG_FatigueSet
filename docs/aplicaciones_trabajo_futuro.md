# Aplicaciones Prácticas y Líneas de Investigación Futura

## Resumen del Contexto Experimental

Este trabajo ha abordado el problema de la **predicción continua de fatiga** —tanto física como mental— a partir de señales fisiológicas multimodales recogidas mediante dispositivos *wearable* (banda de cabeza EEG, pulsómetro de pecho, pulsera de muñeca y sensor auricular PPG/IMU). Se han evaluado y comparado tres familias de modelos:

1. **Modelos clásicos supervisados** (Random Forest, KNN, Lasso, SVM, Regresión Lineal, etc.) entrenados con *features* extraídas de ventanas temporales, usando validación cruzada.
2. **Modelos de aprendizaje profundo para series temporales** (LSTM, GRU, CNN-LSTM, TCN, Transformer, PatchTST, xLSTM) con optimización bayesiana de hiperparámetros mediante Optuna.
3. **Modelos *foundation* de series temporales** (Chronos-T5 y TimesFM) en modo *zero-shot* con sonda lineal.

Los mejores resultados en el conjunto de test interno se obtuvieron con **Random Forest** (R² ≈ 0.94, MAE ≈ 0.88) y con el modelo **Chronos** en predicción probabilística (CRPS física ≈ 10.2, cobertura 90 % ≈ 79 %). Los modelos de *deep learning* entrenados end-to-end mostraron mayor varianza entre *folds*, evidenciando el desafío de la alta variabilidad interpersonal inherente al dataset FatigueSet.

---

## 1. Aplicaciones Potenciales

### 1.1 Salud Laboral y Prevención de Riesgos

La monitorización continua de la fatiga en entornos laborales de alta demanda cognitiva o física representa una de las aplicaciones más directas de los modelos desarrollados. Profesionales como cirujanos, controladores aéreos, operarios de maquinaria pesada o personal de guardia en hospitales están especialmente expuestos a los efectos adversos de la fatiga acumulada. Un sistema de alerta temprana basado en los modelos aquí presentados podría:

- **Detectar de forma no intrusiva** el deterioro del estado fisiológico del trabajador a través de señales EEG, HRV o EDA recogidas por dispositivos *wearable* comerciales.
- **Emitir alertas preventivas** antes de que el nivel de fatiga alcance umbrales críticos, permitiendo la rotación de turnos o la adopción de pausas estratégicas.
- **Reducir el riesgo de accidentes laborales** relacionados con el cansancio, contribuyendo al cumplimiento de normativas de seguridad y salud en el trabajo.

La naturaleza probabilística del modelo Chronos resulta especialmente valiosa en este contexto: proporcionar intervalos de confianza sobre la predicción de fatiga permite calibrar mejor el nivel de alarma sin incurrir en falsas alertas excesivas.

### 1.2 Deporte de Alto Rendimiento y Monitorización de Atletas

La gestión de la carga de entrenamiento es un problema central en el deporte de élite. La fatiga acumulada —tanto física como mental— es el principal factor de riesgo de lesión y sobreentrenamiento. Los modelos desarrollados en este trabajo permiten plantear un sistema de monitorización personalizado que:

- **Modele la evolución temporal de la fatiga** a lo largo de una sesión de entrenamiento o competición, integrando múltiples fuentes fisiológicas (ritmo cardíaco, HRV, acelerómetro, temperatura cutánea).
- **Apoye las decisiones del cuerpo técnico** sobre intensidad y duración del entrenamiento, sustituyendo las percepciones subjetivas (escala RPE) por estimaciones objetivas y continuas.
- **Personalice los planes de recuperación** en función del perfil fisiológico individual del atleta, aprovechando la capacidad de los modelos para capturar dinámicas temporales complejas.

En este contexto, las arquitecturas CNN-LSTM y PatchTST —ambas optimizadas mediante Optuna en este trabajo— son candidatas naturales por su capacidad de modelar tanto patrones locales (convolución) como dependencias a largo plazo (LSTM/Transformer).

### 1.3 Seguridad Vial: Detección de Fatiga al Volante

La fatiga del conductor es una de las principales causas de accidentes de tráfico mortales a nivel mundial. La integración de sistemas *wearable* en el entorno del vehículo podría alimentar modelos como los aquí desarrollados para:

- **Predecir de forma anticipada** el momento en que el conductor alcanzará un nivel de fatiga peligroso, permitiendo al sistema de asistencia a la conducción activar alertas sonoras o recomendar una parada.
- **Complementar los sistemas de visión por computador** (detección de parpadeo o desviación de carril) con información fisiológica interna, aumentando la robustez del sistema ante condiciones de iluminación adversas.
- **Integrarse con vehículos autónomos de nivel 2–3**, donde la intervención humana sigue siendo necesaria y la conciencia situacional del conductor es crítica.

El enfoque multimodal de FatigueSet —que combina señales EEG, ECG, EDA y movimiento— es especialmente adecuado para este escenario, donde la redundancia sensorial incrementa la fiabilidad del sistema.

### 1.4 Tecnología *Wearable* e IoT para Salud Preventiva

El mercado de los dispositivos *wearable* de consumo (smartwatches, auriculares con sensores, anillos inteligentes) ha experimentado un crecimiento exponencial en los últimos años. Los modelos ligeros desarrollados en este trabajo —en particular las variantes LSTM y GRU con pocos parámetros entrenables— son candidatos a ser desplegados en:

- **Dispositivos de bajo consumo energético** con capacidad de inferencia en el borde (*edge inference*), eliminando la necesidad de transmitir datos crudos a la nube.
- **Aplicaciones móviles de bienestar** que ofrezcan al usuario retroalimentación sobre su nivel de fatiga acumulada durante el día, integrando datos de sueño, actividad física y variabilidad cardíaca.
- **Plataformas de salud conectada** que agreguen datos longitudinales de múltiples usuarios para estudios epidemiológicos sobre fatiga crónica, síndrome de *burnout* o trastornos del sueño.

### 1.5 Modelos *Foundation* para Series Temporales Biomédicas

Desde una perspectiva más teórica y de transferencia del conocimiento, este trabajo contribuye a la comprensión de la **transferibilidad de los modelos *foundation* de series temporales al dominio biomédico**. El hecho de que Chronos —entrenado sobre datos financieros, meteorológicos y de demanda energética— logre resultados competitivos en predicción de fatiga fisiológica en modo *zero-shot* abre una línea de investigación de notable relevancia:

- **Adaptación eficiente de modelos *foundation*** mediante técnicas de *fine-tuning* de bajo rango (LoRA, adaptadores), reduciendo el coste de etiquetado de datos clínicos.
- **Evaluación sistemática de la transferibilidad** entre dominios de series temporales, un problema abierto en el área de *transfer learning* para señales fisiológicas.
- **Diseño de *benchmarks* específicos** para modelos *foundation* en salud digital, análogos a los existentes en procesamiento del lenguaje natural (GLUE, SuperGLUE) o visión por computador (ImageNet).

---

## 2. Limitaciones Actuales y Líneas para Superarlas

A continuación se identifican las limitaciones más relevantes observadas durante el desarrollo experimental, acompañadas de propuestas concretas de trabajo futuro para cada una de ellas.

### 2.1 Tamaño y Representatividad del Dataset

El dataset FatigueSet cuenta con un número limitado de participantes (N = 30) y condiciones de fatiga controladas en laboratorio, lo cual limita la generalización de los modelos a poblaciones diversas y contextos del mundo real.

**Línea futura:** Recopilar datasets más extensos y ecológicamente válidos, idealmente en condiciones naturales (*in-the-wild*), con mayor diversidad demográfica (edad, género, condición física) e inclusión de patologías asociadas a la fatiga crónica o al síndrome de *burnout*.

### 2.2 Alta Variabilidad Interpersonal

Los resultados de los modelos de *deep learning* muestran alta varianza entre *folds*, lo que sugiere que los modelos actuales capturan patrones poblacionales pero tienen dificultades para adaptarse a sujetos concretos. Este fenómeno es bien conocido en la literatura de señales fisiológicas y constituye uno de sus retos más persistentes.

**Línea futura:** Explorar enfoques de **aprendizaje personalizado** (*subject-specific fine-tuning*, meta-aprendizaje con MAML o Prototypical Networks) que adapten un modelo base a las características individuales de cada usuario con pocos datos de calibración (*few-shot personalization*).

### 2.3 Dependencia de Sensores Invasivos o Incómodos

La señal EEG requiere un dispositivo de cabeza (Muse S) que, aunque comercial, resulta incómodo para uso cotidiano prolongado. Análogamente, el sensor de pecho (Zephyr BioHarness) requiere un chaleco específico, lo que dificulta su adopción masiva.

**Línea futura:** Realizar un **estudio de ablación por modalidad sensorial** para identificar el subconjunto mínimo de señales que preserva la capacidad predictiva del modelo, priorizando sensores de menor intrusividad (pulsera, auricular) sobre los de mayor comodidad comprometida.

### 2.4 Ausencia de Validación en Tiempo Real

Todos los experimentos realizados son *offline* sobre señales pregrabadas. El despliegue en un entorno real requiere inferencia continua con latencias bajas, manejo robusto de datos perdidos o ruidosos y adaptación a la deriva del sensor (*sensor drift*).

**Línea futura:** Implementar y validar un prototipo de sistema de **inferencia en tiempo real** sobre hardware embebido (Raspberry Pi, microcontroladores ARM Cortex-M) o dispositivos móviles, evaluando cuantitativamente la degradación de rendimiento respecto al escenario *offline*.

### 2.5 Etiquetado Subjetivo de la Fatiga

Las etiquetas de fatiga en FatigueSet provienen de escalas de autoinforme (cuestionarios de percepción subjetiva), que son inherentemente ruidosas y pueden no reflejar el estado fisiológico real del sujeto con total fidelidad.

**Línea futura:** Explorar métodos de **aprendizaje semisupervisado o auto-supervisado** (contrastive learning, masked autoencoders sobre señales fisiológicas) que reduzcan la dependencia de etiquetas subjetivas, extrayendo representaciones latentes de fatiga directamente de las señales sin supervisión externa.

---

## 3. Síntesis y Relevancia del Trabajo

Los resultados obtenidos demuestran que es posible predecir de forma cuantitativa y continua los niveles de fatiga física y mental a partir de señales fisiológicas multimodales, con modelos tan accesibles como Random Forest (R² ≈ 0.94, MAE ≈ 0.88) o tan sofisticados como los *foundation models* probabilísticos (Chronos, CRPS física ≈ 10.2, cobertura 90 % ≈ 79 %). La comparativa sistemática entre familias de modelos —clásicos, *deep learning* y *foundation*— proporciona una visión panorámica y rigurosa del estado del arte aplicado a este problema, y sienta las bases metodológicas para investigaciones futuras en salud digital, sistemas de asistencia y *wearables* inteligentes.

La relevancia de este trabajo trasciende el ámbito puramente académico: en un contexto de creciente preocupación por la salud mental y el bienestar en entornos laborales y deportivos, contar con herramientas objetivas, no invasivas y automáticas para la medición de la fatiga representa un avance significativo con potencial impacto en la calidad de vida de las personas y en la prevención de riesgos asociados al cansancio extremo.
