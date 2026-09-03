# Ficha Técnica y Propuesta Oficial del TFG (SWAD - Universidad de Granada)

## Información General de la Propuesta

| Campo | Detalle |
| :--- | :--- |
| **Título Oficial Propuesto** | Análisis predictivo de fatiga mediante datos biométricos multimodales |
| **Estudiante** | Emilio Gullón López |
| **Tutor / Director** | Diego Jesús García Gil |
| **Departamento** | [Lenguajes y Sistemas Informáticos (LSI)](http://lsi.ugr.es/) |
| **Centro / Titulación** | ETSIIT — Grado en Ingeniería Informática |
| **Estado de la Propuesta** | Aprobado el 2026-03-06 (21:26) |
| **Asignación** | Sí (N.º de estudiantes: 1) |
| **Tipo de Propuesta** | Proyecto nuevo |
| **Conocimientos Necesarios** | Aprendizaje Automático (Machine Learning / Deep Learning) |

---

## Descripción Oficial de la Propuesta

Este trabajo se centra en el análisis de datos biométricos multimodales del conjunto de datos **FatigueSet** para identificar patrones fisiológicos y evaluar el nivel de fatiga de los usuarios. El proyecto aborda el diseño e implementación de un pipeline completo de datos orientado a series temporales, cubriendo desde la ingesta, limpieza y sincronización de las distintas fuentes de sensores, hasta la extracción de características y su análisis final.

Para la fase de modelado y predicción, se contrastará el rendimiento de algoritmos de machine learning tradicional frente a arquitecturas avanzadas de deep learning especializadas en secuencias. En concreto, se diseñarán e implementarán redes neuronales recurrentes (LSTM) y modelos basados en el mecanismo de atención (Transformers). El objetivo final es evaluar la capacidad de estas tecnologías para capturar dependencias temporales complejas, determinando qué enfoque y combinación de datos multimodales resulta más eficaz para el desarrollo de sistemas de monitorización y prevención de riesgos.

---

## Objetivos Propuestos en SWAD

1. **Pipeline Multimodal:** Diseñar e implementar un pipeline de datos completo para el preprocesamiento, sincronización y estructuración de series temporales multimodales.
2. **Análisis Exploratorio y Caracterización:** Realizar un análisis exploratorio de las señales fisiológicas para extraer características e identificar patrones correlacionados con la fatiga.
3. **Modelado Comparativo:** Desarrollar y entrenar modelos predictivos comparando técnicas de machine learning tradicional con arquitecturas de deep learning, como redes LSTM y Transformers.
4. **Evaluación de Rendimiento:** Evaluar el rendimiento de los modelos y el valor predictivo de cada modalidad sensorial para su potencial integración en sistemas de alerta o monitorización.

---

## Alineación con la Memoria del TFG

La memoria principal amplía de forma natural estos 4 objetivos iniciales de la propuesta oficial en 6 Objetivos Específicos (OE1–OE6) con mayor nivel de detalle técnico, incorporando la evaluación de arquitecturas recurrentes avanzadas (xLSTM/sLSTM) y Modelos Fundacionales (*Time Series Foundation Models* como MOMENT, Chronos y TimesFM) en clúster GPU universitario gestionado con SLURM.
