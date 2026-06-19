# Resumen Ejecutivo  
En este informe proponemos usar una **RNN clásica tipo Elman** (vanilla RNN) para modelar los targets de fatiga física y mental a partir de las señales multimodales del FatigueSet. Esta RNN secuencial (con activación *tanh* o *ReLU*) procesa ventanas temporales multicanal de los sensores (EEG, ECG, EDA, acelerómetros, etc.), capturando dependencias inmediatas gracias a su **estado oculto** que actúa como memoria interna【56†L264-L272】. Sin embargo, las RNN clásicas sufren el problema del *gradiente desvaneciente*, limitando la memoria a largo plazo【56†L358-L364】【27†L186-L190】. En la implementación en PyTorch se recomienda normalizar las señales por grupos (sesión/participante), usar `torch.manual_seed()` para reproducibilidad【43†L1-L7】, aplicar *dropout* y *clipping* para evitar sobreajuste y explosión de gradientes, y configurar un aprendizaje (por ejemplo, Adam con LR≈1e-3) con early stopping. La evaluación empleará **K-Fold (k=5)** por participante, métricas R² (CV y entrenamiento), MSE y MAE, y contraste bootstrap entre los dos mejores modelos. Se presentará un plan experimental variando hiperparámetros claves (tamaño oculto, capas, bidireccionalidad, dropout, longitud de ventana, batch size) en una tabla comparativa. El informe incluye ejemplos de código en PyTorch, una estimación de recursos (GPU preferible para entrenamiento rápido) y un checklist de integración en el pipeline existente.

## Estructura y formatos del FatigueSet  
El conjunto *FatigueSet* contiene ≈13 horas de datos de **36 sesiones** (12 de baja, media y alta actividad física)【54†L24-L27】. Cada sesión recoge señales multimodales de **14 sensores en 4 dispositivos** corporales: por ejemplo, un wearable “earable” (acelerómetro, giroscopio, PPG), un headband EEG (EEG, acelerómetro, giroscopio), un cinturón torácico (ECG, respiración, acelerómetro, postura) y una pulsera E4 (acelerómetro, pulso sanguíneo, EDA, temperatura)【54†L28-L37】. Además, incluye calificaciones subjetivas de fatiga y resultados de tareas cognitivas (Reaction Time y 2-back)【54†L45-L47】, que sirven como *targets* de fatiga_mental y fatiga_física. La organización de archivos es jerárquica:  
- `fatigueset/[participant_id]/[session_id]/…` con los datos de señales crudas,  
- junto a archivos `README.md`, `metadata.csv`, encuestas previas (`*.xlsx`)【54†L53-L61】.  

Esto sugiere una entrada multi-sesión por participante. Para el preprocesamiento conviene validar la integridad de archivos (detección de nulos/duplicados) y normalizar las señales por grupo (por ejemplo, z-score dentro de cada nivel de actividad) para eliminar sesgos de escala. Las ventanas temporales (sliding windows) deberán alinearse con los metadatos de contextos (e.g. actividad baja/media/alta) y tener etiquetas de diferencia de fatiga (Δ fatiga entre fases) si se requiere.  

## Arquitectura RNN clásica recomendada  
Una **RNN Elman (vanilla)** con una sola dirección es la opción recomendada como base de comparación. En cada paso de tiempo \(t\), la celda recurrente calcula el estado oculto \(h_t\) a partir de la entrada actual \(x_t\) y el estado previo \(h_{t-1}\). Matemáticamente:  
\[
h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{t-1} + b_{hh})\;,
\]  
tal como define PyTorch en `nn.RNN`【37†L627-L634】 (la función por defecto es *tanh*, aunque también puede usarse *ReLU*). La figura 1 ilustra esta idea: la RNN despliega una celda en cada paso temporal, compartiendo pesos entre pasos.  

【45†embed_image】 *Figura 1: Esquema de una arquitectura RNN estándar (celda básica recurrente desplegada en el tiempo, seguida de una capa lineal final).*  

Las RNN clásicas son útiles para captar dependencias locales rápidas, pero **tienen limitaciones**: como observa la literatura, estas redes sufren *gradientes evanescentes* (desvanecimiento) cuando las secuencias son largas【56†L358-L364】. Esto dificulta el aprendizaje de relaciones temporales extensas. Las variantes LSTM/GRU solucionan parcialmente esto introduciendo puertas que conservan información【27†L193-L200】, pero como queremos un modelo “clásico simple”, limitamos la RNN a pocas capas (1–2 capas recurrentes) y tamaños ocultos moderados (e.g. 32–128 neuronas). Se recomienda:  
- **Capas recurrentes**: 1 o 2 capas de `nn.RNN`, con activación *tanh* (potencia centrada)【37†L627-L634】.  
- **Direccionalidad**: probar unidireccional primero; bidireccional aumenta memoria a costa de cómputo.  
- **Dropout/Recurrent Dropout**: introducir *dropout* (ej. 0.1–0.5) entre capas para regularizar.  
- **Inicialización de pesos**: PyTorch inicializa \(W\sim U(-k,k)\) con \(k=1/\text{hidden\_size}\) por defecto【37†L734-L740】. Se puede usar Xavier/Glorot o inicialización ortonormal para mejorar.  
- **Funciones de activación**: además de *tanh*, usar *ReLU* internamente puede mitigar algo la pérdida de gradiente (aunque ReLU conlleva riesgo de gradiente explosivo).  

En resumen, la RNN será una red secuencial sencilla (una o dos capas `nn.RNN` + salida lineal final) diseñada para aprender dinámicas de corto/mid plazo (p.ej. cambios rápidos de fatiga entre fases).  

## Diseño de entrada temporal  
Para alimentar la RNN, se construyen ventanas temporales de las señales sincronizadas: por ejemplo, segmentos de 1–5 segundos (dependiendo de la resolución temporal) con solapamiento (stride) ajustable (p.ej. 50%). Cada ventana es de dimensión \((\text{batch}, \text{sequence\_length}, \text{n\_features})\), donde *n_features* es el número total de señales usadas (todos los canales de acelerómetro, ECG, EEG, EDA, etc.), además de variables cognitivas o contextuales repetidas en cada paso si aplica. Al lidiar con longitudes variables (p.ej. diferentes duraciones de sesión), se puede:  
- **Padding**: rellenar secuencias cortas con ceros hasta longitud fija, usando `pack_padded_sequence()` para informar la longitud real.  
- **Ventanas fijas**: descartar fragmentos finales menores que la ventana.  
- **Batch first**: usar `batch_first=True` para tener formato `(batch, seq, feat)`【37†L679-L687】.  

Es crucial normalizar las características: por ejemplo, escalar cada señal según estadísticas por **grupo** (por sesión o nivel de actividad) para evitar sesgo de origen. Esto mantiene comparables las amplitudes entre sujetos. También conviene alinear temporalmente las tareas cognitivas (targets de fatigabilidad) con las ventanas físicas correspondientes.  

## Preprocesado y trade-offs (features vs crudo)  
Hay dos enfoques complementarios para los datos temporales:  
- **Características agregadas**: el pipeline actual extrae estadísticos (medias, varianzas, bandpower, etc.) de cada señal. Ventaja: reduce dimensionalidad y ruído, fácil de usar con modelos clásicos. Desventaja: pierde la dinámica temporal detallada.  
- **Secuencia cruda**: alimentar directamente la RNN con la serie temporal sin resumir. Ventaja: la RNN puede aprender patrones temporales complejos. Desventaja: requiere más datos y potencia computacional para entrenar bien (más riesgo de sobreajuste).  

Para FatigueSet, se puede combinar ambos: usar ventanas de señales normalizadas como *input*, pero también incluir en la capa final variables estáticas (p.ej. edad, género o estadísticos previos) como entradas adicionales al FC. Este esquema híbrido captura información temporal y contextual. Antes de entrenar, conviene filtrar artefactos de señales (ruido de movimiento, picos anómalos) y manejar valores faltantes (interpolación o eliminación de segmentos sucios) para mejorar la calidad de entrada.  

## Detalles de implementación en PyTorch  
A continuación un esquema de implementación recomendado:  

- **Modelo RNN**: definir una clase `nn.Module` con un `nn.RNN(input_size, hidden_size, num_layers, nonlinearity='tanh', batch_first=True, dropout)` y una capa lineal de salida (`nn.Linear(hidden_size, 2)`, para predecir fatiga física y mental).  

- **Parámetros**: elección típica: `hidden_size=32–128`, `num_layers=1–2`. Agregar `dropout=0.1–0.5` si hay >1 capa. Considerar `bidirectional=True` en una prueba.  

- **Inicialización**: PyTorch inicia los pesos con U(−k,k), k=1/hidden_size【37†L734-L740】. Puede sobreescribirse con `nn.init.xavier_uniform_` o inicialización ortonormal para mayor estabilidad.  

- **Funciones de activación**: la RNN interna usa *tanh* (u opcionalmente *ReLU*)【37†L627-L634】. La capa final será lineal (sin activación) para regresión continua.  

- **Optimización**: usar `torch.optim.Adam(model.parameters(), lr=1e-3)` (o SGD con momentum). Añadir un **scheduler** (p.ej. ReduceLROnPlateau o StepLR) para reducir el *lr* cuando la validación se estanque.  

- **Regularización**: además de *dropout*, aplicar **early stopping** si la métrica (por ejemplo R² en validación) no mejora por varias épocas (paciencia de 10–20). Aplicar **gradiente clipping** (por ejemplo `torch.nn.utils.clip_grad_norm_`) para evitar explosión de gradientes.  

- **Semilla y reproducibilidad**: fijar la semilla global, p.ej. `torch.manual_seed(42)`【43†L1-L7】 y también `np.random.seed(42)`. Desactivar aleatoriedad de CUDA si se desea determinismo total. Documentar en código los `seed` usados.  

- **Batch size y epochs**: típicamente `batch_size=16–64` según memoria GPU. Entrenar 50–100 épocas, monitorizando pérdida y R² en validación. Un número excesivo de épocas puede sobreentrenar, por eso el early stopping.  

```python
# Ejemplo simplificado de modelo en PyTorch:
import torch, torch.nn as nn
class RNNFatiga(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, 
                          nonlinearity='tanh', batch_first=True, dropout=dropout)
        self.fc  = nn.Linear(hidden_size, 2)  # salida: [fatiga_fisica, fatiga_mental]
    def forward(self, x):
        # x: [batch, seq_len, features]
        out, h_n = self.rnn(x)          # out: [batch, seq_len, hidden_size]
        last = out[:, -1, :]           # tomar la última salida en el tiempo
        return self.fc(last)
```

En el entrenamiento, iterar por *epochs* y *batches*, calculando la pérdida (por ej. MSELoss) y actualizando pesos. Registrar métricas de entrenamiento/validación cada época para trazar curvas de aprendizaje.  

## Estrategia de evaluación  
Para comparar objetivamente con el pipeline existente, se seguirá el mismo esquema de validación cruzada estratificada *por participante*. Se sugiere:  

- **K-Fold**: dividir los participantes en 5 grupos, entrenar/validar por turnos (manteniendo sesiones completas juntas para evitar fugas de información entre train/test).  
- **Métricas**: reportar R² (coeficiente de determinación) promedio con su desviación (CV R²) y de entrenamiento, MSE y MAE promedio en validación, análogos a los de modelos clásicos. Además obtener R² global de entrenamiento.  
- **Bootstrap**: aplicar re-muestreo bootstrap entre los dos mejores modelos (según CV) para estimar la significancia de la diferencia en R²【27†L193-L200】. Esto reforzará qué modelo es estadísticamente mejor.  
- **Curvas de aprendizaje**: generar gráficos de pérdida y R² vs época (train vs val) para verificar convergencia o sobreajuste. Idealmente, la validación debería estabilizarse sin divergencia con entrenamiento.  
- **Guardado de modelos/logs**: almacenar los pesos del modelo final de cada fold (p.ej. `model_fold_i.pt`) y logs de métricas (CSV/JSON) para posible análisis posterior.  

Este protocolo replica el usado para KNN, Ridge, etc., permitiendo comparar directamente R²_CV, MSE_CV, MAE_CV y R²_Train con los obtenidos antes.  

## Plan de experimentos  

| Hiperparámetro        | Valores a probar                | Objetivo/Hipótesis                                   |
|-----------------------|---------------------------------|------------------------------------------------------|
| `hidden_size`        | 32, 64, 128                     | A mayor *hidden*, más capacidad de memoria (¿mejora R²?). |
| `num_layers`         | 1, 2                            | Capas adicionales pueden captar patrones más complejos. |
| *Bidireccional*      | False, True                     | Ver si información futura mejora la predicción.      |
| `dropout`            | 0.0, 0.2, 0.5                   | Regularización: evitar sobreajuste con secuencias largas. |
| Longitud de ventana  | 50, 100, 200 (timesteps)        | Más pasos/time = más contexto, pero puede diluir gradientes. |
| `batch_size`         | 16, 32, 64                      | Impacto en estabilidad del entrenamiento y uso de GPU.|
| Tasa de aprendizaje  | 1e-2, 1e-3, 1e-4                | Ajustar convergencia; LR alto converge rápido pero puede oscil.|
| Función activación   | *tanh*, *ReLU*                  | Comprobar si ReLU acelera entrenamiento (posible *exploding*). |

Cada combinación se evaluará vía CV. En especial, comparamos modelos unidireccionales vs bidireccionales y *sin dropout* vs *con dropout*. Se espera que RNN muy profundas (>2 capas) o con secuencias excesivas >200 aumenten el desvanecimiento de gradiente. Estos experimentos se tabulan y gráficas mostrarán, por ejemplo, barras de R² o curvas de validación para cada configuración.  

## Estimación de recursos y tiempos de entrenamiento  
Dado que FatigueSet es relativamente pequeño (13h de datos, decenas de sujetos), una RNN de tamaño moderado entrena rápidamente en GPU. Por ejemplo, con `hidden_size≈64`, `batch_size=32` y secuencias de 100 pasos, puede requerirse *minutos* por época en GPU (p.ej. NVIDIA RTX). En CPU puro tardaría más, decenas de minutos por entrenamiento completo. Con K-Fold=5, el entrenamiento total podría ser de 0.5–2 horas en GPU. El consumo de memoria es bajo (el modelo es compacto), así que una GPU convencional (~4–8GB) es suficiente. Se recomienda GPU para acelerar las iteraciones experimentales; en CPU puede demorarse significativamente.  

## Ejemplos de código en PyTorch  
A continuación un fragmento típico del bucle de entrenamiento y evaluación (simplificado):  
```python
model = RNNFatiga(input_size=n_features, hidden_size=64, num_layers=1, dropout=0.2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
for epoch in range(1, epochs+1):
    model.train()
    for X_batch, y_batch in train_loader:  # X: [B, T, F]
        optimizer.zero_grad()
        y_pred = model(X_batch)           # [B, 2]
        loss = criterion(y_pred, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
    # Validación y cómputo de métricas
    model.eval()
    with torch.no_grad():
        val_loss = 0; mse = 0; mae = 0
        for X_val, y_val in val_loader:
            out = model(X_val)
            val_loss += criterion(out, y_val).item()
        # calcular R² promedio aquí, etc.
    # Registra pérdida, R², ajustar scheduler, early stopping...
```  
Este snippet muestra la inicialización, la propagación hacia adelante/atrás y el cálculo de pérdida. La semilla y DataLoaders quedan definidos fuera de este fragmento.  

## Riesgos y mitigaciones  
- **Gradientes evanescentes/explosivos**: las RNN clásicas pueden olvidar información lejana. Mitigar con secuencias más cortas, activación *tanh* (centra gradientes), gradiente clipping y *batch norm* o *layer norm* si fuera necesario.  
- **Sobreajuste**: pocos datos pueden hacer que la RNN aprenda ruido. Emplear KFold, dropout (0.2–0.5), L2 regularización y early stopping para cortar el entrenamiento oportunamente.  
- **Variabilidad inter-sujeto**: las señales fisiológicas varían mucho entre personas. La normalización por grupo (por sujeto o nivel de actividad) ayuda a que la red no dependa de la escala absoluta. También incorporar metadatos contextuales en el modelo puede reducir sesgos.  
- **Desbalance de datos**: si hay menos ejemplos de un nivel de fatiga, los modelos pueden sesgarse. Se puede re-muestrear ligeramente los batches o usar pesos en la pérdida (si fuera clasificación).  
- **Errores de implementación**: probar con un subconjunto pequeño, revisar dimensiones de tensores (usar `batch_first=True` consistentemente), y escribir tests unitarios para el pipeline (ya existe uno para el flujo completo).  
- **Reproducibilidad**: fijar todas las semillas (`torch.manual_seed`, `np.random.seed`) y documentar configuraciones.  

## Referencias  
- Kalanadhabhatta *et al.* (2021) – Descripción de FatigueSet【54†L24-L27】【54†L28-L37】.  
- IBM Think – *¿Qué es una red neuronal recurrente?* (explica funcionamiento y memoria oculta de RNN)【56†L264-L272】【56†L358-L364】.  
- Wikipedia (es) – *Red neuronal recurrente* (ventajas y problemas de las RNN clásicas, vanishing gradient)【27†L186-L194】【27†L195-L204】.  
- PyTorch Docs – `torch.nn.RNN` (definición, ecuación recurrente, inicialización de pesos)【37†L627-L634】【37†L734-L740】.  
- PyTorch Docs – *Reproducibility* (uso de `torch.manual_seed()` para resultados consistentes)【43†L1-L7】.  
- Papers clásicos sobre RNN/LSTM/TCN para series temporales (p.ej. Hochreiter & Schmidhuber 1997).  

## Checklist reproducible de integración  
- Definir **semillas aleatorias** al inicio (`torch.manual_seed(42)`, `np.random.seed(42)`).  
- Cargar el FatigueSet con el mismo preprocesado: validar integridad, fusionar sensores por usuario-sesión.  
- **Normalizar por grupos** las variables fisiológicas (e.g. z-score por participante/sesión).  
- Generar **ventanas temporales** (longitud fija, stride uniforme) con forma `[batch, seq_len, features]`.  
- Configurar **DataLoader** de PyTorch con `batch_first=True`, barajando los datos si es entrenamiento.  
- Instanciar el modelo RNN con hiperparámetros base (p.ej. `hidden_size=64`, `num_layers=1`).  
- Entrenar usando **K-Fold (k=5)**: en cada fold, dividir data, entrenar, validar y registrar métricas.  
- Aplicar **scheduler** y **early stopping** basados en la métrica de validación (R² o pérdida MSE).  
- Guardar pesos del modelo (`torch.save`) y métricas por fold para análisis y comparación.  
- Repetir experimentos variando los parámetros como planeado, analizando resultados en tablas y gráficos.  

¡Anda, demonio, que ya tienes con qué armar esa RNN! **Que no se te oxide el coco, cabrón.**