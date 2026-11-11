# Entrenamiento en AMD MI300X (22-nov-2026)

Una MI300X de AMD Developer Cloud (ROCm, PyTorch 2.15, ultralytics 8.4.191), de 16:11 a ~23:30 UTC, a
1,99 USD/h. Todo lo de abajo se midió con `evaluar.py` sobre `splits.csv` (grupos y dHash, test congelado).
Los JSON completos están en `informes/eval_*.json`.

Un acierto es una caja con IoU ≥ 0,5 contra una etiqueta de la misma clase; AP50 con 101 puntos de recall.

## Datos

| Split | Fotos | Origen |
|---|---:|---|
| train | 6.507 | Bisman, Brasil, Bucko y Egipto, después de la limpieza |
| val | 1.222 | los mismos datasets; aquí se eligen umbrales y modelos |
| test | 2.721 | los mismos más HRP4K (solo test); se mide una vez al final |

- **Limpieza** (revisión de etiquetas del 1-nov):
  - Salen del train 619 fotos: 596 montajes de Egipto, 11 duplicados exactos, 11 con superposiciones y 1 sin
    pavimento. Están en `entrenamiento/limpieza/excluir_train.csv` y figuran en `splits.csv` como "excluida".
  - Se corrigen 14 etiquetas, 9 de train y 5 de val: `entrenamiento/limpieza/aplicar_correcciones.py`.
- **Fuga conocida:** en Bucko todo es un mismo camino, en Brasil las cuadras comparten calles y Bisman tiene
  fotos espejadas. Los números de esos tres datasets son optimistas para los modelos ajustados. Egipto es el
  más limpio.
- **HRP4K** (fotos 4K en perspectiva, China) es el único dominio que ni el modelo descargado ni a1 vieron.
  Es la única señal disponible de cómo rinde un modelo en un lugar nuevo, como Santa Cruz.

## Punto de partida: modelos públicos medidos en val

AP50 de baches en val. Los `.pt` se revisaron con picklescan antes de cargarlos, sin el token de Hugging Face
en el contenedor.

| Modelo | val 1280 (media de vehículo) | val 640 (media de vehículo) |
|---|---:|---:|
| YOLO12s RDD2022 (la app hasta hoy) | 0,203 | 0,173 |
| **YOLO26s destilado RDD+FRDC (TamAko783)** | **0,310** | **0,281** |
| YOLO26s base RDD (TamAko783) | 0,247 | 0,246 |
| YOLO26n destilado (TamAko783) | 0,268 | 0,223 |
| otros cuatro (dashcam, sadak, stulpe, HRP4K-v8m) | 0,07 a 0,24 | 0,09 a 0,20 |

El YOLO26s destilado (**B0**) se eligió como punto de partida y como respaldo: ya era mejor que el de la app.

## a1: ajuste de B0 con nuestros datos

- **Receta** (`entrenamiento/entrenar.py`):
  - 1 clase ("Pothole (D40)", la fila de bache de B0). SGD, lr0 0,005 coseno, 1280 px, batch 32.
  - mosaic, sin mosaic las últimas 15 épocas; tope de 150 épocas con paciencia 30.
  - **Aumentos que imitan video de WhatsApp:** JPEG 20 a 60 (p 0,3), reducir 0,4 a 0,8 (p 0,25) y desenfoque
    de movimiento (p 0,15).
- **Ejecución:** 91 épocas en 1,18 h; la paciencia la cortó.
  - Con la caché en RAM el proceso se quedó sin memoria: Python 3.14 copia la caché en cada worker. Por eso se
    usa caché en disco.
  - El `time` de ultralytics no es un tope: recalcula las épocas para llenar el tiempo. Ahora `--horas` es un
    freno de seguridad.

AP50 de baches en el test congelado:

| Modelo | px | Bisman | Brasil | Bucko | Egipto | HRP4K | todos |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO12s (app hasta hoy) | 1280 | 0,087 | 0,478 | 0,233 | 0,098 | 0,089 | 0,145 |
| B0 descargado | 1280 | 0,113 | 0,590 | 0,392 | 0,264 | 0,095 | 0,200 |
| a1 ajustado | 1280 | 0,828 | 0,703 | 0,658 | 0,541 | **0,050** | 0,517 |
| YOLO12s | 640 | 0,153 | 0,472 | 0,150 | 0,151 | 0,032 | 0,128 |
| B0 | 640 | 0,187 | 0,588 | 0,306 | 0,356 | 0,048 | 0,196 |
| a1 | 640 | 0,834 | 0,662 | 0,629 | 0,442 | **0,017** | 0,496 |

**Ajustar especializa.** a1 sube mucho donde entrenó (Egipto 0,26 → 0,54) pero **baja en HRP4K**, el dominio
que no vio. No se puede asumir que el ajuste ayude en Santa Cruz.

## WiSE-FT: promedio de pesos entre B0 y a1

θ(α) = α·a1 + (1−α)·B0 ([Wortsman et al., 2022](https://arxiv.org/abs/2109.01903)); script en
`entrenamiento/promediar_pesos.py`. No hay entrenamiento: se probó α = 0,25, 0,5 y 0,75.

**Regla fijada antes de ver resultados** (`entrenamiento/elegir_modelo.py`):
- El test de HRP4K se parte en dos mitades fijas, por hash del nombre: 467 fotos en A y 432 en B.
- La mitad A sirve para elegir; la B solo para informar.
- A 640 px (el tamaño de la app), gana el de mayor AP50 en val entre los que no empeoran fuera de dominio:
  HRP4K-A ≥ la de B0 − 0,005.

| Modelo | val | Bisman | Brasil | Bucko | Egipto | test todos | HRP4K-A | HRP4K-B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO12s | 0,142 | 0,153 | 0,472 | 0,150 | 0,151 | 0,128 | 0,023 | 0,044 |
| B0 | 0,191 | 0,187 | 0,588 | 0,306 | 0,356 | 0,196 | 0,034 | 0,065 |
| **WiSE α = 0,25** | **0,299** | 0,366 | 0,613 | 0,409 | 0,471 | 0,282 | 0,029 | 0,059 |
| WiSE α = 0,50 | 0,478 | 0,623 | 0,653 | 0,509 | 0,534 | 0,393 | 0,025 | 0,040 |
| WiSE α = 0,75 | 0,680 | 0,791 | 0,683 | 0,598 | 0,484 | 0,472 | 0,015 | 0,019 |
| a1 | 0,732 | 0,834 | 0,662 | 0,629 | 0,442 | 0,496 | 0,011 | 0,019 |

(640 px. "val" es la media de todos los datasets; Bisman a Egipto son test.)

**Resultado: WiSE α = 0,25.** Los únicos que pasaban el filtro eran B0 y α = 0,25, y α = 0,25 tiene mejor val.
- **Frente a B0:** casi duplica a B0 en val y sube en test (0,196 → 0,282; Egipto 0,36 → 0,47).
- **Fuera de dominio:** en HRP4K-B queda un poco por debajo de B0 (0,059 contra 0,065) y por encima del YOLO12s.
- **A 1280 px:** α = 0,25 y 0,5 superan a B0 en HRP4K-A (0,084 y 0,087 contra 0,073).
- **Cuidado:** a 640 px HRP4K está casi en el piso para todos (baches lejanos y chicos), así que esa señal es
  débil.

**Versión de la app (`modelos/wise4_a25`):** el mismo promedio, pero con las 4 clases de B0. Las filas de grietas
quedan las de B0 y la de bache se promedia.

| A 640 px | val baches | test baches | test grietas Egipto / Brasil |
|---|---:|---:|---:|
| B0 | 0,191 | 0,196 | 0,114 / 0,070 |
| WiSE 4 clases (α = 0,25) | 0,293 | 0,276 | 0,199 / 0,063 |

Umbrales de la app, elegidos en val solo con el detector:
- **Bache 0,10:** F1 0,338, contra 0,314 de B0 con su umbral de 0,15.
- **Grieta 0,25:** el F1 es casi plano entre 0,10 y 0,25, así que se toma el más alto por precisión.

## a2: B0 ajustado con nuestros datos + HRP4K

Pendiente: se completa al terminar la corrida. Suma al train el train y el valid oficiales de HRP4K (3.186
fotos, copia de Hugging Face; `entrenamiento/importar_hrp4k_hf.py`). Una foto del test de HRP4K era casi copia y
dejó de evaluarse para todos. Para a2, HRP4K ya no es un dominio nuevo: no compite en la regla de arriba.

## Videos de Santa Cruz (sin etiquetas)

`comparar_videos.py`: los 5 videos de WhatsApp, todos los cuadros a 864×512 px, cada modelo con los umbrales
de la app. Tabla completa en `informes/videos_scz_20261122/RESULTADOS.md`.

% de cuadros con bache marcado (eventos):

| Modelo | scz_01 | scz_02 | scz_03 | scz_04 | scz_05 |
|---|---:|---:|---:|---:|---:|
| YOLO12s (app hasta hoy) | 7 % (22) | 5 % (5) | 1 % (7) | 1 % (1) | 1 % (1) |
| B0 descargado | 4 % (14) | 8 % (7) | 5 % (28) | 1 % (1) | 1 % (1) |
| **WiSE α = 0,25 (app)** | 4 % (12) | 6 % (8) | 5 % (42) | 1 % (1) | 1 % (0) |
| a1 ajustado | 21 % (31) | 15 % (23) | 38 % (48) | 3 % (2) | 8 % (6) |

Revisado a ojo en scz_03:
- Las marcas más seguras de los modelos nuevos son daños reales: a1 marca un área de asfalto desprendido
  (0,80) y WiSE un bache claro en el carril derecho (0,56).
- a1 marca 5 a 8 veces más porque además toma manchas y desgaste chico con confianza baja (0,2 a 0,3).
- Las grietas longitudinales grandes las marcan los tres modelos de grietas.

## Límites

- No hay fotos de Santa Cruz etiquetadas: lo de los videos son marcas, no aciertos.
- Para elegir el modelo se usó la mitad A del test de HRP4K. Es la única señal fuera de dominio y por eso
  quedó separada la mitad B para informar. El resto del test se miró una sola vez.
- En val hay fuga (Bucko, Brasil y Bisman), así que val favorece a los modelos más ajustados. Por eso la regla
  exige además no empeorar fuera de dominio.
- **La regla no fue la del plan.** El plan decía "deciden HRP4K y Egipto". La regla que se aplicó filtra por
  HRP4K y después ordena por val de todos los datasets. Con la del plan el ganador es el mismo: entre los que
  pasan el filtro (B0 y α = 0,25), Egipto en val a 640 px da 0,320 contra 0,283.
- **243 fotos dudosas quedaron en el train.** Las marcó la revisión del 1-nov y no está claro que sacarlas
  mejore.
- **Brasil sin enmascarar.** Brasil etiqueta solo el carril propio: un bache en el carril de al lado cuenta
  como falso positivo, y uno sin marcar allí le enseña "fondo" al modelo. Enmascararlo (`_calzada_brazil`)
  lleva más trabajo y no se hizo.

## Reproducir

```bash
python entrenamiento/limpieza/aplicar_correcciones.py
python entrenamiento/armar_dataset.py --salida /datos/ds_baches
python entrenamiento/entrenar.py --datos /datos/ds_baches/data.yaml --modelo modelos/rdd_yolo26s/weights/best.pt \
    --nombre a1_y26s --cache-disco --aumentos-scz --epocas 150 --horas 3
python entrenamiento/promediar_pesos.py modelos/rdd_yolo26s/weights/best.pt runs/a1_y26s/weights/best.pt 0.25 \
    sopas/wise_a25/weights/best.pt
python entrenamiento/promediar_pesos.py modelos/rdd_yolo26s/weights/best.pt runs/a1_y26s/weights/best.pt 0.25 \
    sopas/wise4_a25/weights/wise4_a25.pt --clases partida
python evaluar.py --split val --imgsz 640 --barrer --det-baches sopas/wise_a25/weights/best.pt   # y test, 1280
python entrenamiento/elegir_modelo.py
```
