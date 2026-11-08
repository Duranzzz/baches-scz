# Detector de baches y grietas (Santa Cruz de la Sierra)

## Estado (21-nov-2026)

- Corre: `preview.py`, `visor.py` y `evaluar.py` con dos modelos públicos (abajo), y la línea base
  medida en `informes/linea_base.md`, incluido el YOLO del 1.er parcial.
- **Todavía no hay modelo propio.** El reentrenamiento de `entrenamiento/` (corrida A1 en AMD) no se
  ejecutó; el armado del dataset y `entrenar.py` se probaron en chico, en CPU. Fuera de este repo
  hubo un piloto de 2 épocas en Colab, que no se usa.
- No hay fotos ni video de Santa Cruz etiquetados: todos los números son de datasets de otros países.
- `calzada.py` tiene pruebas, pero todavía no lo usa ningún script.

Preview que combina dos modelos ya entrenados, sin reentrenar:

- **Detección**: YOLO12s entrenado en RDD2022 ([rezzzq/yolo12s-road-damage-rdd2022](https://huggingface.co/rezzzq/yolo12s-road-damage-rdd2022), MIT).
  Clases D00 grieta longitudinal, D10 transversal, D20 cocodrilo, D40 bache. La clase `Repair` se ignora.
- **Segmentación**: YOLOv8n-seg de baches ([FarzadNekouee](https://github.com/FarzadNekouee/YOLOv8_Pothole_Segmentation_Road_Damage_Assessment), MIT).
  Da el contorno de cada bache, y con eso el área real.

## Estructura

```
preview.py            corre los modelos sobre imágenes y dibuja daños y severidad
visor.py              ventana para probar los modelos sobre videos de videos/
evaluar.py            mide contra las etiquetas, por dataset y clase; busca umbrales en val
preparar_splits.py    arma splits.csv (train/val/test por video o tramo de ruta, sin casi copias)
splits.csv            partición congelada de cada foto
calzada.py            proyección de píxeles al plano de la calle (para pasar áreas a m²)
tests/                pruebas de calzada.py (pytest)
informes/             resultados medidos (línea base, comparaciones)
entrenamiento/        reentrenamiento en GPU: GUIA.md (pasos), PLAN.md (decisiones) y sus scripts

fotos/        12.204 fotos de datasets públicos, ver fotos/FUENTES.txt   (no se sube a git)
etiquetas/    una etiqueta YOLO por foto, mismo nombre; _origen/ = fuente, licencia, videos
modelos/      pesos descargados                                         (no se sube a git)
muestras/     imágenes de prueba                                        (no se sube a git)
salidas/      resultados de preview.py y visor.py, caché de evaluar.py  (no se sube a git)
videos/       videos para probar en el visor                            (no se sube a git)
```

Clases de `etiquetas/`: 0 D00, 1 D10, 2 D20, 3 D40 (bache), 4 grieta sin tipo, 5 tapa o rejilla.
`fotos/` y `etiquetas/` no van a git: viajan juntas en el paquete privado de Hugging Face (`entrenamiento/hf.py`).

## Instalar

Probado con Python 3.12 en Windows, en CPU:

```bash
uv venv --python 3.12 .venv          # o: py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt
```

Si hay GPU, instalar torch con CUDA o ROCm en vez de la versión CPU; `preview.py` la usa sola.

## Correr

```bash
.venv\Scripts\python preview.py                        # muestras/
.venv\Scripts\python preview.py --source foto.jpg
.venv\Scripts\python preview.py --source fotos\         # carpeta entera
```

La primera vez descarga los dos modelos en `modelos/`. Por cada imagen guarda en `salidas/`:

- `<nombre>.jpg`: máscaras semitransparentes y cajas con `clase | severidad | confianza`
  (verde baja, amarillo media, rojo alta).
- `<nombre>.json`: lista de `{tipo, severidad, confianza, caja:[x1,y1,x2,y2], area_rel, fuente}`.

Con `--det pesos.pt` usa otro detector (por ejemplo uno reentrenado). `--det-baches` acepta cualquier
detector de una clase llamada D40, pothole o bache (por ejemplo, el YOLO del 1.er parcial). Para video o fotos de dashcam
(1080p o más) usar `--imgsz 1280`: en la línea base encuentra hasta 3 veces más baches que a 640.
Con fotos chicas (≤ 640 px) dejar 640 (ver `informes/linea_base.md`).

## Probar con video

Copiar los videos en la carpeta `videos/` y abrir el visor:

```bash
.venv\Scripts\python visor.py
.venv\Scripts\python visor.py --det-baches pesos.pt      # suma un detector de baches reentrenado
```

Muestra, sobre el video, las **cajas** de YOLO12s (un color por clase), las **manchas** de la
segmentación de baches, las dos juntas, las dos lado a lado, o la combinación con severidad de
`preview.py`. Se elige la confianza mínima, el tamaño de análisis (640 o 1280 px) y cada cuántos
cuadros procesar (en CPU, cada cuadro tarda ~0,5 a 1,5 s). Teclas: espacio reproduce o pausa,
flecha derecha avanza un cuadro, S guarda una captura en `salidas/capturas/`.
"Exportar video" guarda en `salidas/videos/` el video anotado (H.264 si hay ffmpeg) y un CSV con
cada detección: cuadro, tiempo, modelo, clase, confianza, caja, área y severidad.

En `videos/` hay un video de prueba armado con 150 cuadros seguidos de una dashcam de Bucko
(Eslovaquia, split de test).

## Medir

```bash
.venv\Scripts\python evaluar.py --split val --barrer      # umbrales: se eligen en val
.venv\Scripts\python evaluar.py --split test              # el número que se informa
.venv\Scripts\python evaluar.py --split test --imgsz 1280 --det pesos.pt
.venv\Scripts\python -m pytest tests
```

Cada dataset se mide solo en las clases que tiene completas (Bucko, HRP4K y Bisman solo marcan baches).
El test no se usa para elegir nada: los umbrales salen de val.

## Salida de preview.py

`fuente` dice qué modelo vio el daño: `det` (solo el detector), `seg` (solo la segmentación: un bache
que el detector no vio) o `det+seg` (caja D40 emparejada con una máscara). Cuando hay máscara, el área
sale de la máscara. `area_rel` es la fracción de la imagen.

## Severidad (provisional)

Los umbrales están al inicio de `preview.py`.

- **Baches**: área > 5 % alta, > 1,5 % media, si no baja. El área es la de la máscara si la caja D40
  se superpone con una (IoU > 0,3 entre la caja y la caja de la máscara); si no, la de la caja.
- **Grietas**: `puntaje = peso × (1 + área_caja × 10) × confianza`, con peso D00 = D10 = 1 y D20 = 2.
  ≥ 3 alta, ≥ 1,5 media, si no baja.

El área relativa depende de la cámara (altura, ángulo, zoom): los umbrales hay que calibrarlos con
fotos de la cámara que se use al final.

## Ojo

- Los `.pt` son pickles de terceros: cargarlos ejecuta código. Bajarlos solo de las fuentes de arriba.
- `ultralytics` es AGPL-3.0. Para un uso universitario no importa; si la alcaldía lo despliega como
  servicio, hay que revisarlo.
- YOLO12s se entrenó con RDD2022: las fotos `rdd2022_*` pueden ser parte de su entrenamiento y
  sus resultados ahí salen optimistas.
