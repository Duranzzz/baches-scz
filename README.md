# Baches y grietas en Santa Cruz de la Sierra

Detección de baches y grietas en fotos y videos de la calle, tomados a pie o desde un vehículo, para
ubicarlos en el mapa de Santa Cruz de la Sierra. Proyecto del 2.º parcial de Inteligencia Artificial (UCB).

- **App web:** detecta en el teléfono y marca cada daño en el mapa de la ciudad:
  https://duranzzz.github.io/baches-scz-app/ (código en
  [Duranzzz/baches-scz-app](https://github.com/Duranzzz/baches-scz-app)).
- **Este repositorio:** los datos y cómo se partieron, la evaluación por dataset y país, las pruebas
  sobre videos de Santa Cruz y el entrenamiento en GPU.

## Datos

12.204 fotos de seis datasets públicos, con las etiquetas pasadas a un esquema común. Las fotos y las
etiquetas no se suben aquí: cada dataset se baja de su fuente. La copia ya convertida del equipo está en
el repositorio privado de Hugging Face `Duranishhh/baches-scz` (`paquete/`).

| Prefijo | País | Cámara | Fotos | Qué marca | Fuente | Licencia |
|---|---|---|---:|---|---|---|
| `bisman_` | sin dato | celular, de cerca | 4.719 | baches | [repositorio de Bisman-Singh-Dev en GitHub](https://github.com/Bisman-Singh-Dev/YOLO-Pothole-Detection-Model), versión bd8238b | no documentada |
| `brazil_` | Brasil | desde vehículo, en ruta | 2.235 | baches y grietas sin tipo | Passos et al., *Cracks and Potholes in Road Images*, Mendeley Data v4, [doi:10.17632/t576ydh9v8.4](https://doi.org/10.17632/t576ydh9v8.4) | CC BY 4.0 |
| `bucko_` | Eslovaquia | dashcam 1920×1080 | 2.099 | baches y tapas | Bučko et al., Sensors 22:8878 (2022), [doi:10.3390/s22228878](https://doi.org/10.3390/s22228878), figshare 21214400 | CC BY 4.0 |
| `egypt_` | Egipto | celular frontal (cuadros de video) | 1.951 | baches y grietas sin tipo | Roboflow "rdd-6706t" v8, vía [Hanno100/RoadDamageDetection-Egypt](https://huggingface.co/datasets/Hanno100/RoadDamageDetection-Egypt) | CC BY 4.0 |
| `hrp4k_` | sin dato | 4K, en perspectiva | 900 | baches | HRP4K (test), Chen et al., Scientific Data 13:961 (2026), [doi:10.1038/s41597-026-07317-w](https://doi.org/10.1038/s41597-026-07317-w) | CC BY 4.0 |
| `rdd2022_` | Japón, India, Rep. Checa, Noruega, EE. UU., China | celular, dashcam, moto | 300 | grietas de 3 tipos | RDD2022 (muestra del test), Arya et al., [arXiv:2209.08538](https://arxiv.org/abs/2209.08538) | CC BY-SA 4.0 |

Esquema de etiquetas (un `.txt` por foto, una línea por caja: clase, centro x, centro y, ancho y alto,
relativos a la imagen): 0 grieta longitudinal (D00), 1 transversal
(D10), 2 cocodrilo (D20), 3 bache (D40), 4 grieta sin tipo, 5 tapa o rejilla. La columna "Qué marca"
dice qué clases están completas en cada dataset, y solo esas se evalúan.

Cuidado con:

- **Egipto:** Roboflow aumentó todos sus splits (mosaicos, rotaciones, volteos) y los splits comparten
  videos. Aquí se parte por video y solo se evalúan 216 cuadros reales, revisados uno por uno.
- **RDD2022:** en esta redistribución, la clase "pothole" también marca sendas, tapas y parches. De
  RDD2022 solo se evalúan las grietas.
- **Bucko:** los autores no documentan la clase de tapas; se verificó a ojo.
- **Bisman:** fotos de primer plano, casi todas con bache.

`splits.csv` fija la partición de cada foto (`foto, dataset, split, grupo, condicion, evaluable, motivo`).
Se parte por video o tramo de ruta, para que train y test no compartan escena, y sin casi copias (dHash)
entre train y val/test. HRP4K y RDD2022 van enteros a test. `preparar_splits.py` la reconstruye.

## Estructura

```
preview.py            detecta baches y grietas en imágenes; dibuja daño, severidad y confianza
visor.py              ventana para probar la detección sobre los videos de videos/
comparar_videos.py    corre uno o varios detectores sobre videos sin etiquetas: un video por detector,
                      uno de 6 paneles cuadro a cuadro, CSV y tabla
evaluar.py            mide contra las etiquetas, por dataset y clase; busca umbrales en val
preparar_splits.py    arma splits.csv
splits.csv            partición congelada de cada foto
exportar_web.py       lleva la detección a la app web (ONNX) y verifica que dé lo mismo
informes/             resultados (ver abajo)
entrenamiento/        reentrenamiento en una GPU AMD: GUIA.md (pasos), PLAN.md (decisiones) y sus scripts
tests/                pruebas (pytest)
```

Quedan en la PC y no se suben: `fotos/` y `etiquetas/` (los datasets), `muestras/`, `videos/` y `salidas/`.

## Instalar

Probado con Python 3.12 en Windows, en CPU:

```bash
uv venv --python 3.12 .venv          # o: py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt
```

Con GPU, instalar torch con CUDA o ROCm en vez de la versión CPU; los scripts la usan solos.

## Uso

```bash
.venv\Scripts\python preview.py --source foto.jpg                       # una imagen o una carpeta
.venv\Scripts\python visor.py                                           # videos de videos/
.venv\Scripts\python comparar_videos.py videos\scz_02_frontal_avenida.mp4
.venv\Scripts\python evaluar.py --split val --barrer                     # umbrales: se eligen en val
.venv\Scripts\python evaluar.py --split test                             # el número que se informa
.venv\Scripts\python -m pytest tests
```

`preview.py` guarda en `salidas/`, por imagen:

- `<nombre>.jpg`: cajas y manchas con `clase | severidad | confianza` (verde baja, amarillo media, rojo alta).
- `<nombre>.json`: `{tipo, severidad, confianza, caja:[x1,y1,x2,y2], area_rel, fuente}`. `fuente` dice
  qué parte lo vio: `det` (la detección), `seg` (la segmentación de baches) o `det+seg` (las dos).

Para video o dashcam de 1080p o más, usar `--imgsz 1280`: a 640, el bache lejano queda en pocos píxeles.
Con imágenes chicas (≤ 640 px), dejar 640.

## Severidad (provisional)

Los umbrales están al inicio de `preview.py`. No hay etiquetas de severidad: son reglas a calibrar con
la cámara que se use al final.

- **Baches:** área > 5 % de la imagen, alta; > 1,5 %, media; si no, baja. El área es la de la mancha
  si la caja se superpone con una (IoU > 0,3); si no, la de la caja.
- **Grietas:** `puntaje = factor × (1 + área_caja × 10) × confianza`, con factor 1 para longitudinal y
  transversal y 2 para cocodrilo. ≥ 3, alta; ≥ 1,5, media; si no, baja.

## Evaluación y resultados

Un acierto es una caja con IoU ≥ 0,5 contra una etiqueta de la misma clase. AP50 con 101 puntos de
recall. Además, "por foto": de las fotos con el daño, cuántas se marcaron, y de las marcadas, cuántas
lo tenían. Cada dataset se mide solo en las clases que tiene completas. Los umbrales se eligen en val
y el test se mide una sola vez.

- `informes/linea_base.md`: AP50 y P/R por dataset (país) en baches y grietas, a 640 y a 1280 px.
  Los JSON completos están al lado.
- `informes/ejemplos/`: una muestra de cada dataset y país (Bisman, Brasil, Bucko/Eslovaquia, Egipto,
  HRP4K, y RDD2022 de China, Japón y EE. UU.) con lo detectado dibujado, a 640 y a 1280 px, más su JSON.
- `informes/videos_scz/`: los cinco videos de Santa Cruz. Cuadros con marca y eventos de cada detector,
  la marca más segura de cada uno y el CSV de cada detección. No tienen etiquetas: son marcas, no aciertos.

## Videos de Santa Cruz

En `videos/` (no se suben: muestran personas y placas). Todos de 848 x 478, comprimidos por WhatsApp.

| Video | Cámara y calle |
|---|---|
| `scz_01_lateral_auto_hormigon.mp4` | baja, al costado de un auto; hormigón con grietas y parches |
| `scz_02_frontal_avenida.mp4` | de frente, desde dentro del auto; avenida de la ciudad |
| `scz_03_frontal_doble_via.mp4` | de frente, desde dentro del auto; carretera de doble vía |
| `scz_04_lateral_micro_losas.mp4` | al costado de un micro; losas de hormigón |
| `scz_05_lateral_micro_hormigon.mp4` | al costado de un micro; hormigón |

## Límites

- No hay fotos ni video de Santa Cruz etiquetados: todos los números son de datasets de otros países.
- Desde un vehículo, los baches se detectan poco; con la cámara al costado, la carrocería entra en el cuadro.
- La severidad es provisional.

## Licencias

El código usa [ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0). Las fotos de cada
dataset tienen su propia licencia (tabla de Datos).
