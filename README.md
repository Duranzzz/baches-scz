# Baches y grietas en Santa Cruz de la Sierra

Detección de baches y grietas en fotos y videos de la calle, tomados a pie o desde un vehículo, para
ubicarlos en el mapa de Santa Cruz de la Sierra. Proyecto del 2.º parcial de Inteligencia Artificial (UCB), de
Leonardo Duran Cuenca y Dana Gabriela Quisberth Sursi.

- **App web:** detecta en el teléfono y marca cada daño en el mapa de la ciudad:
  https://duranzzz.github.io/baches-scz-app/ (código en
  [Duranzzz/baches-scz-app](https://github.com/Duranzzz/baches-scz-app)).
- **Modelo del proyecto:** Baches SCZ v1, en `modelos/` (ficha en [`MODEL_CARD.md`](MODEL_CARD.md)).
- **Datos:** en Hugging Face, [Duranishhh/baches-scz](https://huggingface.co/datasets/Duranishhh/baches-scz), con una
  vista previa de 300 fotos del test y sus cajas.
- **Artículo:** [`docs/articulo_2do_parcial.pdf`](docs/articulo_2do_parcial.pdf), en el formato de la Revista 3i
  (también en `.docx`).
- **Este repositorio:**
  - los datos y cómo se partieron;
  - la limpieza y el entrenamiento;
  - la evaluación por conjunto de datos y país;
  - las pruebas sobre videos de Santa Cruz.

## Inicio rápido

```bash
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python herramientas\preview.py --source foto.jpg          # deja la imagen marcada en salidas/
```

O en Python, con el modelo en ONNX:

```python
from ultralytics import YOLO

modelo = YOLO("modelos/baches_scz_v1.onnx")
r = modelo.predict("foto.jpg", imgsz=640, conf=0.10)[0]
print([(modelo.names[int(c)], round(p, 2)) for c, p in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist())])
```

## Resultados principales

AP50 en el test congelado (`splits.csv`), a 640 px, el tamaño de la app:

| | Bisman | Brasil | Bucko | Egipto | HRP4K | Total |
|---|---:|---:|---:|---:|---:|---:|
| Baches | 0,346 | 0,607 | 0,408 | 0,472 | 0,040 | 0,276 |
| Grietas | | 0,063 | | 0,199 | | |

- **Contra el detector anterior:** el que usaba la app antes obtiene 0,128 en baches, así que el actual duplica
  el total y triplica Egipto, el conjunto más limpio.
- **En los videos de Santa Cruz**, que no tienen etiquetas, se revisaron a ciegas las marcas de cinco modelos:
  - el 64 % de los eventos de este modelo son baches, contra el 32 % de a1, el mismo sin promediar;
  - los dos encuentran 21 baches distintos.
  - Detalle en `informes/videos_scz_20261122/revision/RESULTADOS.md`.
- **En 200 cuadros de Santa Cruz etiquetados con cajas** (`informes/scz_cuadros/`), este modelo obtiene un AP50
  de 0,067 (0,162 con IoU 0,3).
  - Sus marcas suelen ser baches, pero encuentra menos que a1 (0,158) y a2 (0,299), ajustados por completo.
- **Ajuste con cuadros de la ciudad (v2):** con 120 de esos cuadros, el AP50 en los otros 80 sube de 0,087 a hasta
  0,275, pero baja en HRP4K, el conjunto que el entrenamiento no ve.
  - Por la regla fijada antes de entrenar, este modelo sigue en la app.
  - Detalle en `informes/ajuste_scz.md`.
- **Más detalle:**
  - `informes/entrenamiento_amd.md`: cómo se eligió el modelo y sus límites.
  - `MODEL_CARD.md`: el modelo en sí.

## Datos

Todo está en Hugging Face, en [Duranishhh/baches-scz](https://huggingface.co/datasets/Duranishhh/baches-scz). Su
pestaña de datos muestra 300 fotos del test con sus cajas.

| Carpeta | Contenido |
|---|---|
| `muestra/` | la vista previa: 50 fotos del test por fuente, con sus cajas (la arma `entrenamiento/hf_muestra.py`) |
| `paquete/` | 12.204 fotos de seis conjuntos públicos (reducidas a 1.280 px) y sus etiquetas en un esquema común |
| `correcciones/` | las 14 etiquetas corregidas y la lista de las 619 fotos retiradas del train |
| `videos_scz/` | los cinco videos de Santa Cruz, con caras y placas difuminadas (ver abajo) |
| `santa_cruz/` | 200 cuadros de esos videos con cajas de bache: 120 de ajuste y 80 de prueba |
| `resultados/` | lo que dejaron los entrenamientos del 22 y del 23-nov: curvas, pesos y evaluaciones |

El train y el valid de HRP4K (3.186 fotos más) se bajan de su copia pública
[dronefreak/HRP4K](https://huggingface.co/datasets/dronefreak/HRP4K) con `entrenamiento/importar_hrp4k_hf.py`.
Los datos no se suben a este repositorio: `fotos/`, `etiquetas/`, `videos/` y `salidas/` quedan en la PC.

| Prefijo | País | Cámara | Fotos | Qué marca | Fuente | Licencia |
|---|---|---|---:|---|---|---|
| `bisman_` | sin dato | celular, de cerca | 4.719 | baches | [repositorio de Bisman-Singh-Dev en GitHub](https://github.com/Bisman-Singh-Dev/YOLO-Pothole-Detection-Model), versión bd8238b | no documentada |
| `brazil_` | Brasil | desde vehículo, en ruta | 2.235 | baches y grietas sin tipo | Passos, *Cracks and Potholes in Road Images*, Mendeley Data v4, [doi:10.17632/t576ydh9v8.4](https://doi.org/10.17632/t576ydh9v8.4) | CC BY 4.0 |
| `bucko_` | Eslovaquia | dashcam 1920×1080 | 2.099 | baches y tapas | Bučko et al., Sensors 22:8878 (2022), [doi:10.3390/s22228878](https://doi.org/10.3390/s22228878), figshare 21214400 | CC BY 4.0 |
| `egypt_` | Egipto | celular frontal (cuadros de video) | 1.951 | baches y grietas sin tipo | Roboflow "rdd-6706t" v8, vía [Hanno100/RoadDamageDetection-Egypt](https://huggingface.co/datasets/Hanno100/RoadDamageDetection-Egypt) | CC BY 4.0 |
| `hrp4k_` | sin dato | 4K, en perspectiva | 900 | baches | HRP4K (test), Chen et al., Scientific Data 13:961 (2026), [doi:10.1038/s41597-026-07317-w](https://doi.org/10.1038/s41597-026-07317-w) | CC BY 4.0 |
| `rdd2022_` | Japón, India, Rep. Checa, Noruega, EE. UU., China | celular, dashcam, moto | 300 | grietas de 3 tipos | RDD2022 (muestra del test), Arya et al., [arXiv:2209.08538](https://arxiv.org/abs/2209.08538) | CC BY-SA 4.0 |

**Esquema de etiquetas:**
- Un `.txt` por foto y una línea por caja: clase, centro x, centro y, ancho y alto, relativos a la imagen.
- Clases: 0 grieta longitudinal (D00), 1 transversal (D10), 2 cocodrilo (D20), 3 bache (D40), 4 grieta sin
  tipo, 5 tapa o rejilla.
- La columna "Qué marca" dice qué clases están completas en cada conjunto, y solo esas se evalúan.

Cuidado con:

- **Egipto:** Roboflow aumentó todos sus splits (mosaicos, rotaciones, volteos) y los splits comparten
  videos. Aquí se parte por video y solo se evalúan 216 cuadros reales, revisados uno por uno.
- **RDD2022:** en esta redistribución, la clase "pothole" también marca sendas, tapas y parches. De
  RDD2022 solo se evalúan las grietas.
- **Bucko:** los autores no documentan la clase de tapas; se verificó a ojo.
- **Bisman:** fotos de primer plano, casi todas con bache.

`splits.csv` fija la partición de cada foto (`foto, dataset, split, grupo, condicion, evaluable, motivo`):
- **Cómo se parte:** por video o tramo de ruta, para que train y test no compartan escena, y sin casi copias
  (dHash) entre train y val/test.
- **Test:** RDD2022 y el test oficial de HRP4K van enteros a test.
- **Reconstruirla:** `herramientas/preparar_splits.py`.

Además:
- **Limpieza (1-nov):** del train salieron 619 fotos (596 montajes de Egipto, 11 duplicados, 11 con
  superposiciones y 1 sin pavimento), que figuran como "excluida". Se corrigieron 14 etiquetas: correr
  `entrenamiento/limpieza/aplicar_correcciones.py` después de bajar los datos.
- **HRP4K:** su train y su valid suman 3.186 fotos al train. Una foto de su test era casi copia de una de ellas y
  dejó de evaluarse.

## Estructura

```
herramientas/         detección y evaluación
  preview.py            detecta baches y grietas en imágenes; dibuja daño, severidad y confianza
  visor.py              ventana para probar la detección sobre los videos de videos/
  comparar_videos.py    corre detectores sobre videos sin etiquetas: un video por detector, una grilla, CSV y tabla
  revisar_eventos.py    revisión a ciegas de las marcas en los videos: hojas de recortes y precisión por modelo
  difuminar_videos.py   difumina caras y placas de los videos y les quita el audio antes de publicarlos
  evaluar.py            mide contra las etiquetas, por conjunto y clase; busca umbrales en val
  preparar_splits.py    arma splits.csv
  exportar_web.py       lleva el modelo a la app web (ONNX) y verifica que dé lo mismo
entrenamiento/        limpieza de datos y entrenamiento en GPU: GUIA.md (pasos), PLAN.md (decisiones) y scripts
modelos/              el modelo del proyecto: baches_scz_v1.pt y baches_scz_v1.onnx
informes/             evaluaciones, curvas de entrenamiento y pruebas en videos (ver abajo)
tests/                pruebas (pytest)
docs/                 el artículo (.docx y .pdf), su plantilla y los scripts que lo arman con sus figuras
splits.csv            partición congelada de cada foto
MODEL_CARD.md         ficha del modelo
CHANGELOG.md          cambios por versión
CITATION.cff          cómo citar el proyecto
LICENSE               AGPL-3.0
```

## Uso

```bash
.venv\Scripts\python herramientas\preview.py --source foto.jpg            # una imagen o una carpeta
.venv\Scripts\python herramientas\visor.py                                # videos de videos/
.venv\Scripts\python herramientas\comparar_videos.py videos\scz_02_frontal_avenida.mp4
.venv\Scripts\python herramientas\evaluar.py --split val --barrer          # umbrales: se eligen en val
.venv\Scripts\python herramientas\evaluar.py --split test                  # el número que se informa
.venv\Scripts\python -m pytest tests
```

`preview.py` guarda en `salidas/`, por imagen:

- `<nombre>.jpg`: cajas y manchas con `clase | severidad | confianza` (verde baja, amarillo media, rojo alta).
- `<nombre>.json`: `{tipo, severidad, confianza, caja:[x1,y1,x2,y2], area_rel, fuente}`. `fuente` dice
  qué parte lo vio: `det` (la detección), `seg` (la segmentación de baches, que da el área) o `det+seg`.

**Tamaño de entrada:**
- Video o dashcam de 1080p o más: usar `--imgsz 1280`, porque a 640 el bache lejano queda en pocos píxeles.
- Imágenes chicas (≤ 640 px): dejar 640.

## Severidad (provisional)

Los umbrales están al inicio de `herramientas/preview.py`. No hay etiquetas de severidad: son reglas a
calibrar con la cámara que se use al final.

- **Baches:** área > 5 % de la imagen, alta; > 1,5 %, media; si no, baja. El área es la de la mancha
  si la caja se superpone con una (IoU > 0,3); si no, la de la caja.
- **Grietas:** `puntaje = factor × (1 + área_caja × 10) × confianza`, con factor 1 para longitudinal y
  transversal y 2 para cocodrilo. ≥ 3, alta; ≥ 1,5, media; si no, baja.

## Evaluación e informes

Un acierto es una caja con IoU ≥ 0,5 contra una etiqueta de la misma clase. AP50 con 101 puntos de
recall. Además, "por foto": de las fotos con el daño, cuántas se marcaron, y de las marcadas, cuántas
lo tenían. Cada conjunto se mide solo en las clases que tiene completas. Los umbrales se eligen en val
y el test se mide una sola vez.

- `informes/entrenamiento_amd.md`: el entrenamiento en GPU del 22-nov. Incluye datos, receta, resultados por
  conjunto, la regla para elegir el modelo (fijada antes de ver resultados) y sus límites.
- `informes/corridas/`: curvas y argumentos de cada corrida de entrenamiento.
- `informes/eval_*.json`: cada evaluación completa. `informes/linea_base.md` es la primera medición (28-oct).
- `informes/ejemplos/`: una muestra por conjunto y país con lo detectado dibujado, a 640 y a 1280 px.
- `informes/videos_scz/` y `informes/videos_scz_20261122/`: los cinco videos de Santa Cruz.
  - Contiene los cuadros con marca y los eventos de cada detector, y el CSV de cada detección.
  - No tienen etiquetas: son marcas, no aciertos.
- `informes/scz_cuadros/`: los 200 cuadros etiquetados (`entrenamiento/scz_cuadros.py`): cajas, juicios y el AP de
  cada modelo.
- `informes/ajuste_scz.md`: el ajuste v2 con esos cuadros, con su regla (fijada antes de etiquetar) y el resultado.

## Videos de Santa Cruz

Los originales están en `videos/`. En Hugging Face (`videos_scz/`) se publican con caras y placas difuminadas y sin
audio; los procesa `herramientas/difuminar_videos.py`. No se suben a este repositorio. Todos son de 848 × 478, comprimidos por WhatsApp.

| Video | Cámara y calle |
|---|---|
| `scz_01_lateral_auto_hormigon.mp4` | baja, al costado de un auto; hormigón con grietas y parches |
| `scz_02_frontal_avenida.mp4` | de frente, desde dentro del auto; avenida de la ciudad |
| `scz_03_frontal_doble_via.mp4` | de frente, desde dentro del auto; carretera de doble vía |
| `scz_04_lateral_micro_losas.mp4` | al costado de un micro; losas de hormigón |
| `scz_05_lateral_micro_hormigon.mp4` | al costado de un micro; hormigón |

## Límites

- **Santa Cruz a medias:** hay 200 cuadros etiquetados, pero de solo cinco videos y por un solo revisor, que partió
  de las cajas propuestas por los modelos. La revisión a ciegas de los videos mide qué parte de las marcas son
  baches.
- **Elección en el límite:** con otra semilla de entrenamiento, el mismo promedio da 0,027 en HRP4K-A, por debajo
  del margen de la regla (0,029). El resto del patrón se repite (`informes/ajuste_scz.md`).
- **Videos laterales:** con la cámara al costado entra la carrocería en el cuadro, y las juntas del hormigón
  pueden salir como grietas.
- **Severidad:** es provisional.

## Cómo citar

Los datos para citar el proyecto están en [`CITATION.cff`](CITATION.cff); GitHub los muestra en "Cite this
repository".

## Licencia

Código y modelo, AGPL-3.0 (`LICENSE`); el origen del modelo está en `MODEL_CARD.md`. El texto y las figuras del
artículo, CC BY 4.0, como dice el propio artículo. Las fotos de cada conjunto de datos tienen su propia licencia
(tabla de Datos).
