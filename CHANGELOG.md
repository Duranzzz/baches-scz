# Cambios

## [1.3.0] - 2026-11-23

- **Cuadros de Santa Cruz etiquetados** (`entrenamiento/scz_cuadros.py`, `informes/scz_cuadros/`): 200 cuadros de
  los cinco videos con cajas de bache, 120 de ajuste y 80 de prueba.
  - AP50 a 640 px en los 200: a2, 0,299; a1, 0,158, y Baches SCZ v1, 0,067.
  - También están en Hugging Face, en la config `santa_cruz`.
- **Ajuste v2 con esos cuadros** (`informes/ajuste_scz.md`), con una regla fijada antes de etiquetar y de entrenar:
  - el AP50 en los 80 cuadros de prueba sube de 0,087 a hasta 0,275, pero baja en HRP4K-A;
  - ninguna variante cumple la regla, así que Baches SCZ v1 sigue en la app;
  - la regla la aplica `entrenamiento/decidir_ajuste.py`; pesos y logs, en Hugging Face (`resultados/amd_20261123/`).
- **Corrección:** `armar_dataset.py` nombra la clase "Pothole (D40)" para que ultralytics reuse la fila de bache del
  modelo de partida; con "D40" la dejaba al azar.
- **Entrenamiento:** `entrenar.py --semilla`.
- **Videos:** retoques a mano del difuminado (un reflejo en scz_01 y dos placas lejanas).
- **App:** a2 se suma a los modelos para comparar.
- **Artículo:** tabla 6 con los cuadros etiquetados y el resultado de v2; resúmenes al día.

## [1.2.0] - 2026-11-23

- **Revisión a ciegas de los videos de Santa Cruz** (`herramientas/revisar_eventos.py`): se juzgaron los 355
  eventos de cinco modelos, con resultados en `informes/videos_scz_20261122/revision/`.
  - El 64 % de los eventos de Baches SCZ v1 son baches, contra el 32 % de a1, que no encuentra más baches.
  - El artículo suma la tabla 5 y lo dice en el resumen.
- **Privacidad:** `herramientas/difuminar_videos.py` difumina caras y placas y quita el audio de los videos que se
  publican en Hugging Face.
- **Artículo:** carrera y correo de Dana Gabriela Quisberth Sursi.
- **App:** a1 (solo baches) se suma a los modelos para comparar.

## [1.1.0] - 2026-11-22

- **Artículo** en `docs/`, en el formato de la Revista 3i (`.docx` y `.pdf`), con sus figuras y los scripts que lo
  arman.
- **Cita:** `CITATION.cff`, con los dos autores, Leonardo Duran Cuenca y Dana Gabriela Quisberth Sursi, y sus ORCID.
- **Datos:** el repositorio de Hugging Face pasa a ser público. Tiene una vista previa de 300 fotos del test
  (`entrenamiento/hf_muestra.py`).

## [1.0.0] - 2026-11-22

- **Modelo:** el del proyecto pasa a ser **Baches SCZ v1** (baches y grietas).
  - Está en `modelos/` en dos formatos, `.pt` y `.onnx`, con su ficha (`MODEL_CARD.md`) y licencia AGPL-3.0
    (`LICENSE`).
  - Es el único modelo que se sube al repositorio.
- **Entrenamiento en una GPU AMD MI300X:**
  - **Limpieza de datos:** salen del train 619 fotos y se corrigen 14 etiquetas.
  - **Corridas y selección:** dos corridas, promedio de pesos y una regla de selección fijada antes de ver
    resultados (`informes/entrenamiento_amd.md`).
- **Videos:** comparación en los cinco videos de Santa Cruz (`informes/videos_scz_20261122/`).
- **App web:** publicada con este modelo ([Duranzzz/baches-scz-app](https://github.com/Duranzzz/baches-scz-app)).
- **Orden del repositorio:**
  - Las herramientas pasan a `herramientas/`.
  - Las curvas de las corridas pasan a `informes/corridas/`.
  - Salen del repositorio los cuadros de los videos de Santa Cruz, que muestran personas y placas.

## 2026-11-21

- **App web instalable:** detecta en el teléfono y marca cada daño en el mapa de Santa Cruz de la Sierra.
- **Videos:** comparación de detectores en los videos de la ciudad.
- **App:** exportación a ONNX, verificada contra la biblioteca de entrenamiento.

## 2026-11-14

- **Primera versión del repositorio:**
  - Detección en imágenes y visor de videos.
  - Evaluación por conjunto de datos y país.
  - Partición congelada (`splits.csv`) y plan de entrenamiento.
