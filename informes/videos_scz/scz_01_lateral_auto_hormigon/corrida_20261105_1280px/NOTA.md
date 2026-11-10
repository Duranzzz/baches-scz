# Corrida del 5-nov-2026 sobre este mismo video, a 1280 px

Rescatado de `salidas/videos/whatsapp_20261104_comparacion_20261105/` antes de borrarla (22-nov-2026).
El video es el mismo: SHA-256 `96ec2d5117a7d33c7ac9437fda48a0cb29c67ee48d93bc07278f733e48019223`, que es el de
`videos/scz_01_lateral_auto_hormigon.mp4` (antes "WhatsApp Video 2026-11-04 at 08.45.28.mp4").

Diferencias con la corrida de esta carpeta (21-nov):
- Analizaba a 1280 px (el video, de 848 x 478, se agrandaba); la nueva analiza a 864 px, su tamaño real.
- Tres modelos, sin el del 1er parcial.
- Tenía audio; los videos nuevos no.

| Modelo | Cuadros con marca, 1280 px (5-nov) | Cuadros con marca, 864 px (21-nov) |
|---|---:|---:|
| YOLO12s RDD2022 | 1716 | 1628 |
| Farzad YOLOv8n-seg | 186 | 87 |
| Piloto v1 | 676 | 471 |

Se guardaron solo los datos que la corrida nueva no tiene: la tabla original (`RESULTADOS_original.md`,
`comparacion_resumen.csv`), los resúmenes por modelo con la huella de cada peso, la verificación de que ONNX daba lo
mismo que PyTorch a 1280 (`concordancia_onnx.json`), los metadatos del video (`preparacion.json`) y la imagen
resumen. Los videos a 1280 px, los .zip duplicados, los JSONL y las capturas se borraron: los de 864 px de esta
carpeta los reemplazan.
