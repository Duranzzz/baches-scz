# Comparación del video de WhatsApp — 5 de noviembre de 2026

Se procesó el video completo con los tres checkpoints locales disponibles, sin descargar otros modelos ni ejecutar entrenamiento. Duración original: 99.168 s. 2976 fotogramas, 848 × 478 píxeles. Cada modelo analizó todos los fotogramas.

## Videos

- [Comparación de cuatro paneles](comparacion_4_paneles.mp4): original arriba izquierda, YOLO12s arriba derecha, Farzad abajo izquierda y piloto previo abajo derecha.
- [YOLO12s RDD2022](yolo12s_rdd2022.mp4): grietas, baches y su clase Repair.
- [Farzad YOLOv8n-seg](farzad_segmentacion.mp4): baches con contornos y cajas.
- [Piloto v1 previo](piloto_v1_previo.mp4): referencia del modelo anterior de dos épocas, ya existente; no es el nuevo piloto v2 ni un modelo final.

Todos los MP4 finales están en H.264 y conservan el audio original AAC y la duración. La cabecera y el pie se añadieron fuera de la imagen de la carretera; el video original no se modificó. No se saltaron fotogramas, no se usó interpolación de detecciones ni se aplicó seguimiento o suavizado temporal.

## Configuración común

Tamaño de inferencia 1280, umbral de confianza 0,25, NMS IoU 0,70, FP32 en CPU. Estos umbrales permiten una comparación visual con parámetros comunes; las confianzas de modelos distintos no están calibradas entre sí. El tamaño original del video se conserva al dibujar.

Rojo: bache; naranja: grieta longitudinal; cian: transversal; magenta: cocodrilo; gris: Repair/parche. En Farzad el área coloreada es la predicción de su máscara; no es área física medida.

## Observaciones por fotograma

| Modelo | Fotogramas con alguna marca | Observaciones de bache | Observaciones de grieta | Observaciones de Repair |
|---|---:|---:|---:|---:|
| YOLO12s RDD2022 | 1716 | 308 | 2623 | 0 |
| Farzad YOLOv8n-seg | 186 | 214 | 0 | 0 |
| Piloto v1 previo - referencia | 676 | 776 | 0 | 0 |

Una lesión puede aparecer cientos de veces al continuar visible en el video. Estos números **no son el conteo de daños únicos ni una medida de precisión**. Un mayor número de marcas puede incluir más falsos positivos. No se dispone de etiquetas exhaustivas de este video para calcular precisión, recall o mAP.

La severidad queda **sin evaluar** en todos los resultados; no se deduce del tamaño del cuadro, de la máscara ni de la confianza. Hay CSV por modelo con coordenadas/confianza y JSONL por cada fotograma, incluidos los que no tienen detecciones. Los resúmenes JSON incluyen nombres originales de clases y huellas de cada checkpoint para reproducir la prueba.

Se verificaron los cuatro videos finales: 2976 fotogramas cada uno, audio presente, duración coincidente y decodificación completa con ffprobe. Se comprobaron la integridad del original y la de los pesos. El hash del video original es `96ec2d5117a7d33c7ac9437fda48a0cb29c67ee48d93bc07278f733e48019223`.
