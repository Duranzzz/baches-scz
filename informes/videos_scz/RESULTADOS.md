# Modelos sobre videos de Santa Cruz (2026-11-22)

Todos los cuadros, a 864x512 px (el tamaño de estos videos, sin agrandarlos: en la línea base, agrandar imágenes chicas empeoró), confianza ≥ 0.25, NMS 0.7, en CPU.
Sin etiquetas: son marcas, no aciertos. Más marcas puede ser más falsos positivos. Un **evento** es un tramo de 3 cuadros o más con marca (huecos de hasta 0.5 s se unen): se parece más a "un daño visto" que contar cuadros.

## Baches: % de cuadros con marca (eventos)

| Modelo | scz_01_lateral_auto_hormigon | scz_02_frontal_avenida | scz_03_frontal_doble_via | scz_04_lateral_micro_losas | scz_05_lateral_micro_hormigon |
|---|---:|---:|---:|---:|---:|
| YOLO12s RDD2022 (baches y grietas) | 7 % (22) | 5 % (5) | 1 % (7) | 1 % (1) | 1 % (1) |
| Farzad YOLOv8n-seg (baches) | 3 % (14) | 4 % (15) | 3 % (17) | 1 % (0) | 1 % (1) |
| Piloto v1: Colab, 2 epocas (baches) | 16 % (49) | 14 % (15) | 10 % (49) | 3 % (3) | 3 % (4) |
| YOLO del 1er parcial, Bisman (baches) | 9 % (28) | 4 % (9) | 6 % (19) | 16 % (9) | 1 % (2) |

## Grietas (solo YOLO12s las detecta): % de cuadros con marca (eventos)

| Video | Cuadros con grieta | Eventos | Longitudinales | Transversales | Cocodrilo |
|---|---:|---:|---:|---:|---:|
| scz_01_lateral_auto_hormigon | 51 % | 18 | 1838 | 173 | 173 |
| scz_02_frontal_avenida | 10 % | 11 | 143 | 1 | 82 |
| scz_03_frontal_doble_via | 24 % | 34 | 1020 | 5 | 143 |
| scz_04_lateral_micro_losas | 5 % | 1 | 24 | 0 | 0 |
| scz_05_lateral_micro_hormigon | 17 % | 3 | 115 | 18 | 3 |

Las tres últimas columnas son observaciones (cajas sumadas en todos los cuadros).

## Por video

### scz_01_lateral_auto_hormigon.mp4 (99.2 s, 2976 cuadros, 848x478)

| Modelo | Cuadros con bache | Eventos de bache | Conf. máx. bache | Cuadros con grieta | s por cuadro |
|---|---:|---:|---:|---:|---:|
| YOLO12s RDD2022 (baches y grietas) | 203 (7 %) | 22 | 0.86 | 1503 | 0.17 |
| Farzad YOLOv8n-seg (baches) | 87 (3 %) | 14 | 0.87 | 0 | 0.06 |
| Piloto v1: Colab, 2 epocas (baches) | 471 (16 %) | 49 | 0.59 | 0 | 0.12 |
| YOLO del 1er parcial, Bisman (baches) | 270 (9 %) | 28 | 0.83 | 0 | 0.11 |

### scz_02_frontal_avenida.mp4 (67.6 s, 2027 cuadros, 848x478)

| Modelo | Cuadros con bache | Eventos de bache | Conf. máx. bache | Cuadros con grieta | s por cuadro |
|---|---:|---:|---:|---:|---:|
| YOLO12s RDD2022 (baches y grietas) | 108 (5 %) | 5 | 0.81 | 209 | 0.14 |
| Farzad YOLOv8n-seg (baches) | 89 (4 %) | 15 | 0.78 | 0 | 0.07 |
| Piloto v1: Colab, 2 epocas (baches) | 291 (14 %) | 15 | 0.55 | 0 | 0.14 |
| YOLO del 1er parcial, Bisman (baches) | 72 (4 %) | 9 | 0.62 | 0 | 0.05 |

### scz_03_frontal_doble_via.mp4 (141.7 s, 4119 cuadros, 848x478)

| Modelo | Cuadros con bache | Eventos de bache | Conf. máx. bache | Cuadros con grieta | s por cuadro |
|---|---:|---:|---:|---:|---:|
| YOLO12s RDD2022 (baches y grietas) | 44 (1 %) | 7 | 0.83 | 1002 | 0.16 |
| Farzad YOLOv8n-seg (baches) | 110 (3 %) | 17 | 0.75 | 0 | 0.07 |
| Piloto v1: Colab, 2 epocas (baches) | 432 (10 %) | 49 | 0.67 | 0 | 0.13 |
| YOLO del 1er parcial, Bisman (baches) | 235 (6 %) | 19 | 0.57 | 0 | 0.10 |

### scz_04_lateral_micro_losas.mp4 (17.0 s, 475 cuadros, 850x478)

| Modelo | Cuadros con bache | Eventos de bache | Conf. máx. bache | Cuadros con grieta | s por cuadro |
|---|---:|---:|---:|---:|---:|
| YOLO12s RDD2022 (baches y grietas) | 3 (1 %) | 1 | 0.68 | 24 | 0.15 |
| Farzad YOLOv8n-seg (baches) | 3 (1 %) | 0 | 0.47 | 0 | 0.09 |
| Piloto v1: Colab, 2 epocas (baches) | 13 (3 %) | 3 | 0.34 | 0 | 0.13 |
| YOLO del 1er parcial, Bisman (baches) | 77 (16 %) | 9 | 0.62 | 0 | 0.08 |

### scz_05_lateral_micro_hormigon.mp4 (20.2 s, 565 cuadros, 850x478)

| Modelo | Cuadros con bache | Eventos de bache | Conf. máx. bache | Cuadros con grieta | s por cuadro |
|---|---:|---:|---:|---:|---:|
| YOLO12s RDD2022 (baches y grietas) | 6 (1 %) | 1 | 0.66 | 97 | 0.18 |
| Farzad YOLOv8n-seg (baches) | 6 (1 %) | 1 | 0.67 | 0 | 0.09 |
| Piloto v1: Colab, 2 epocas (baches) | 19 (3 %) | 4 | 0.43 | 0 | 0.14 |
| YOLO del 1er parcial, Bisman (baches) | 6 (1 %) | 2 | 0.36 | 0 | 0.12 |

