# Modelos sobre videos de Santa Cruz (2026-11-22)

Todos los cuadros, a 864x512 px (el tamaño de estos videos, sin agrandarlos: en la línea base, agrandar imágenes chicas empeoró), con la confianza mínima de cada modelo (entre paréntesis; la de la app, elegida en val) y NMS 0.7.
Sin etiquetas: son marcas, no aciertos. Más marcas puede ser más falsos positivos. Un **evento** es un tramo de 3 cuadros o más con marca (huecos de hasta 0.5 s se unen): se parece más a "un daño visto" que contar cuadros.

## Baches: % de cuadros con marca (eventos)

| Modelo | scz_01_lateral_auto_hormigon | scz_02_frontal_avenida | scz_03_frontal_doble_via | scz_04_lateral_micro_losas | scz_05_lateral_micro_hormigon |
|---|---:|---:|---:|---:|---:|
| YOLO12s RDD2022 (baches y grietas) (bache ≥ 0.25, grieta ≥ 0.4) | 7 % (22) | 5 % (5) | 1 % (7) | 1 % (1) | 1 % (1) |
| YOLO26s RDD destilado (baches y grietas) (bache ≥ 0.15, grieta ≥ 0.3) | 4 % (14) | 8 % (7) | 5 % (28) | 1 % (1) | 1 % (1) |
| YOLO26s promedio WiSE-FT a=0.25 (baches y grietas) (bache ≥ 0.1, grieta ≥ 0.25) | 4 % (12) | 6 % (8) | 5 % (42) | 1 % (1) | 1 % (0) |
| YOLO26s ajustado con nuestros datos (baches) (conf ≥ 0.15) | 21 % (31) | 15 % (23) | 38 % (48) | 3 % (2) | 8 % (6) |

## Grietas (modelos que las detectan): % de cuadros con marca (eventos)

| Modelo | scz_01_lateral_auto_hormigon | scz_02_frontal_avenida | scz_03_frontal_doble_via | scz_04_lateral_micro_losas | scz_05_lateral_micro_hormigon |
|---|---:|---:|---:|---:|---:|
| YOLO12s RDD2022 (baches y grietas) (bache ≥ 0.25, grieta ≥ 0.4) | 41 % (25) | 8 % (10) | 21 % (33) | 4 % (1) | 14 % (2) |
| YOLO26s RDD destilado (baches y grietas) (bache ≥ 0.15, grieta ≥ 0.3) | 49 % (23) | 17 % (19) | 36 % (31) | 22 % (6) | 64 % (4) |
| YOLO26s promedio WiSE-FT a=0.25 (baches y grietas) (bache ≥ 0.1, grieta ≥ 0.25) | 24 % (37) | 11 % (17) | 19 % (34) | 2 % (1) | 18 % (8) |

## Por video

### scz_01_lateral_auto_hormigon.mp4 (99.2 s, 2976 cuadros, 848x478)

| Modelo | Cuadros con bache | Eventos de bache | Conf. máx. bache | Cuadros con grieta | s por cuadro |
|---|---:|---:|---:|---:|---:|
| YOLO12s RDD2022 (baches y grietas) | 203 (7 %) | 22 | 0.86 | 1223 | 0.17 (cpu) |
| YOLO26s RDD destilado (baches y grietas) | 109 (4 %) | 14 | 0.71 | 1445 | 0.02 (gpu 0) |
| YOLO26s promedio WiSE-FT a=0.25 (baches y grietas) | 106 (4 %) | 12 | 0.70 | 716 | 0.01 (gpu 0) |
| YOLO26s ajustado con nuestros datos (baches) | 634 (21 %) | 31 | 0.81 | 0 | 0.01 (gpu 0) |

### scz_02_frontal_avenida.mp4 (67.6 s, 2027 cuadros, 848x478)

| Modelo | Cuadros con bache | Eventos de bache | Conf. máx. bache | Cuadros con grieta | s por cuadro |
|---|---:|---:|---:|---:|---:|
| YOLO12s RDD2022 (baches y grietas) | 108 (5 %) | 5 | 0.81 | 162 | 0.14 (cpu) |
| YOLO26s RDD destilado (baches y grietas) | 155 (8 %) | 7 | 0.67 | 336 | 0.01 (gpu 0) |
| YOLO26s promedio WiSE-FT a=0.25 (baches y grietas) | 116 (6 %) | 8 | 0.62 | 233 | 0.01 (gpu 0) |
| YOLO26s ajustado con nuestros datos (baches) | 305 (15 %) | 23 | 0.71 | 0 | 0.01 (gpu 0) |

### scz_03_frontal_doble_via.mp4 (141.7 s, 4119 cuadros, 848x478)

| Modelo | Cuadros con bache | Eventos de bache | Conf. máx. bache | Cuadros con grieta | s por cuadro |
|---|---:|---:|---:|---:|---:|
| YOLO12s RDD2022 (baches y grietas) | 44 (1 %) | 7 | 0.83 | 880 | 0.16 (cpu) |
| YOLO26s RDD destilado (baches y grietas) | 203 (5 %) | 28 | 0.62 | 1469 | 0.01 (gpu 0) |
| YOLO26s promedio WiSE-FT a=0.25 (baches y grietas) | 220 (5 %) | 42 | 0.56 | 787 | 0.01 (gpu 0) |
| YOLO26s ajustado con nuestros datos (baches) | 1559 (38 %) | 48 | 0.80 | 0 | 0.01 (gpu 0) |

### scz_04_lateral_micro_losas.mp4 (17.0 s, 475 cuadros, 850x478)

| Modelo | Cuadros con bache | Eventos de bache | Conf. máx. bache | Cuadros con grieta | s por cuadro |
|---|---:|---:|---:|---:|---:|
| YOLO12s RDD2022 (baches y grietas) | 3 (1 %) | 1 | 0.68 | 17 | 0.15 (cpu) |
| YOLO26s RDD destilado (baches y grietas) | 5 (1 %) | 1 | 0.43 | 105 | 0.01 (gpu 0) |
| YOLO26s promedio WiSE-FT a=0.25 (baches y grietas) | 5 (1 %) | 1 | 0.44 | 8 | 0.01 (gpu 0) |
| YOLO26s ajustado con nuestros datos (baches) | 12 (3 %) | 2 | 0.47 | 0 | 0.01 (gpu 0) |

### scz_05_lateral_micro_hormigon.mp4 (20.2 s, 565 cuadros, 850x478)

| Modelo | Cuadros con bache | Eventos de bache | Conf. máx. bache | Cuadros con grieta | s por cuadro |
|---|---:|---:|---:|---:|---:|
| YOLO12s RDD2022 (baches y grietas) | 6 (1 %) | 1 | 0.66 | 78 | 0.18 (cpu) |
| YOLO26s RDD destilado (baches y grietas) | 6 (1 %) | 1 | 0.46 | 361 | 0.01 (gpu 0) |
| YOLO26s promedio WiSE-FT a=0.25 (baches y grietas) | 4 (1 %) | 0 | 0.21 | 100 | 0.01 (gpu 0) |
| YOLO26s ajustado con nuestros datos (baches) | 46 (8 %) | 6 | 0.61 | 0 | 0.01 (gpu 0) |

