# Línea base: modelos públicos sin reentrenar

Medido el 2026-10-28 en CPU, sobre el test congelado de `splits.csv` (solo fotos evaluables).
Detector: YOLO12s RDD2022 (rezzzq). Segmentación: YOLOv8n-seg de baches (FarzadNekouee).
Umbrales de `preview.py` elegidos en val (ver abajo). Reproducir:

```bash
python herramientas/evaluar.py --split val --barrer
python herramientas/evaluar.py --split test
python herramientas/evaluar.py --split test --imgsz 1280
```

Un acierto es una caja con IoU ≥ 0,5 contra una etiqueta de la misma clase. AP50 con 101 puntos de
recall. "preview" es la salida combinada de `preview.py` en su punto de operación. "Por foto": de las
fotos que tienen ese daño, cuántas marcó (R), y de las que marcó, cuántas lo tenían (P). No depende de
cómo se cortó cada daño en cajas, y es lo que importa para elegir cuadros de un video.
JSON completos: `eval_yolo12s_RDD2022_best_{val,test}_{640,1280}.json`.

## Baches (D40), 640 px

| Dataset | Cámara | Baches | AP50 det | AP50 seg | preview P | preview R | preview F1 | por foto P | por foto R |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Bisman | celular, de cerca | 887 | 0,153 | **0,481** | 0,59 | 0,51 | 0,546 | 1,00 | 0,76 |
| Brasil | desde vehículo, ruta | 222 | **0,472** | 0,056 | 0,53 | 0,47 | 0,499 | 0,71 | 0,70 |
| Bucko | dashcam 1080p | 744 | 0,156 | 0,003 | 0,50 | 0,12 | 0,197 | 0,97 | 0,39 |
| Egipto | celular frontal | 56 | 0,151 | 0,098 | 0,29 | 0,21 | 0,245 | 0,69 | 0,35 |
| HRP4K | 4K en perspectiva | 921 | 0,035 | 0,005 | 0,21 | 0,04 | 0,073 | 0,90 | 0,26 |
| Todos | | 2.830 | 0,133 | 0,157 | 0,51 | 0,25 | 0,332 | 0,93 | 0,51 |

Fotos de test con al menos un bache: Bisman 100 %, Brasil 32 %, Bucko 78 %, Egipto 56 %, HRP4K 67 %.
La P por foto solo dice algo cuando supera esa proporción: marcar todo al azar ya la alcanza.

## Grietas, 640 px

| Dataset | Clase | Etiquetas | AP50 det | preview F1 | por foto P | por foto R |
|---|---|---:|---:|---:|---:|---:|
| RDD2022 (muestra) | D00 longitudinal | 256 | 0,924 | 0,891 | 0,97 | 0,99 |
| | D10 transversal | 87 | 0,957 | 0,898 | 0,91 | 0,97 |
| | D20 cocodrilo | 79 | 1,000 | 0,981 | 0,98 | 1,00 |
| | grieta (cualquiera) | 422 | 0,948 | 0,912 | 0,99 | 0,99 |
| Brasil | grieta | 2.986 | 0,061 | 0,117 | 0,92 | 0,82 |
| Egipto | grieta | 70 | 0,075 | 0,092 | 0,75 | 0,22 |

Fotos de test con al menos una grieta: RDD2022 64 %, Brasil 81 %, Egipto 45 %.

## 640 contra 1280 px

Mismos modelos y umbrales (calibrados a 640; a 1280 no se recalibraron).

| Dataset | Tamaño de las fotos | Baches AP50 det 640 → 1280 | Baches por foto R 640 → 1280 |
|---|---|---|---|
| Bisman | ≤ 640 | 0,153 → 0,087 | 0,76 → 0,64 |
| Brasil | 1024 × 630 | 0,472 → 0,478 | 0,70 → 0,80 |
| Bucko | 1920 × 1080 | 0,156 → **0,226** | 0,39 → 0,48 |
| Egipto | 640 × 640 | 0,151 → 0,098 | 0,35 → 0,33 |
| HRP4K | 3840 × 2160 | 0,035 → **0,110** | 0,26 → 0,41 |

Grietas en RDD2022 (fotos de 512 a 720 px): AP50 0,948 → 0,789.

**1280 ayuda cuando la foto es grande** (dashcam 1080p: +45 %; 4K: ×3), porque a 640 el bache
lejano queda en unos pocos píxeles. **Empeora con fotos chicas**, que hay que agrandar. Regla:
analizar a un tamaño cercano al de la foto, con tope en 1280. El video de los camiones va a ser
1080p o más, así que va a 1280, y el modelo reentrenado también se entrena a 1280.

## Umbrales elegidos en val (640 px)

Barrido de `CONF_DET` × `CONF_SEG_SOLO`, maximizando el F1 medio de baches entre Bisman, Brasil,
Bucko y Egipto (cada dataset pesa igual). Mejor: **`CONF_DET` 0,20 y `CONF_SEG_SOLO` 0,55**, con F1
medio 0,320, contra 0,314 con 0,25 y 0,25. La superficie es casi plana (0,30 a 0,32 en casi toda la grilla).
Apagar los baches que solo ve la segmentación la baja a 0,20: en fotos de cerca la segmentación es
la que encuentra los baches.

## Qué dice

1. **En cámara frontal los baches casi no se detectan**: AP50 de 0,16 en dashcam (Bucko), 0,15 en
   celular frontal (Egipto) y 0,035 en HRP4K; a 1280 px, 0,23 y 0,11 en Bucko y HRP4K. Es la cámara
   de los camiones. Reentrenar es obligatorio.
2. La segmentación solo sirve con fotos de cerca (0,48 en Bisman). En cámara frontal da ≤ 0,10.
3. Brasil (0,47) es la excepción: fotos de ruta tomadas más cerca y más hacia abajo que una dashcam.
4. Las grietas en RDD2022 salen casi perfectas (D20: 1,000). Es muy probable que el modelo haya visto
   esas fotos al entrenar. **No sirven como medida de generalización.**
5. Las grietas en Brasil dan AP50 0,06 por cómo están marcadas: las cajas salen de máscaras partidas
   en trozos (~8 por foto), y el detector marca una caja por grieta. Por foto, el detector encuentra
   grietas en el 82 % de las fotos de Brasil que las tienen, con precisión 0,92 (marcar todas daría
   0,81, porque el 81 % de las fotos tiene grietas). **Para grietas, la medida por foto es
   la que vale.** En Egipto (celular frontal) sí es débil: 22 % por foto.
6. Por foto, en dashcam (Bucko) el sistema marca el 39 % de los cuadros con bache, con precisión
   0,97 contra un 78 % al azar. Sirve para elegir algunos cuadros y revisarlos a mano; para contar
   baches, no.

## El YOLO del 1.er parcial como detector de baches (medido el 2026-11-21)

El modelo de Bisman que se verificó en el 1.er parcial (una clase, "pothole", entrenado a 640 con el
train de Bisman), copiado de `1st_Parcial/baches/modelo/best.pt` (ya no se guarda en este repositorio).
Mismo test y 640 px, con `--det-baches`: los baches salen de ese modelo y las grietas siguen saliendo de
YOLO12s. Umbrales elegidos en val: `CONF_BACHE` 0,25 y `CONF_SEG_SOLO` 0,55.

| Dataset | Cámara | AP50 YOLO12s | AP50 1.er parcial | Por foto R: YOLO12s → 1.er parcial |
|---|---|---:|---:|---|
| Bisman | celular, de cerca | 0,153 | **0,681** | 0,76 → 0,95 |
| Brasil | desde vehículo, ruta | **0,472** | 0,190 | 0,70 → 0,35 |
| Bucko | dashcam 1080p | **0,156** | 0,006 | 0,39 → 0,13 |
| Egipto | celular frontal | **0,151** | 0,081 | 0,35 → 0,39 |
| HRP4K | 4K en perspectiva | **0,035** | 0,009 | 0,26 → 0,14 |

Sirve para fotos de cerca, que es de donde viene, y casi no ve baches desde un vehículo.
JSON: `eval_yolo12s_RDD2022_best+parcial1_bisman_best_{val,test}_640.json`.

## Lo que no se mide aquí

- **Baches de RDD2022**: la redistribución de dronefreak marca como "pothole" también sendas y líneas
  borrosas, tapas de alcantarilla y parches. Revisado a ojo en 8 fotos de su test. Para entrenar
  hay que usar el RDD2022 oficial (ver `entrenamiento/PLAN.md`).
- **Santa Cruz**: no hay video propio todavía. Es el número que importa.
- La severidad: no hay etiquetas de severidad.
