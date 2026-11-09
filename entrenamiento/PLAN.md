# Plan de entrenamiento

> **Nota del 2026-11-21:** este plan no se ejecutó (ni A1 ni A2; tampoco se bajaron RDD2022 ni HRP4K
> completos). `armar_dataset.py` y `entrenar.py --prueba` se probaron en CPU con los datos actuales:
> 7.126 fotos de train (5.315 con bache) y 1.222 de val. Para presentar al día siguiente se usa una
> versión rápida de A1 con esos datos: ver la ruta rápida en `GUIA.md`.

Estado al 2026-10-28. Los pasos concretos, con comandos, están en `GUIA.md`. La línea base está en
`informes/linea_base.md`: los modelos públicos casi no encuentran baches en cámara frontal (dashcam,
celular en el parabrisas), que es justo la cámara de los camiones. Reentrenar es obligatorio.

## 1. Dos detectores, no uno

Las etiquetas son parciales: Bisman, Bucko y HRP4K solo marcan baches; Brasil y Egipto marcan grietas
pero sin tipo. Un solo detector de 4 clases entrenado con todo aprendería que las grietas sin marcar
de esas fotos son fondo, y perdería grietas.

- **Modelo A, baches (1 clase, nombre "D40").** Usa todas las fuentes. Es el que más falta hace.
- **Modelo B, grietas.** Sigue el YOLO12s de RDD2022 sin tocar. Se reentrenaría solo con RDD2022,
  la única fuente con grietas tipificadas, y solo si sobra presupuesto después del modelo A.
- La segmentación de Farzad se queda para medir el área de un bache que ya encontró el detector.

`preview.py --det-baches` y `evaluar.py --det-baches` combinan los dos.

## 2. Datos del modelo A

| Fuente | Train | Val | Test | Nota |
|---|---|---|---|---|
| RDD2022 oficial (sekilab) | 80 % | 10 % | 10 % | ~36.000 fotos de train etiquetadas en 6 países, sin China_Drone (vista cenital). Solo D40 = bache. Fotos sin bache: 1 de cada 3 |
| HRP4K oficial | 2.286 | 900 | 900 | El zip oficial trae 4.203 etiquetas de train pero solo 2.286 imágenes |
| Bisman | 3.281 | 686 | 677 | Splits originales sin casi copias |
| Bucko (dashcam) | 1.261 (sin noche) | 1 video de día | 2 videos de día | Por video |
| Brasil | 1.518 | 358 | 359 | Bloques de 25 fotos consecutivas por ruta |
| Egipto | 1.066 | 45 limpias | 91 limpias | Por video; las aumentadas de val/test se excluyen |

Estimado: ~21.500 fotos de train por época y ~3.500 de val. El número exacto lo imprime `armar_dataset.py`.

**Verificado el 2026-10-28, leyendo partes de los zips oficiales sin bajarlos enteros:**

- RDD2022: los enlaces de S3 del README de sekilab dan "Access Denied". La copia válida es la de
  Figshare (artículo 21431547): un zip de 13,3 GB con un zip por país (Noruega: 10,6 GB, fotos de
  4040 × 2035), CC BY 4.0, MD5 publicado. Los XML traen D00, D10, D20 y D40, y además D43, D44, D50
  y Repair, que se descartan. Así el modelo aprende que una senda, una tapa o un parche no es un bache.
- **El RDD2022 de dronefreak mezcla esos otros códigos dentro de su clase "pothole"** (revisado a ojo
  en su test). No usarlo para baches; sus grietas sí están bien.
- En Noruega, los XML tienen `<name>rotation</name>` dentro de atributos, fuera de los objetos: el
  importador lee solo los `<object>`, con un parser.
- HRP4K en Zenodo (10.5281/zenodo.17522874): 8,7 GB, CC BY 4.0, ya en formato YOLO. Sus etiquetas de
  test coinciden con la copia que teníamos (solo cambia el redondeo).

Otras decisiones de datos:

- Todo sale de `fotos/`, `etiquetas/` y `splits.csv`. Al importar RDD2022 y HRP4K se vuelve a correr
  `preparar_splits.py`: no cambia la partición de las fotos que ya estaban (probado), suma las nuevas,
  excluye la muestra de dronefreak y vuelve a buscar casi copias contra todo el train. **La línea base
  se vuelve a medir sobre esos splits finales** antes de comparar.
- Tapas y grietas quedan como fondo: las tapas marcadas de Bucko enseñan que no son baches (Brasil no marca tapas).
- Las fotos se guardan con un lado mayor de 1.280 px como máximo; las etiquetas YOLO son relativas y
  no cambian.

## 3. Modelo y entrenamiento

Corrida principal **A1**: YOLO12s partiendo del checkpoint de RDD2022 (rezzzq), que ya conoce
la calzada. Control opcional **A2**: YOLO26s desde COCO, con la misma configuración. Gana el de mejor
resultado en val **de cámara frontal**, no en el promedio.

| Parámetro | Valor | Por qué |
|---|---|---|
| `imgsz` | 1280 | En dashcam el bache ocupa ~0,4 % de la imagen; a 640 se pierde (ver línea base) |
| `epochs` / `patience` | 120 / 30 | Corta antes si val deja de mejorar 30 épocas seguidas |
| `batch` | 32 | Entra holgado en 192 GB |
| `optimizer`, `lr0`, `lrf`, `cos_lr` | SGD, 0,005, 0,05, sí | Ajuste fino: la mitad del lr por defecto, para no borrar lo que ya sabe |
| `warmup_epochs` | 3 | Valor por defecto |
| `mosaic` / `close_mosaic` | 1,0 / 15 | Mosaico ayuda con objetos chicos; se apaga al final para afinar con fotos reales |
| `degrees`, `flipud` | 0, 0 | La cámara no rota y el cielo siempre está arriba |
| `fliplr` | 0,5 | Una calle espejada sigue siendo realista |
| `scale`, `translate` | 0,5, 0,1 | Valores por defecto: simulan baches más cerca o más lejos |
| `mixup`, `copy_paste`, `cutmix` | 0 | Dan imágenes irreales para este problema |
| `cache` | no | En RAM ocuparía ~110 GB; el cuello de botella es la GPU, no leer JPEG con 16 procesos |
| `seed`, `deterministic` | 0, no | Repetible en lo que importa, sin frenar la GPU |
| `save_period` | 10 | Un checkpoint cada 10 épocas, por si se cae la máquina |
| `amp` | sí | Ultralytics lo prueba antes de entrenar y lo apaga si en ROCm da distinto; si no, `--sin-amp` |

## 4. Umbrales y medición

1. `best.pt` lo elige ultralytics por val.
2. `evaluar.py --split val --barrer --det-baches best.pt --imgsz 1280` busca `CONF_BACHE` y
   `CONF_SEG_SOLO`, maximizando el F1 medio entre datasets. Cada dataset pesa igual, para que las
   fotos de cerca de Bisman no tapen a las de cámara frontal.
3. Con esos umbrales fijos, el test **una sola vez**, y comparar contra la línea base por dataset.
4. Cuando haya video de Santa Cruz etiquetado: el mismo informe sobre ese video. Es el número que importa.

## 5. Presupuesto: 100 USD ≈ 50 h de una MI300X a 1,99 USD/h

Detalle por paso en `GUIA.md`. Primera tanda (preparar + A1): ~8 a 17 h, ~16 a 34 USD. Las horas
de entrenamiento son estimadas (~21.500 fotos por época a 1280 px, del orden de 50 a 100 fotos por
segundo); la prueba de 2 épocas mide el número real antes de gastar. Reservar al menos 40 USD para
reentrenar con datos de Santa Cruz, dentro de los 30 días de vigencia de los créditos.

## 6. Riesgos y cómo se cubren

- **ROCm.** La atención de YOLO12 en ultralytics usa multiplicación de matrices común, sin
  flash-attention. Probado en CPU. En GPU lo cubre la prueba de 2 épocas.
- **Primera época lenta.** MIOpen afina sus kernels la primera vez que ve cada tamaño de tensor.
  Por eso la prueba proyecta con la segunda época.
- **pip y el torch de ROCm.** Ultralytics 8.4.191 pide `torch>=1.8`, que el torch de la imagen ya
  cumple. Aun así, `preparar_maquina.sh` compara la versión antes y después, y se detiene si cambió.
- **El checkpoint de rezzzq viene de un fork de ultralytics.** Verificado: se reentrena con
  ultralytics 8.4.191 y pasa a 1 clase.
- **`optimizer=auto` ignora `lr0`.** Por eso va `optimizer="SGD"` explícito.
- **El split de RDD2022 es nuestro** (80/10/10 al azar por país): el test oficial no tiene etiquetas.
- **Licencias.** Bisman no documenta la licencia de sus imágenes, y ultralytics es AGPL-3.0.

Toda la cadena se probó en esta PC sobre réplicas chicas de los zips reales, armadas con archivos de
verdad de RDD2022 y HRP4K: importar, splits, dataset, prueba de entrenamiento (CPU, 320 px),
evaluación y preview con el detector nuevo, armado del paquete y descarga con MD5.
