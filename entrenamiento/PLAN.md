# Plan de entrenamiento

Los pasos, con comandos, están en `GUIA.md`. La línea base está en `informes/linea_base.md`: con cámara
frontal (dashcam, celular en el parabrisas) los baches casi no se detectan, y esa es la cámara de los
vehículos. Por eso se reentrena.

## 1. Dos detectores, no uno

Las etiquetas son parciales: Bisman, Bucko y HRP4K solo marcan baches; Brasil y Egipto marcan grietas
pero sin tipo. Un solo detector de 4 clases entrenado con todo aprendería que las grietas sin marcar
de esas fotos son fondo, y perdería grietas.

- **Modelo A, baches (1 clase, nombre "D40").** Usa todas las fuentes. Es el que más falta hace.
- **Modelo B, grietas.** Sigue el detector actual de RDD2022, sin tocar.
- La segmentación de baches se queda para medir el área de un bache que ya encontró el detector.

`preview.py --det-baches` y `evaluar.py --det-baches` combinan los dos.

## 2. Datos del modelo A

Las fotos del proyecto con la partición de `splits.csv` (por video o tramo de ruta, sin casi copias
entre train y val/test). `armar_dataset.py` arma:

| Dataset | Train | Con bache | Val | Cómo se partió |
|---|---:|---:|---:|---|
| Bisman (de cerca) | 3.281 | 3.263 | 686 | Splits originales |
| Brasil | 1.518 | 363 | 358 | Bloques de 25 fotos consecutivas por ruta |
| Bucko (dashcam) | 1.261 | 875 | 133 | Por video; train suma atardecer, tarde y lluvia; noche fuera |
| Egipto | 1.066 | 814 | 45 | Por video; las aumentadas de Roboflow solo en train |
| **Total** | **7.126** | **5.315** | **1.222** | |

- Tapas y grietas quedan como fondo: las tapas marcadas de Bucko enseñan que no son baches.
- **HRP4K (900 fotos 4K) y la muestra de RDD2022 no entran al entrenamiento.** El test de HRP4K mide si
  el modelo generaliza a una cámara y un país que no vio.
- Se pensó sumar RDD2022 oficial (~36.000 fotos) y el train de HRP4K (22 GB para bajar e importar).
  No se hizo por tiempo; es el siguiente paso, junto con fotos de Santa Cruz.

## 3. Modelo y entrenamiento

Corrida **A1**: YOLO12s partiendo del checkpoint de RDD2022, que ya conoce la calzada. Ultralytics
rehace la cabeza para 1 clase y reusa la fila de "D40" del checkpoint por nombre (probado en CPU).
Control opcional **A2**: YOLO26s desde COCO, con la misma configuración, solo si sobra tiempo.

| Parámetro | Valor | Por qué |
|---|---|---|
| `imgsz` | 1280 | En dashcam el bache ocupa ~0,4 % de la imagen; a 640 se pierde (ver línea base) |
| `epochs` / `patience` | hasta 120 / 30 | Las épocas se eligen con la prueba para que entre en ~4 h; corta antes si val no mejora en 30 |
| `batch` | 32 | Entra holgado en 192 GB |
| `optimizer`, `lr0`, `lrf`, `cos_lr` | SGD, 0,005, 0,05, sí | Ajuste fino: la mitad del lr por defecto, para no borrar lo que ya sabe |
| `warmup_epochs` | 3 | Valor por defecto |
| `mosaic` / `close_mosaic` | 1,0 / 15 | Mosaico ayuda con objetos chicos; se apaga al final para afinar con fotos reales |
| `degrees`, `flipud` | 0, 0 | La cámara no rota y el cielo siempre está arriba |
| `fliplr` | 0,5 | Una calle espejada sigue siendo realista |
| `scale`, `translate` | 0,5, 0,1 | Valores por defecto: simulan baches más cerca o más lejos |
| `mixup`, `copy_paste`, `cutmix` | 0 | Dan imágenes irreales para este problema |
| `cache` | RAM (`--cache-ram`) | Con 7.126 fotos ocupa ~27 GB y saca la lectura de JPEG del camino |
| `seed`, `deterministic` | 0, no | Repetible en lo que importa, sin frenar la GPU |
| `save_period` | 10 | Un checkpoint cada 10 épocas, por si se cae la máquina |
| `amp` | sí | Ultralytics lo prueba antes de entrenar y lo apaga si en ROCm da distinto; si no, `--sin-amp` |

## 4. Umbrales y medición

1. `best.pt` lo elige ultralytics por val.
2. `evaluar.py --split val --barrer --det-baches best.pt` busca `CONF_BACHE` y `CONF_SEG_SOLO`,
   maximizando el F1 medio entre datasets. Cada dataset pesa igual, para que las fotos de cerca de Bisman
   no tapen a las de cámara frontal.
3. Con esos umbrales fijos, el test **una sola vez**, contra la línea base medida en la misma máquina.
4. **Regla de decisión, fijada antes de ver resultados:** A1 reemplaza al detector actual para baches
   solo si su AP50 de test lo supera en Bucko, Egipto y HRP4K (cámaras de vehículo).
5. Cuando haya video de Santa Cruz etiquetado: el mismo informe sobre ese video. Es el número que importa.

## 5. Tiempo y costo

Una MI300X a 1,99 USD/h. Preparación ~15 min, prueba de 2 épocas 10 a 20 min y corrida hasta ~4 h,
con la medición y la subida incluidas. En total, unos 10 USD. La prueba mide el tiempo real por época
antes de gastar.

## 6. Riesgos y cómo se cubren

- **ROCm.** La atención de YOLO12 en ultralytics usa multiplicación de matrices común, sin
  flash-attention. Probado en CPU; en GPU lo cubre la prueba de 2 épocas.
- **Primera época lenta.** MIOpen afina sus kernels la primera vez que ve cada tamaño de tensor.
  Por eso la prueba proyecta con la segunda época.
- **pip y el torch de ROCm.** Ultralytics 8.4.191 pide `torch>=1.8`, que el torch de la imagen ya
  cumple. La guía igual imprime `torch.version.hip` antes de seguir.
- **El checkpoint de RDD2022 viene de un fork de ultralytics.** Verificado: se reentrena con
  ultralytics 8.4.191 y pasa a 1 clase.
- **`optimizer=auto` ignora `lr0`.** Por eso va `optimizer="SGD"` explícito.
- **Lo que no se puede saber antes:** si mejorar en estos datasets se traslada a Santa Cruz. En otros
  trabajos, cambiar de región bajó mucho el rendimiento hasta ajustar con fotos locales.
