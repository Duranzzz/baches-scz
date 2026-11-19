# Ajuste con cuadros de Santa Cruz (23-nov-2026)

Plan y regla de decisión, fijados el 23-nov **antes de etiquetar los cuadros y antes de entrenar**, como con la
regla del 22-nov (`entrenamiento_amd.md`). Los resultados se agregan abajo cuando estén.

## Datos

- **200 cuadros** de los cinco videos de Santa Cruz, con cajas de bache, armados con
  `entrenamiento/scz_cuadros.py`:
  - por video: scz_01, 55; scz_02, 40; scz_03, 75; scz_04, 15; scz_05, 15;
  - cuadros tomados de los videos ya difuminados (caras y placas), así que se pueden publicar.
- **Partición en el tiempo:** en cada video, el 60 % inicial sirve para ajustar (120 cuadros) y el 40 % final,
  después de 2 s de separación, para probar (80 cuadros). Así la prueba tiene baches que el ajuste no vio.
- **Qué cuadros:** en cada parte, la mitad se elige al azar y la otra mitad entre los cuadros donde algún modelo
  marcó un bache. Entre dos cuadros elegidos hay al menos 0,5 s.
- **Etiquetas:**
  - Un solo revisor ve cada cuadro con las cajas que propusieron los cinco modelos, numeradas y sin saber de
    cuál es cada una.
  - Acepta las que son baches y agrega, con coordenadas, los baches que ninguno vio.
  - Las grietas no se etiquetan.

## Candidatos

- **Tal como están:** v1 (Baches SCZ v1), a1, a2, B0 y el YOLO12s de referencia.
- **v2:**
  - Parte de v1 y se ajusta con las 6.507 fotos de entrenamiento más los 120 cuadros de ajuste, repetidos 5 veces.
  - Usa la misma receta de a1 (1.280 px, aumentos tipo WhatsApp).
  - Se promedia con v1 (α = 0,25, 0,5, 0,75 y 1), conservando las filas de grietas de v1.

## Regla

Un candidato reemplaza a v1 en la app solo si cumple las tres condiciones, todas a 640 px (el tamaño de la app):

1. AP50 de baches en los 80 cuadros de prueba de Santa Cruz: **al menos 0,03 más** que v1.
2. AP50 de baches en Egipto (test congelado): **no menos de 0,01 por debajo** de v1.
3. AP50 de baches en HRP4K-A: **no menos de 0,005 por debajo** de v1.

- **Entre los que cumplen:** gana el de mayor AP50 en Santa Cruz.
- **Si ninguno cumple:** v1 sigue y el resultado queda aquí.
- **Además:** se informa el AP50 de los modelos que no se ajustaron con los 200 cuadros, medido en los 200.

## Estabilidad

La receta de a1 se repite desde B0 con las semillas 1 y 2 (`entrenar.py --semilla`). A cada corrida se le aplica
el mismo promedio de v1 (α = 0,25, 4 clases), para ver cuánto varía el resultado con el azar del entrenamiento.

## Notas de la ejecución

- **Primer intento, descartado:** la primera v2 y las repeticiones con las semillas 1 y 2 se entrenaron con la
  clase "D40" en el `data.yaml`.
  - Ultralytics no emparejó esa clase con la de bache del modelo de partida: el log dice "Transferred 696/708" y
    la fila de bache quedó al azar, distinta de la receta de a1.
  - Se descartaron esas corridas.
- **Corrección (23-nov, 07:50 UTC):** se renombró la clase a "Pothole (D40)" (ahora 708/708) y v2 se volvió a
  entrenar sola en la GPU. Desde entonces, `armar_dataset.py` escribe ese nombre.
- **Estabilidad:** de las repeticiones se rehace solo la semilla 1, después de v2, si alcanza el tiempo de máquina.

## Resultados (23-nov)

**v2:** 96 épocas en 1,35 h en la MI300X, desde v1 (708/708). Luego se promedió con v1 a α = 0,25, 0,5, 0,75 y 1.

AP50 de baches a 640 px:

| Modelo | Santa Cruz, prueba | ídem, AP30 | Egipto | HRP4K-A | HRP4K-B | Test total |
|---|---:|---:|---:|---:|---:|---:|
| v1 | 0,087 | 0,219 | 0,472 | **0,029** | **0,059** | 0,276 |
| v2, α = 0,25 | 0,125 | 0,298 | **0,496** | 0,022 | 0,038 | 0,332 |
| v2, α = 0,5 | 0,218 | **0,423** | 0,482 | 0,016 | 0,017 | 0,396 |
| v2, α = 0,75 | **0,275** | 0,406 | 0,478 | 0,013 | 0,012 | 0,458 |
| v2, α = 1 | 0,214 | 0,349 | 0,419 | 0,007 | 0,014 | **0,477** |
| Mínimo que pide la regla | 0,117 | | 0,462 | 0,024 | | |

**Decisión: ninguna variante cumple la regla, así que v1 sigue en la app.**
- Todas bajan en HRP4K-A más de 0,005: de 0,029 a entre 0,022 y 0,007.
- La de α = 1 baja además en Egipto.

**Lectura:**
- **Los cuadros de la ciudad sirven en la ciudad.** Todas las variantes suben en la prueba de Santa Cruz; con
  α = 0,75, a más del triple de v1.
- **Se repite el patrón de a1 (22-nov).** Suben también en los conjuntos que el entrenamiento ya veía, pero bajan en
  HRP4K, el único que no vio. La caída se repite en la mitad B, que no entra en la regla.
- **HRP4K a 640 px está cerca del piso** (todos por debajo de 0,06), así que las diferencias son chicas en valor
  absoluto.
- **a2 sigue primero en la prueba de Santa Cruz** (0,325). Con IoU 0,3, v2 con α = 0,5 queda a la par (0,423 frente
  a 0,413).
- **Las etiquetas pueden jugar en contra de v2.** Salieron de las cajas de v1, a1, a2, B0 y la referencia, no de
  las de v2. Con IoU 0,3 esto pesa menos.
- **La prueba viene de los mismos cinco videos que el ajuste**, más adelante en el tiempo: mismas cámaras y días.
  La mejora puede ser menor en otras calles o con otros teléfonos.

**Archivos:**
- `informes/ajuste_scz.json`: la regla aplicada (`entrenamiento/decidir_ajuste.py`).
- `informes/ajuste_tabla_amd.json`: la tabla de la AMD (`entrenamiento/tabla_ajuste.py`).
- `informes/scz_cuadros/eval_prueba_640*.json`: el AP en los cuadros de prueba.
- Pesos, curvas y logs: Hugging Face, `resultados/amd_20261123/`.

## Estabilidad: resultado (semilla 1)

- **a1_s1:** la receta de a1 repetida desde B0 con la semilla 1. Hizo 84 épocas en 1,08 h (708/708).
- **wise4_s1:** su promedio con B0 (α = 0,25, 4 clases), hecho igual que v1.

AP50 de baches a 640 px en el test congelado:

| Modelo | Bisman | Brasil | Bucko | Egipto | Total | HRP4K-A | HRP4K-B |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 (pesos iniciales) | 0,187 | 0,588 | 0,306 | 0,356 | 0,196 | 0,034 | 0,065 |
| a1 (semilla 0, 22-nov) | 0,834 | 0,662 | 0,629 | 0,442 | 0,496 | 0,011 | 0,019 |
| a1_s1 (semilla 1) | 0,839 | 0,663 | 0,593 | 0,452 | 0,481 | 0,009 | 0,017 |
| v1 (promedio α = 0,25 de a1) | 0,346 | 0,607 | 0,408 | 0,472 | 0,276 | 0,029 | 0,059 |
| wise4_s1 (ídem, de a1_s1) | 0,327 | 0,624 | 0,377 | 0,476 | 0,260 | 0,027 | 0,043 |

- **La medición es reproducible:** B0 da lo mismo que el 22-nov.
- **El patrón se repite.**
  - El ajuste completo sube mucho en los conjuntos que vio y baja en HRP4K, y el promedio recupera buena parte de
    HRP4K.
  - Entre semillas, cada conjunto cambia hasta 0,04, el total 0,016 y HRP4K-A 0,002.
- **La elección del 22-nov estuvo en el límite.**
  - La regla pedía HRP4K-A ≥ B0 − 0,005 = 0,0291, y v1 la pasó con 0,0293, por 0,0002.
  - wise4_s1 queda en 0,0269. Con la semilla 1, el promedio α = 0,25 no habría pasado y la regla habría dejado
    B0.
  - El margen por el que pasó v1 es diez veces menor que la diferencia entre semillas, así que esa elección
    dependió del azar del entrenamiento.
  - Aun así, wise4_s1 supera a B0 en todos los conjuntos salvo HRP4K, igual que v1.
- **El descarte de v2 es firme salvo con α = 0,25.**
  - Esa variante queda 0,002 por debajo del mínimo, del orden de la variación entre semillas.
  - Las otras quedan entre 0,008 y 0,018 por debajo.
- **La variación es una estimación gruesa:** hubo una sola repetición, porque la de la semilla 2 se descartó con el
  primer intento.
