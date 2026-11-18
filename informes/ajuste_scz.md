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
