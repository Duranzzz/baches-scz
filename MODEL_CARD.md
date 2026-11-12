# Ficha del modelo: Baches SCZ v1

| | |
|---|---|
| Archivos | `modelos/baches_scz_v1.pt` (Ultralytics, fp16, 20,4 MB) y `modelos/baches_scz_v1.onnx` (ONNX, fp32, 38,5 MB, entrada dinámica) |
| Tarea | Detección de daños en el pavimento, con 4 clases |
| Clases | 0 grieta longitudinal (D00), 1 grieta transversal (D10), 2 grieta de cocodrilo (D20), 3 bache (D40) |
| Arquitectura | YOLO26s (Ultralytics 8.4.191), 9.466.763 parámetros |
| Entrada | Imagen RGB: 640 px de lado mayor en la app; 1280 px para video de 1080p o más |
| Versión | 1.0.0, 22 de noviembre de 2026 |
| Licencia | AGPL-3.0 (ver [Origen y licencia](#origen-y-licencia)) |

| Archivo | SHA256 |
|---|---|
| `baches_scz_v1.pt` | `965452093a29b836237c003e92d21ee3449f00d97b33ba1ed3ea9b57e9fd5215` |
| `baches_scz_v1.onnx` | `92da8c0f71d83ef9614b7e9321d385ea114d4825019186d20695d43659b8821a` |

## Uso

```python
from ultralytics import YOLO

modelo = YOLO("modelos/baches_scz_v1.onnx")          # el ONNX no ejecuta código al cargarse
r = modelo.predict("foto.jpg", imgsz=640, conf=0.10)[0]
for caja, clase, conf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
    print(modelo.names[int(clase)], round(conf, 2), caja)
```

- **Umbrales recomendados**, elegidos en validación: 0,10 para baches y 0,25 para grietas. Son los de la app.
- **El `.pt` es un pickle:** cargarlo ejecuta código, así que solo debe usarse el de este repositorio, con su
  SHA256 verificado. Para inferencia, mejor el ONNX.
- **En el navegador:** la app web usa el ONNX con ONNX Runtime Web
  ([Duranzzz/baches-scz-app](https://github.com/Duranzzz/baches-scz-app)).

## Cómo se obtuvo

1. **Datos:** 6.507 fotos de entrenamiento y 1.222 de validación de cuatro conjuntos públicos: Bisman, Brasil,
   Bucko (Eslovaquia) y Egipto. Se depuraron antes: salieron 619 fotos y se corrigieron 14 etiquetas
   (ver `README.md`).
2. **Pesos de partida:** un YOLO26s público con las clases de RDD2022.
3. **Ajuste:** con esas fotos, en una GPU AMD MI300X.
   - 91 épocas a 1.280 px, con SGD (tasa inicial 0,005, coseno).
   - Aumentos que imitan el video enviado por WhatsApp: compresión, menor resolución y desenfoque de movimiento.
4. **Promedio de pesos** (WiSE-FT, α = 0,25) entre los pesos de partida y los ajustados.
   - Conserva las filas de grietas de partida y promedia la de baches.
   - Se eligió con una regla fijada antes de ver los resultados: el mejor en validación entre los que no
     empeoran en un conjunto que el ajuste no vio (HRP4K).
   - Detalle completo en `informes/entrenamiento_amd.md`.

## Evaluación

AP50 (IoU ≥ 0,5, 101 puntos de exhaustividad) en el test congelado de `splits.csv`, a 640 px:

| | Bisman | Brasil | Bucko | Egipto | HRP4K | Total |
|---|---:|---:|---:|---:|---:|---:|
| Baches | 0,346 | 0,607 | 0,408 | 0,472 | 0,040 | 0,276 |
| Grietas | | 0,063 | | 0,199 | | |

- **Contra el detector anterior:** el que usaba la app antes obtiene 0,128 de AP50 total en baches.
- **Fuga en los datos:** Bucko, Brasil y Bisman comparten caminos o fotos espejadas entre partes, así que sus
  números son optimistas. Egipto es el más limpio.
- **Videos de Santa Cruz:** en los cinco videos, sin etiquetas, marca baches en el 4,4 % de los cuadros (451 de
  10.162).

## Usos previstos y no previstos

**Previstos:**
- Apoyar la inspección de calles: marcar posibles baches y grietas en fotos o videos de celular para que una
  persona los revise.
- Demostraciones y docencia.

**No previstos:**
- Decisiones automáticas sin revisión humana, como priorizar obras o asignar responsabilidades.
- Conducción o alertas de seguridad en tiempo real.
- Medir la profundidad o la severidad real de un daño.

## Limitaciones y sesgos

- **Santa Cruz sin medir:** no hay fotos de Santa Cruz de la Sierra etiquetadas, así que la precisión local no
  está medida.
- **Datos de otros países:** las fotos son de Brasil, Eslovaquia, Egipto y otras fuentes, con pavimentos,
  cámaras y luz distintos a los de la ciudad.
- **Baches lejanos:** los chicos y lejanos casi no se detectan a 640 px (HRP4K: 0,040).
- **Falsas grietas:** las juntas de losas de hormigón y la carrocería del propio vehículo aparecen a veces como
  grietas.
- **Etiquetas parciales:** Brasil marca solo el carril propio, y 243 fotos de entrenamiento dudosas siguen en los
  datos.

## Consideraciones éticas

- **Personas y placas:** los videos de la calle muestran personas y placas. La app procesa en el teléfono y no
  envía video a ningún servidor.
- **Ubicación:** cada daño se guarda en el teléfono con su ubicación GPS; solo sale del dispositivo si la
  persona lo exporta.

## Origen y licencia

- **Modelo derivado:**
  - **Pesos de partida:** [TamAko783/YOLO26s_RDD_FRDC_Distilled_v2](https://huggingface.co/TamAko783/YOLO26s_RDD_FRDC_Distilled_v2) (AGPL-3.0).
  - **Biblioteca de entrenamiento:** [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0).
  - **Modificaciones del 22-nov-2026:** ajuste con las fotos del proyecto y promedio de pesos.
  - **Licencia:** por todo lo anterior, se distribuye bajo la misma licencia, AGPL-3.0 (`LICENSE`).
- **Datos:** cada conjunto de datos conserva su propia licencia (ver `README.md`).
