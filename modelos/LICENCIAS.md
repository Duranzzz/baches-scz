# Pesos: origen y licencias

Las licencias de estos pesos piden conservar el aviso de su origen junto a las copias.

| Archivo | Origen | Licencia |
|---|---|---|
| `yolo12s_rdd2022/yolo12s_RDD2022_best.pt` | [rezzzq/yolo12s-road-damage-rdd2022](https://huggingface.co/rezzzq/yolo12s-road-damage-rdd2022), entrenado con RDD2022 | MIT |
| `farzad_pothole_seg/best.pt` | [FarzadNekouee/YOLOv8_Pothole_Segmentation_Road_Damage_Assessment](https://github.com/FarzadNekouee/YOLOv8_Pothole_Segmentation_Road_Damage_Assessment) | MIT |
| `parcial1_bisman/weights/best.pt` | Bisman-Singh-Dev/YOLO-Pothole-Detection-Model, verificado en el 1er parcial | no documentada |
| `piloto_v1/weights/best.pt` | ajuste de 2 épocas en Colab (30-oct-2026) partiendo de [peterhdd/pothole-detection-yolov8](https://huggingface.co/peterhdd/pothole-detection-yolov8) | la del modelo de partida |
| `rdd_yolo26s/weights/best.pt` | [TamAko783/YOLO26s_RDD_FRDC_Distilled_v2](https://huggingface.co/TamAko783/YOLO26s_RDD_FRDC_Distilled_v2): YOLO26s destilado con RDD2022 y FRDC (D00, D10, D20, D40). Revisado con picklescan antes de cargarlo | AGPL-3.0 |
| `a1_y26s/weights/best.pt` | ajuste propio (AMD MI300X, 22-nov-2026) de `rdd_yolo26s` con las 6.507 fotos de train limpias y aumentos tipo WhatsApp; 91 épocas (paciencia 30). Curvas y argumentos en `a1_y26s/` | AGPL-3.0 (derivado) |
| `wise4_a25/weights/best.pt` | el de la app: promedio de pesos WiSE-FT (alfa 0,25) entre `rdd_yolo26s` y `a1_y26s`, con las 4 clases (`entrenamiento/promediar_pesos.py --clases partida`) | AGPL-3.0 (derivado) |

Todos se entrenaron con [ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0).
Los `.pt` son pickles: cargarlos ejecuta código, así que solo se cargan estos.
