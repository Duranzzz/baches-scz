# Pesos: origen y licencias

Las licencias de estos pesos piden conservar el aviso de su origen junto a las copias.

| Archivo | Origen | Licencia |
|---|---|---|
| `yolo12s_rdd2022/yolo12s_RDD2022_best.pt` | [rezzzq/yolo12s-road-damage-rdd2022](https://huggingface.co/rezzzq/yolo12s-road-damage-rdd2022), entrenado con RDD2022 | MIT |
| `farzad_pothole_seg/best.pt` | [FarzadNekouee/YOLOv8_Pothole_Segmentation_Road_Damage_Assessment](https://github.com/FarzadNekouee/YOLOv8_Pothole_Segmentation_Road_Damage_Assessment) | MIT |
| `parcial1_bisman/weights/best.pt` | Bisman-Singh-Dev/YOLO-Pothole-Detection-Model, verificado en el 1er parcial | no documentada |
| `piloto_v1/weights/best.pt` | ajuste de 2 épocas en Colab (30-oct-2026) partiendo de [peterhdd/pothole-detection-yolov8](https://huggingface.co/peterhdd/pothole-detection-yolov8) | la del modelo de partida |

Todos se entrenaron con [ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0).
Los `.pt` son pickles: cargarlos ejecuta código, así que solo se cargan estos.
