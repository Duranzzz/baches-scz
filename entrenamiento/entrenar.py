"""Entrena el detector de baches (1 clase) con la configuración de entrenamiento/PLAN.md.

    python entrenamiento/entrenar.py --datos /datos/ds_baches/data.yaml --modelo yolo12s-rdd --nombre a1 --prueba
    python entrenamiento/entrenar.py --datos /datos/ds_baches/data.yaml --modelo yolo12s-rdd --nombre a1
    python entrenamiento/entrenar.py --nombre a1 --reanudar           # si se cortó: sigue desde last.pt

--prueba: 2 épocas con el 10 % del train. Mide por separado el entrenamiento y la validación de la
segunda época (en la primera, ROCm afina sus kernels para cada tamaño nuevo y tarda más) y proyecta
cuánto tardaría y costaría la corrida completa. Hacerla siempre antes de una corrida larga.
Los resultados quedan en runs/<nombre>/ (weights/best.pt, results.csv, gráficos) y runs/<nombre>/resumen.json.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comun  # noqa: E402

PRECIO_HORA = 1.99   # USD por hora de una MI300X en AMD Developer Cloud (confirmar en la consola)
MODELOS = {
    # YOLO12s ya ajustado a RDD2022 (rezzzq): conoce la calzada. Ultralytics rehace la cabeza para 1 clase.
    "yolo12s-rdd": comun.RAIZ / "modelos" / "yolo12s_rdd2022" / "yolo12s_RDD2022_best.pt",
    # Controles desde COCO (ultralytics los baja solo)
    "yolo26s": "yolo26s.pt",
    "yolo12s": "yolo12s.pt",
    "yolo26m": "yolo26m.pt",
}
# Ver la tabla de entrenamiento/PLAN.md: cada valor tiene su porqué allí.
HIPER = dict(
    epochs=120, patience=30, imgsz=comun.LADO, batch=32,
    optimizer="SGD", lr0=0.005, lrf=0.05, momentum=0.937, weight_decay=5e-4, warmup_epochs=3, cos_lr=True,
    mosaic=1.0, close_mosaic=15, degrees=0.0, flipud=0.0, fliplr=0.5, scale=0.5, translate=0.1,
    mixup=0.0, copy_paste=0.0, cutmix=0.0,
    cache=False, seed=0, deterministic=False, save_period=10, amp=True, plots=True,
)


class Cronometro:
    """Mide preparación, entrenamiento y validación por época con los callbacks de ultralytics."""

    def __init__(self):
        self.t0 = time.time()
        self.inicio_entrenamiento = None
        self.epocas = []   # (s entrenando, s validando)
        self._t = None

    def registrar(self, modelo):
        modelo.add_callback("on_train_start", lambda tr: setattr(self, "inicio_entrenamiento", time.time()))
        modelo.add_callback("on_train_epoch_start", lambda tr: setattr(self, "_t", time.time()))
        modelo.add_callback("on_train_epoch_end", self._fin_train)
        modelo.add_callback("on_fit_epoch_end", self._fin_val)

    def _fin_train(self, tr):
        self._tt = time.time() - self._t
        self._t = time.time()

    def _fin_val(self, tr):
        self.epocas.append((self._tt, time.time() - self._t))


def resumen_resultados(carpeta):
    filas = list(csv.DictReader(open(carpeta / "results.csv", encoding="utf-8")))
    filas = [{k.strip(): v for k, v in f.items()} for f in filas]
    mejor = max(filas, key=lambda f: float(f["metrics/mAP50(B)"]))
    return {"epocas": len(filas), "mejor_epoca": int(float(mejor["epoch"])),
            "mAP50": float(mejor["metrics/mAP50(B)"]), "mAP50-95": float(mejor["metrics/mAP50-95(B)"]),
            "P": float(mejor["metrics/precision(B)"]), "R": float(mejor["metrics/recall(B)"])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", type=Path, help="data.yaml de armar_dataset.py")
    ap.add_argument("--modelo", default="yolo12s-rdd", help=f"{', '.join(MODELOS)} o una ruta .pt")
    ap.add_argument("--nombre", required=True, help="carpeta en runs/")
    ap.add_argument("--epocas", type=int, default=HIPER["epochs"])
    ap.add_argument("--batch", type=int, default=HIPER["batch"])
    ap.add_argument("--imgsz", type=int, default=HIPER["imgsz"])
    ap.add_argument("--workers", type=int, default=min(16, max(2, (os.cpu_count() or 4) - 4)))
    ap.add_argument("--sin-amp", action="store_true", help="si aparecen pérdidas NaN o mAP 0 (ROCm)")
    ap.add_argument("--cache-ram", action="store_true",
                    help="guardar las fotos decodificadas en RAM (~110 GB); solo si la GPU espera al disco")
    ap.add_argument("--prueba", action="store_true", help="2 épocas con el 10 %% del train y proyección de costo")
    ap.add_argument("--reanudar", action="store_true")
    ap.add_argument("--device", default="0", help="0 = primera GPU; cpu para probar sin GPU")
    args = ap.parse_args()

    from ultralytics import YOLO

    runs = comun.RAIZ / "runs"
    if args.reanudar:
        last = runs / args.nombre / "weights" / "last.pt"
        if not last.exists():
            sys.exit(f"no existe {last}")
        YOLO(str(last)).train(resume=True)
        return
    if not args.datos or not args.datos.exists():
        sys.exit("falta --datos (el data.yaml de armar_dataset.py)")

    pesos = MODELOS.get(args.modelo, args.modelo)
    if args.modelo == "yolo12s-rdd" and not Path(pesos).exists():
        sys.path.insert(0, str(comun.RAIZ))
        import preview
        preview.descargar_modelos()
    hiper = dict(HIPER, epochs=args.epocas, batch=args.batch, imgsz=args.imgsz, workers=args.workers,
                 amp=not args.sin_amp, cache="ram" if args.cache_ram else HIPER["cache"])
    fraccion = 1.0
    if args.prueba:
        fraccion = 0.1
        hiper.update(epochs=2, close_mosaic=0, save_period=-1, plots=False)
    nombre = args.nombre + ("_prueba" if args.prueba else "")

    modelo = YOLO(str(pesos))
    crono = Cronometro()
    crono.registrar(modelo)
    modelo.train(data=str(args.datos), project=str(runs), name=nombre, exist_ok=args.prueba,
                 fraction=fraccion, device=args.device, **hiper)
    carpeta = Path(modelo.trainer.save_dir)  # si runs/<nombre> ya existía, ultralytics le agrega un número
    total = time.time() - crono.t0
    resumen = {"modelo": args.modelo, "datos": str(args.datos), "hiper": hiper, "fraccion": fraccion,
               "segundos_total": round(total), "epocas_medidas": crono.epocas}

    if args.prueba and crono.epocas:
        preparacion = crono.inicio_entrenamiento - crono.t0
        s_train, s_val = crono.epocas[-1]
        por_epoca = s_train / fraccion + s_val
        horas = (preparacion + args.epocas * por_epoca) / 3600
        resumen["proyeccion"] = {"s_por_epoca": round(por_epoca), "epocas": args.epocas, "horas": round(horas, 2),
                                 "usd": round(horas * PRECIO_HORA, 2)}
        e1_train, e1_val = crono.epocas[0]
        print(f"\nPRUEBA: preparación {preparacion:.0f} s | época 1: {e1_train:.0f} + {e1_val:.0f} s (afinando "
              f"kernels) | época 2: entrenamiento {s_train:.0f} s con el {fraccion:.0%} del train, "
              f"validación {s_val:.0f} s")
        print(f"Proyección: ~{por_epoca:.0f} s por época completa -> {args.epocas} épocas ≈ {horas:.1f} h "
              f"≈ {horas * PRECIO_HORA:.0f} USD (tope: patience puede cortarla antes)")
    elif (carpeta / "results.csv").exists():
        resumen["mejor"] = resumen_resultados(carpeta)
        print(f"\nMejor época: {resumen['mejor']}")
    (carpeta / "resumen.json").write_text(json.dumps(resumen, indent=1, default=str), encoding="utf-8")
    print(f"{carpeta / 'resumen.json'}  ({total / 3600:.2f} h)")


if __name__ == "__main__":
    main()
