"""Entrena el detector de baches (1 clase) con la configuración de entrenamiento/PLAN.md.

    python entrenamiento/entrenar.py --datos /datos/ds_baches/data.yaml --modelo baches-scz-v1 --nombre a1 --prueba
    python entrenamiento/entrenar.py --datos /datos/ds_baches/data.yaml --modelo baches-scz-v1 --nombre a1
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
    # El modelo del proyecto: punto de partida para seguir ajustando (ultralytics rehace la cabeza si cambian las clases)
    "baches-scz-v1": comun.RAIZ / "modelos" / "baches_scz_v1.pt",
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


def aumentos_scz():
    """Aumentos que imitan los videos de Santa Cruz: celular comprimido por WhatsApp (848x478), con movimiento.
    Reemplazan a los de ultralytics por defecto, que se repiten aquí (Blur, MedianBlur, ToGray y CLAHE con p=0.01).
    Necesita albumentations (en la máquina: pip install albumentations, con opencv-python-headless)."""
    import albumentations as A
    return [
        A.Blur(p=0.01), A.MedianBlur(p=0.01), A.ToGray(p=0.01), A.CLAHE(p=0.01),
        A.ImageCompression(quality_range=(20, 60), p=0.3),   # WhatsApp recomprime fuerte
        A.Downscale(scale_range=(0.4, 0.8), p=0.25),          # video de 848x478 frente a fotos de 1280 o más
        A.MotionBlur(blur_limit=(3, 9), p=0.15),              # cámara en un vehículo en marcha
    ]


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
    ap.add_argument("--modelo", default="baches-scz-v1", help=f"{', '.join(MODELOS)} o una ruta .pt")
    ap.add_argument("--nombre", required=True, help="carpeta en runs/")
    ap.add_argument("--epocas", type=int, default=HIPER["epochs"])
    ap.add_argument("--batch", type=int, default=HIPER["batch"])
    ap.add_argument("--imgsz", type=int, default=HIPER["imgsz"])
    ap.add_argument("--workers", type=int, default=min(16, max(2, (os.cpu_count() or 4) - 4)))
    ap.add_argument("--sin-amp", action="store_true", help="si aparecen pérdidas NaN o mAP 0 (ROCm)")
    ap.add_argument("--cache-ram", action="store_true",
                    help="guardar las fotos decodificadas en RAM. Ojo: con Python 3.14 (forkserver) cada proceso "
                         "de carga recibe su propia copia y se acaba la memoria; usar --cache-disco")
    ap.add_argument("--cache-disco", action="store_true",
                    help="guardar las fotos decodificadas como .npy en disco (rápido y sin copias en RAM)")
    ap.add_argument("--prueba", action="store_true", help="2 épocas con el 10 %% del train y proyección de costo")
    ap.add_argument("--fraccion-prueba", type=float, default=0.1, help="fracción del train en --prueba")
    ap.add_argument("--horas", type=float,
                    help="freno de seguridad: si se pasa de estas horas, termina al cerrar la época en curso")
    ap.add_argument("--aumentos-scz", action="store_true",
                    help="aumentos que imitan video de celular por WhatsApp: compresión, menos resolución, movimiento")
    ap.add_argument("--reanudar", action="store_true")
    ap.add_argument("--device", default="0", help="0 = primera GPU; cpu para probar sin GPU")
    ap.add_argument("--semilla", type=int, default=HIPER["seed"], help="otra semilla repite la corrida con otro azar")
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
    cache = "ram" if args.cache_ram else "disk" if args.cache_disco else HIPER["cache"]
    hiper = dict(HIPER, epochs=args.epocas, batch=args.batch, imgsz=args.imgsz, workers=args.workers,
                 amp=not args.sin_amp, cache=cache, seed=args.semilla)
    fraccion = 1.0
    if args.prueba:
        fraccion = args.fraccion_prueba
        hiper.update(epochs=2, close_mosaic=0, save_period=-1, plots=False)
    nombre = args.nombre + ("_prueba" if args.prueba else "")
    extra = {"augmentations": aumentos_scz()} if args.aumentos_scz else {}

    modelo = YOLO(str(pesos))
    crono = Cronometro()
    crono.registrar(modelo)
    if args.horas and not args.prueba:
        # Freno de seguridad. No se usa el `time` de ultralytics: ese recalcula las épocas para LLENAR el
        # tiempo (con 4 h y 49 s por época pasaba de 150 a ~294 épocas), no es un tope.
        limite = time.time() + args.horas * 3600

        def frenar(tr):
            if time.time() > limite and not tr.stop:
                print(f"\nTope de {args.horas} h alcanzado: termina tras la época {tr.epoch + 1}", flush=True)
                tr.stop = True

        modelo.add_callback("on_fit_epoch_end", frenar)
        hiper_registro = {"tope_horas": args.horas}
    else:
        hiper_registro = {}
    modelo.train(data=str(args.datos), project=str(runs), name=nombre, exist_ok=args.prueba,
                 fraction=fraccion, device=args.device, **hiper, **extra)
    hiper["aumentos_scz"] = bool(args.aumentos_scz)
    hiper.update(hiper_registro)
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
