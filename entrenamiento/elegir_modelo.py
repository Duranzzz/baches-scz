"""Tabla de AP50 de baches (D40) de los candidatos y la regla de selección, fijada el 22-nov-2026 ANTES de ver
los resultados:
  - el test de HRP4K (el único dominio que los ajustes sin HRP4K no vieron) se parte en dos mitades fijas por
    hash del nombre: la A sirve para elegir, la B solo para informar;
  - a 640 px (el tamaño de la app), se elige el de mayor AP50 en val (todos) entre los que no empeoran fuera de
    dominio: HRP4K-A >= la del modelo de partida (B0) - 0,005.
Los modelos entrenados con HRP4K (a2) no compiten en esa regla: para ellos HRP4K no es un dominio nuevo.
    python entrenamiento/elegir_modelo.py

Lee informes/eval_*.json y, para las mitades de HRP4K, las predicciones guardadas en salidas/cache/ (las deja
evaluar.py; si faltan, correr antes evaluar.py --split test con ese detector). Salida del 22-nov:
informes/entrenamiento_amd.md."""
import csv
import hashlib
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "herramientas"))
import evaluar as ev  # noqa: E402
import preview as pv  # noqa: E402

BASE = "yolo12s_RDD2022_best"
MODELOS = [  # (título, nombre en informes/ y salidas/cache, ¿vio HRP4K al entrenar?)
    ("YOLO12s (app hoy)", BASE, False),
    ("B0 YOLO26s destilado", f"{BASE}+rdd_yolo26s_dist_best", False),
    ("WiSE a=0,25", f"{BASE}+wise_a25_best", False),
    ("WiSE a=0,50", f"{BASE}+wise_a50_best", False),
    ("WiSE a=0,75", f"{BASE}+wise_a75_best", False),
    ("a1 ajustado", f"{BASE}+a1_y26s_best", False),
    ("a2 ajustado +HRP4K", f"{BASE}+a2_y26s_hrp4k_best", True),
]
DS = ["bisman", "brazil", "bucko", "egypt", "hrp4k"]

filas = list(csv.DictReader(open(RAIZ / "splits.csv", encoding="utf-8")))
hrp = [r["foto"] for r in filas if r["split"] == "test" and r["evaluable"] == "1" and r["dataset"] == "hrp4k"]
mitad = {f: "A" if int(hashlib.md5(f.encode()).hexdigest(), 16) % 2 == 0 else "B" for f in hrp}


def ap_hrp4k(nombre, imgsz):
    cache = RAIZ / "salidas" / "cache" / f"{nombre}_test_{imgsz}.json"
    if not cache.exists():
        return None
    preds = json.loads(cache.read_text(encoding="utf-8"))
    res = {}
    for m in ("A", "B"):
        sel = [p for p in preds if mitad.get(p[0]) == m]
        dataset_de = {p[0]: "hrp4k" for p in sel}
        gts_de = {f: ev.leer_etiquetas(RAIZ / "etiquetas" / f"{f}.txt", w, h) for f, w, h, _, _ in sel}
        reg, n_gt = ev.medir(sel, dataset_de, gts_de, {"conf_bache": 0.2, "conf_grieta": 0.2,
                                                        "conf_seg": pv.CONF_SEG, "conf_seg_solo": 1.01})
        res[m] = ev.ap50(reg[("hrp4k", "D40", "det")], n_gt[("hrp4k", "D40")])
    return res


def informe(nombre, split, imgsz):
    p = RAIZ / "informes" / f"eval_{nombre}_{split}_{imgsz}.json"
    if not p.exists():
        return None
    return {f["dataset"]: f["AP50_det"] for f in json.loads(p.read_text(encoding="utf-8"))["filas"] if f["clase"] == "D40"}


print(f"HRP4K test: {sum(v == 'A' for v in mitad.values())} fotos en la mitad A, {sum(v == 'B' for v in mitad.values())} en la B")
datos = {}
for imgsz in (1280, 640):
    print(f"\n=== {imgsz} px === AP50 de baches. val: todos; test: por dataset; HRP4K-A (elegir) y HRP4K-B (informar)")
    print(f"{'modelo':22}{'val':>7} |" + "".join(f"{d:>8}" for d in DS) + f"{'todos':>8} |{'H4K-A':>7}{'H4K-B':>7}")
    for titulo, nombre, vio in MODELOS:
        v, t, h = informe(nombre, "val", imgsz), informe(nombre, "test", imgsz), ap_hrp4k(nombre, imgsz)
        if v is None or t is None or h is None:
            print(f"{titulo:22}  (falta)")
            continue
        datos[(nombre, imgsz)] = (v["todos"], h["A"], vio)
        print(f"{titulo:22}{v['todos']:>7.3f} |" + "".join(f"{t.get(d, 0):>8.3f}" for d in DS) +
              f"{t['todos']:>8.3f} |{h['A']:>7.3f}{h['B']:>7.3f}" + ("  (vio HRP4K)" if vio else ""))

b0 = datos.get((MODELOS[1][1], 640))
if b0:
    aptos = [(v, n) for (n, s), (v, a, vio) in datos.items() if s == 640 and not vio and a >= b0[1] - 0.005]
    print(f"\nRegla (640 px): HRP4K-A del B0 = {b0[1]:.3f}; aptos (HRP4K-A >= {b0[1] - 0.005:.3f}):",
          ", ".join(f"{n.split('+')[-1]} (val {v:.3f})" for v, n in sorted(aptos, reverse=True)))
    if aptos:
        print("ELEGIDO:", max(aptos)[1])
