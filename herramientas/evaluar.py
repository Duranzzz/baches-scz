"""Mide detector, segmentación y la combinación de preview.py contra las etiquetas, por dataset y clase.

    python herramientas/evaluar.py --split test                         # línea base sobre el test congelado
    python herramientas/evaluar.py --split val --barrer                 # además busca umbrales (solo en val)
    python herramientas/evaluar.py --split test --det pesos/best.pt     # un detector reentrenado, mismo test
    python herramientas/evaluar.py --split test --imgsz 1280 --det-baches runs/a1/weights/best.pt   # + detector de baches

Lee splits.csv (solo fotos evaluables) y etiquetas/. Un acierto es una caja con IoU >= 0,5 con una
etiqueta de la misma clase. Cada dataset se mide solo en las clases que tiene completas: en Bucko solo
hay baches marcados, así que una grieta detectada ahí no cuenta ni a favor ni en contra.
"grieta" junta D00, D10 y D20, para comparar con Brasil y Egipto, que no distinguen el tipo.
Las predicciones se guardan en salidas/cache/ para no volver a correr los modelos al barrer umbrales.
"""
import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import preview as pv

GRIETAS = ("D00", "D10", "D20")
CLASES_EVAL = {
    # Sin D40: en la redistribución de dronefreak la clase "pothole" también marca sendas y líneas
    # borrosas, tapas y parches (otros códigos del RDD2022 original). Revisado a ojo el 2026-10-28.
    "rdd2022": ["D00", "D10", "D20", "grieta"],
    "rddo": ["D00", "D10", "D20", "D40", "grieta"],  # RDD2022 oficial (sekilab): D40 es solo bache
    "brazil": ["D40", "grieta"],
    "egypt": ["D40", "grieta"],
    "bucko": ["D40"],
    "hrp4k": ["D40"],
    "bisman": ["D40"],
}
# Esquema de etiquetas/: 0 D00, 1 D10, 2 D20, 3 D40, 4 grieta sin tipo, 5 tapa
ETIQUETA_CLASE = {"D00": {0}, "D10": {1}, "D20": {2}, "D40": {3}, "grieta": {0, 1, 2, 4}}
IOU_ACIERTO = 0.5


def leer_etiquetas(p, w, h):
    cajas = []
    for linea in p.read_text(encoding="utf-8").splitlines():
        c = linea.split()
        if len(c) >= 5:
            k, cx, cy, bw, bh = int(c[0]), *(float(v) for v in c[1:5])
            cajas.append((k, [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]))
    return cajas


def predicciones(fotos, det_pt, imgsz, cache, det_baches_pt=None):
    """[(foto, w, h, dets, segs)] con confianza baja; segs sin máscara (solo cajas).

    La caché se guarda cada 200 fotos: si se corta, la próxima corrida sigue donde quedó.
    """
    out = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else []
    faltan = [f for f in fotos if f not in {p[0] for p in out}]
    if not faltan:
        print(f"usando {cache.relative_to(pv.RAIZ)}")
        return out
    if out:
        print(f"retomando {cache.relative_to(pv.RAIZ)}: faltan {len(faltan)} de {len(fotos)}")
    device = 0 if torch.cuda.is_available() else "cpu"
    det, seg = YOLO(str(det_pt)), YOLO(str(pv.SEG_PT))
    det_baches = YOLO(str(det_baches_pt)) if det_baches_pt else None
    cache.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, f in enumerate(faltan):
        img = cv2.imdecode(np.fromfile(pv.RAIZ / "fotos" / f"{f}.jpg", dtype=np.uint8), cv2.IMREAD_COLOR)
        dets, segs = pv.predecir(img, det, seg, device, imgsz, conf=0.001, mascaras=False, det_baches=det_baches)
        out.append([f, img.shape[1], img.shape[0], dets, [[s["conf"], s["caja"]] for s in segs]])
        if (i + 1) % 200 == 0 or i + 1 == len(faltan):
            cache.write_text(json.dumps(out), encoding="utf-8")
            print(f"  {i + 1}/{len(faltan)}  {time.time() - t0:.0f} s", flush=True)
    return out


def emparejar(preds, gts):
    """preds [(conf, caja)], gts [caja] -> [(conf, acierto)], voraz por confianza."""
    libres, res = set(range(len(gts))), []
    for conf, caja in sorted(preds, key=lambda x: -x[0]):
        mejor, j = max(((pv.iou(caja, gts[j]), j) for j in libres), default=(0.0, None))
        if mejor >= IOU_ACIERTO:
            libres.discard(j)
        res.append((conf, mejor >= IOU_ACIERTO))
    return res


def ap50(registros, n_gt):
    """AP con 101 puntos de recall (como COCO)."""
    if not registros or not n_gt:
        return 0.0
    r = sorted(registros, key=lambda x: -x[0])
    tp = np.cumsum([a for _, a in r])
    fp = np.cumsum([not a for _, a in r])
    rec, prec = tp / n_gt, tp / (tp + fp)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    return float(np.mean([prec[rec >= t].max() if (rec >= t).any() else 0.0 for t in np.linspace(0, 1, 101)]))


def prf(registros, n_gt):
    tp = sum(a for _, a in registros)
    p = tp / len(registros) if registros else 0.0
    r = tp / n_gt if n_gt else 0.0
    return {"P": p, "R": r, "F1": 2 * p * r / (p + r) if p + r else 0.0, "tp": tp,
            "fp": len(registros) - tp, "fn": n_gt - tp}


def de_clase(items, clase):
    """items [(código, conf, caja)] -> [(conf, caja)] de la clase pedida."""
    ok = set(GRIETAS) if clase == "grieta" else {clase}
    return [(c, caja) for codigo, c, caja in items if codigo in ok]


def medir(preds, dataset_de, gts_de, umbrales, solo_preview=False):
    """Registros por (dataset, clase, fuente). fuente: det y seg (para AP) y preview (en el punto de operación)."""
    reg = defaultdict(list)
    n_gt = defaultdict(int)
    for f, w, h, dets, segs in preds:
        ds = dataset_de[f]
        gts = gts_de[f]
        danos, _ = pv.combinar(dets, [{"conf": c, "caja": b, "mask": None, "area_rel": None} for c, b in segs],
                               float(w * h), **umbrales)
        salida = [(d["tipo"], d["confianza"], d["caja"]) for d in danos]
        for clase in CLASES_EVAL[ds]:
            g = [caja for k, caja in gts if k in ETIQUETA_CLASE[clase]]
            n_gt[(ds, clase)] += len(g)
            p = de_clase(salida, clase)
            reg[(ds, clase, "preview")] += emparejar(p, g)
            # Por foto: ¿la marcó si tiene el daño, y no la marcó si no lo tiene? No depende de cómo se
            # cortó cada grieta en cajas, y es lo que importa para elegir cuadros de un video.
            reg[(ds, clase, "foto")].append((bool(g), bool(p)))
            if solo_preview:
                continue
            reg[(ds, clase, "det")] += emparejar(de_clase(dets, clase), g)
            if clase == "D40":
                reg[(ds, clase, "seg")] += emparejar([(c, b) for c, b in segs], g)
    return reg, n_gt


def tabla(reg, n_gt, conf_bache, conf_grieta):
    filas = []
    claves = sorted({(ds, c) for ds, c, _ in reg})
    # "todos": junta los datasets que miden esa clase
    for clase in ("D00", "D10", "D20", "D40", "grieta"):
        dss = [ds for ds, c in claves if c == clase]
        if len(dss) > 1:
            for fuente in ("det", "seg", "preview", "foto"):
                reg[("todos", clase, fuente)] = sum((reg[(ds, clase, fuente)] for ds in dss), [])
            n_gt[("todos", clase)] = sum(n_gt[(ds, clase)] for ds in dss)
            claves.append(("todos", clase))
    for ds, clase in claves:
        fila = {"dataset": ds, "clase": clase, "etiquetas": n_gt[(ds, clase)],
                "AP50_det": ap50(reg[(ds, clase, "det")], n_gt[(ds, clase)]),
                "AP50_seg": ap50(reg[(ds, clase, "seg")], n_gt[(ds, clase)]) if clase == "D40" else None,
                "det": prf([x for x in reg[(ds, clase, "det")]
                            if x[0] >= (conf_bache if clase == "D40" else conf_grieta)], n_gt[(ds, clase)]),
                "preview": prf(reg[(ds, clase, "preview")], n_gt[(ds, clase)]),
                "foto": por_foto(reg[(ds, clase, "foto")])}
        filas.append(fila)
    return filas


def por_foto(pares):
    tp = sum(g and p for g, p in pares)
    marcadas, con_dano = sum(p for _, p in pares), sum(g for g, _ in pares)
    return {"P": tp / marcadas if marcadas else 0.0, "R": tp / con_dano if con_dano else 0.0,
            "fotos": len(pares), "con_dano": con_dano}


def imprimir(filas):
    print(f"\n{'dataset':<8} {'clase':<7} {'etiq':>5}  {'AP50det':>7} {'AP50seg':>7}   "
          f"{'det P':>5} {'det R':>5}   {'prev P':>6} {'prev R':>6} {'prev F1':>7}   {'foto P':>6} {'foto R':>6}")
    for f in filas:
        seg = f"{f['AP50_seg']:.3f}" if f["AP50_seg"] is not None else "-"
        print(f"{f['dataset']:<8} {f['clase']:<7} {f['etiquetas']:>5}  {f['AP50_det']:>7.3f} {seg:>7}   "
              f"{f['det']['P']:>5.2f} {f['det']['R']:>5.2f}   {f['preview']['P']:>6.2f} {f['preview']['R']:>6.2f}"
              f" {f['preview']['F1']:>7.3f}   {f['foto']['P']:>6.2f} {f['foto']['R']:>6.2f}")


def barrer(preds, dataset_de, gts_de):
    """Busca en val los umbrales de preview.py que maximizan el F1 medio entre datasets (cada dataset pesa
    igual, para que Bisman, con fotos de cerca, no tape a los de cámara frontal)."""
    def f1_medio(clase, umbrales):
        reg, n_gt = medir(preds, dataset_de, gts_de, umbrales, solo_preview=True)
        f1s = {ds: prf(reg[(ds, c, "preview")], n_gt[(ds, c)])["F1"] for ds, c in n_gt if c == clase and n_gt[(ds, c)]}
        return float(np.mean(list(f1s.values()))), f1s

    res = {}
    print("\nBarrido en val (F1 medio entre datasets):")
    mejor = None
    for cd in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
        for css in (0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 1.01):
            m, f1s = f1_medio("D40", {"conf_bache": cd, "conf_seg_solo": css})
            res[f"D40 conf_bache={cd} conf_seg_solo={css}"] = {"F1_medio": m, **f1s}
            if mejor is None or m > mejor[0]:
                mejor = (m, cd, css, f1s)
    print(f"  baches: mejor CONF_BACHE={mejor[1]} CONF_SEG_SOLO={mejor[2]} (1.01 = nunca) -> F1 medio {mejor[0]:.3f} "
          + " ".join(f"{k} {v:.2f}" for k, v in mejor[3].items()))
    mejor_g = None
    for cd in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
        m, f1s = f1_medio("grieta", {"conf_grieta": cd})
        res[f"grieta conf_grieta={cd}"] = {"F1_medio": m, **f1s}
        if mejor_g is None or m > mejor_g[0]:
            mejor_g = (m, cd, f1s)
    print(f"  grietas: mejor CONF_GRIETA={mejor_g[1]} -> F1 medio {mejor_g[0]:.3f} "
          + " ".join(f"{k} {v:.2f}" for k, v in mejor_g[2].items()))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--det", type=Path, default=pv.DET_PT)
    ap.add_argument("--det-baches", type=Path, help="detector solo de baches; reemplaza las D40 de --det")
    ap.add_argument("--imgsz", type=int, default=pv.IMGSZ)
    ap.add_argument("--datasets", help="lista separada por comas; por defecto, todos")
    ap.add_argument("--barrer", action="store_true", help="buscar umbrales (usar solo con --split val)")
    ap.add_argument("--conf-bache", type=float, default=pv.CONF_BACHE, help="umbral elegido en val con --barrer")
    ap.add_argument("--conf-grieta", type=float, default=pv.CONF_GRIETA)
    ap.add_argument("--conf-seg-solo", type=float, default=pv.CONF_SEG_SOLO)
    args = ap.parse_args()

    with open(pv.RAIZ / "splits.csv", encoding="utf-8") as f:
        filas = [r for r in csv.DictReader(f) if r["split"] == args.split and r["evaluable"] == "1"]
    if args.datasets:
        elegidos = set(args.datasets.split(","))
        filas = [r for r in filas if r["dataset"] in elegidos]
    fotos = [r["foto"] for r in filas]
    dataset_de = {r["foto"]: r["dataset"] for r in filas}
    detectores = args.det.stem + (f"+{args.det_baches.parent.parent.name}_{args.det_baches.stem}"
                                  if args.det_baches else "")
    nombre = f"{detectores}_{args.split}_{args.imgsz}"
    print(f"{len(fotos)} fotos de {args.split}, detectores {detectores}, imgsz {args.imgsz}")
    todas = predicciones(sorted(fotos), args.det, args.imgsz, pv.RAIZ / "salidas" / "cache" / f"{nombre}.json",
                         args.det_baches)
    preds = [p for p in todas if p[0] in dataset_de]
    gts_de = {f: leer_etiquetas(pv.RAIZ / "etiquetas" / f"{f}.txt", w, h) for f, w, h, _, _ in preds}

    umbrales = {"conf_bache": args.conf_bache, "conf_grieta": args.conf_grieta, "conf_seg": pv.CONF_SEG,
                "conf_seg_solo": args.conf_seg_solo}
    reg, n_gt = medir(preds, dataset_de, gts_de, umbrales)
    resultado = {"split": args.split, "detectores": detectores, "imgsz": args.imgsz, "fotos": len(preds),
                 "umbrales": umbrales, "filas": tabla(reg, n_gt, args.conf_bache, args.conf_grieta)}
    imprimir(resultado["filas"])
    if args.barrer:
        resultado["barrido"] = barrer(preds, dataset_de, gts_de)
    destino = pv.RAIZ / "informes" / f"eval_{nombre}.json"
    destino.parent.mkdir(exist_ok=True)
    destino.write_text(json.dumps(resultado, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nguardado en {destino.relative_to(pv.RAIZ)}")


if __name__ == "__main__":
    main()
