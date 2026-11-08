"""Preview: detección de daños (YOLO12s, RDD2022) + segmentación de baches (YOLOv8n-seg).

Uso:
    python preview.py                       # todas las imágenes de muestras/
    python preview.py --source foto.jpg     # una imagen
    python preview.py --source carpeta/     # todas las imágenes de una carpeta (sin subcarpetas)
    python preview.py --det-baches pesos/baches.pt --imgsz 1280   # con un detector de baches reentrenado

Si faltan los modelos, los descarga en modelos/. Guarda en salidas/ la imagen anotada y un JSON por imagen.
Con --det-baches, los baches salen de ese modelo y las grietas del detector general.
"""
import argparse
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# --- Umbrales ---
# CONF_BACHE y CONF_SEG_SOLO salen de `evaluar.py --split val --barrer` (2026-10-28, modelos públicos, 640 px):
# máximo F1 medio de baches entre Bisman, Brasil, Bucko y Egipto (0,320 contra 0,314 con 0,25/0,25).
# La superficie es casi plana; lo que la mueve es apagar los baches de solo segmentación (F1 0,20).
# Recalibrar con el mismo comando cada vez que cambie un modelo.
CONF_BACHE = 0.20          # confianza mínima de una caja D40
CONF_GRIETA = 0.20         # confianza mínima de una caja D00, D10 o D20
CONF_SEG = 0.25            # confianza mínima de una máscara para darle su área a una caja D40
CONF_SEG_SOLO = 0.55       # confianza mínima de un bache que solo ve la segmentación (sin caja D40)
IMGSZ = 640                # los dos modelos se entrenaron a 640
IOU_DET_SEG = 0.3          # caja D40 vs caja de la máscara: por encima, el área sale de la máscara
BACHE_ALTA = 0.05          # área del bache / área de la imagen
BACHE_MEDIA = 0.015
PESO_GRIETA = {"D00": 1, "D10": 1, "D20": 2}
GRIETA_ALTA = 3.0          # puntaje = peso * (1 + área_caja * 10) * confianza
GRIETA_MEDIA = 1.5

# --- Modelos ---
RAIZ = Path(__file__).resolve().parent
DET_REPO = "rezzzq/yolo12s-road-damage-rdd2022"
DET_PT = RAIZ / "modelos" / "yolo12s_rdd2022" / "yolo12s_RDD2022_best.pt"
SEG_URL = ("https://github.com/FarzadNekouee/YOLOv8_Pothole_Segmentation_Road_Damage_Assessment"
           "/raw/master/model/best.pt")
SEG_PT = RAIZ / "modelos" / "farzad_pothole_seg" / "best.pt"

DESCRIPCION = {"D00": "longitudinal", "D10": "transversal", "D20": "cocodrilo", "D40": "bache"}
COLOR = {"baja": (0, 190, 0), "media": (0, 215, 255), "alta": (0, 0, 230)}  # BGR
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def descargar_modelos():
    if not DET_PT.exists():
        from huggingface_hub import hf_hub_download
        print(f"Descargando {DET_REPO} ...")
        hf_hub_download(DET_REPO, DET_PT.name, local_dir=DET_PT.parent)
    if not SEG_PT.exists():
        print("Descargando segmentación de FarzadNekouee ...")
        SEG_PT.parent.mkdir(parents=True, exist_ok=True)
        parcial = SEG_PT.with_suffix(".part")
        urllib.request.urlretrieve(SEG_URL, parcial)
        parcial.replace(SEG_PT)
    for p in (DET_PT, SEG_PT):
        mb = p.stat().st_size / 1e6
        if mb < 1:
            sys.exit(f"{p} pesa {mb:.2f} MB: no es un modelo (¿puntero de Git LFS?). Bórralo y vuelve a correr.")


NOMBRES_BACHE = {"POTHOLE", "POTHOLES", "BACHE", "BACHES"}


def codigo_det(nombre):
    """'D40', 'D40 pothole', 'd40_pothole', 'pothole', 'bache' -> 'D40'; 'Repair' u otra cosa -> None.

    Los detectores de una sola clase "pothole" o "bache" (el YOLO del 1er parcial, el de PeterHdd)
    cuentan como D40; sin esto, --det-baches los ignoraba sin avisar."""
    partes = nombre.replace("_", " ").split()
    if not partes:
        return None
    codigo = partes[0].upper()
    if codigo in DESCRIPCION:
        return codigo
    return "D40" if codigo in NOMBRES_BACHE else None


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def severidad_bache(area_rel):
    if area_rel > BACHE_ALTA:
        return "alta"
    return "media" if area_rel > BACHE_MEDIA else "baja"


def severidad_grieta(codigo, area_rel, conf):
    puntaje = PESO_GRIETA[codigo] * (1 + area_rel * 10) * conf
    if puntaje >= GRIETA_ALTA:
        return "alta"
    return "media" if puntaje >= GRIETA_MEDIA else "baja"


def area_caja(caja, area_img):
    return (caja[2] - caja[0]) * (caja[3] - caja[1]) / area_img


def cajas(modelo, img, conf, imgsz, device):
    """[(código, conf, caja)] de un detector, sin las clases que no son daño (Repair)."""
    r = modelo.predict(img, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
    out = [(codigo_det(modelo.names[k]), c, caja) for caja, c, k in
           zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.int().tolist())]
    return [d for d in out if d[0] is not None]


def predecir(img, det, seg, device, imgsz=IMGSZ, conf=None, mascaras=True, det_baches=None):
    """Corre los modelos sobre una imagen BGR, sin combinar.

    Devuelve (dets, segs): dets = [(código, conf, caja)] sin Repair;
    segs = [{"conf", "caja", "mask", "area_rel"}], con mask y area_rel en None si mascaras=False.
    Con det_baches, las cajas D40 salen de ese modelo y no del detector general.
    conf=None usa los umbrales de arriba; evaluar.py pasa uno bajo y filtra después con combinar().
    """
    area_img = float(img.shape[0] * img.shape[1])
    c_det = min(CONF_BACHE, CONF_GRIETA) if conf is None else conf
    c_seg = min(CONF_SEG, CONF_SEG_SOLO) if conf is None else conf
    dets = cajas(det, img, c_det, imgsz, device)
    if det_baches is not None:
        dets = [d for d in dets if d[0] != "D40"] + [d for d in cajas(det_baches, img, c_det, imgsz, device)
                                                       if d[0] == "D40"]
    # retina_masks: máscaras al tamaño original, así el área no cuenta el relleno del letterbox
    rs = seg.predict(img, conf=c_seg, imgsz=imgsz, device=device, retina_masks=mascaras, verbose=False)[0]
    segs = []
    if rs.masks is not None:
        for m, caja, c in zip(rs.masks.data, rs.boxes.xyxy.tolist(), rs.boxes.conf.tolist()):
            mb = m.cpu().numpy() > 0.5 if mascaras else None
            segs.append({"conf": c, "caja": caja, "mask": mb,
                         "area_rel": None if mb is None else mb.sum() / area_img})
    return dets, segs


def combinar(dets, segs, area_img, conf_bache=CONF_BACHE, conf_grieta=CONF_GRIETA, conf_seg=CONF_SEG,
             conf_seg_solo=CONF_SEG_SOLO):
    """Une detección y segmentación en una lista de daños con severidad.

    Devuelve (danos, mascaras), con mascaras = [(máscara booleana, daño al que pertenece)] para dibujar.
    """
    danos, baches = [], []
    for codigo, conf, caja in dets:
        if conf < (conf_bache if codigo == "D40" else conf_grieta):
            continue
        a = area_caja(caja, area_img)
        d = {"tipo": codigo, "severidad": None, "confianza": round(conf, 3),
             "caja": [round(v, 1) for v in caja], "area_rel": a, "fuente": "det"}
        if codigo == "D40":
            baches.append(d)
        else:
            d["severidad"] = severidad_grieta(codigo, a, conf)
        danos.append(d)
    segs = [s for s in segs if s["conf"] >= conf_seg]

    # Emparejar cajas D40 con máscaras, de mayor a menor IoU, cada una a lo sumo una vez
    pares = sorted(((iou(b["caja"], s["caja"]), i, j) for i, b in enumerate(baches)
                    for j, s in enumerate(segs)), reverse=True)
    usados_b, dueno = set(), {}
    for valor, i, j in pares:
        if valor <= IOU_DET_SEG or i in usados_b or j in dueno:
            continue
        usados_b.add(i)
        dueno[j] = baches[i]
        if segs[j]["area_rel"] is not None:
            baches[i]["area_rel"] = segs[j]["area_rel"]
        baches[i]["fuente"] = "det+seg"
    # Máscaras sin caja D40: baches que solo vio la segmentación. Fuera de su dominio (fotos de cerca)
    # da falsos positivos (fachadas, tierra), por eso piden más confianza.
    for j, s in enumerate(segs):
        if j in dueno or s["conf"] < conf_seg_solo:
            continue
        a = s["area_rel"] if s["area_rel"] is not None else area_caja(s["caja"], area_img)
        d = {"tipo": "D40", "severidad": None, "confianza": round(s["conf"], 3),
             "caja": [round(v, 1) for v in s["caja"]], "area_rel": a, "fuente": "seg"}
        baches.append(d)
        danos.append(d)
        dueno[j] = d
    for b in baches:
        b["severidad"] = severidad_bache(b["area_rel"])
    for d in danos:
        d["area_rel"] = round(float(d["area_rel"]), 5)
    return danos, [(segs[j]["mask"], d) for j, d in dueno.items() if segs[j]["mask"] is not None]


def analizar(img, det, seg, device, imgsz=IMGSZ, det_baches=None):
    dets, segs = predecir(img, det, seg, device, imgsz, det_baches=det_baches)
    return combinar(dets, segs, float(img.shape[0] * img.shape[1]))


def dibujar(img, danos, mascaras):
    capa = img.copy()
    for mb, d in mascaras:
        capa[mb] = COLOR[d["severidad"]]
    out = cv2.addWeighted(capa, 0.45, img, 0.55, 0)
    grosor = max(2, round(min(img.shape[:2]) / 400))
    escala = grosor * 0.35
    for d in danos:
        x1, y1, x2, y2 = (int(v) for v in d["caja"])
        color = COLOR[d["severidad"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, grosor)
        texto = f'{d["tipo"]} {DESCRIPCION[d["tipo"]]} | {d["severidad"]} | {d["confianza"]:.2f}'
        if d["fuente"] == "seg":
            texto += " (seg)"
        (tw, th), base = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, max(1, grosor // 2))
        ty = y1 - 4 if y1 - th - base - 4 > 0 else y1 + th + base + 4
        tx = max(0, min(x1, img.shape[1] - tw - 4))  # que la etiqueta no se salga por la derecha
        cv2.rectangle(out, (tx, ty - th - base), (tx + tw + 4, ty + base // 2), color, -1)
        cv2.putText(out, texto, (tx + 2, ty - base // 2), cv2.FONT_HERSHEY_SIMPLEX, escala,
                    (0, 0, 0), max(1, grosor // 2), cv2.LINE_AA)
    return out


def listar_imagenes(source):
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted(p for p in source.iterdir() if p.suffix.lower() in EXTENSIONES)
    sys.exit(f"No existe: {source}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=RAIZ / "muestras", help="imagen o carpeta")
    ap.add_argument("--salidas", type=Path, default=RAIZ / "salidas")
    ap.add_argument("--imgsz", type=int, default=IMGSZ,
                    help="lado al que se reescala la imagen; 1280 ayuda con baches lejanos en dashcam")
    ap.add_argument("--det", type=Path, default=DET_PT, help="pesos del detector general (grietas y baches)")
    ap.add_argument("--det-baches", type=Path, help="pesos de un detector solo de baches (p. ej. el reentrenado)")
    args = ap.parse_args()

    imagenes = listar_imagenes(args.source)
    if not imagenes:
        sys.exit(f"No hay imágenes en {args.source}")
    descargar_modelos()
    device = 0 if torch.cuda.is_available() else "cpu"
    det, seg = YOLO(str(args.det)), YOLO(str(SEG_PT))
    det_baches = YOLO(str(args.det_baches)) if args.det_baches else None
    print(f"dispositivo: {'GPU' if device == 0 else 'CPU'}")
    print(f"detección    {args.det.name}: {det.names}")
    if det_baches is not None:
        print(f"baches       {args.det_baches.name}: {det_baches.names}")
    print(f"segmentación {SEG_PT.name}: {seg.names}\n")
    args.salidas.mkdir(parents=True, exist_ok=True)

    total = Counter()
    for p in imagenes:
        # imdecode en vez de imread: imread falla con tildes o ñ en la ruta en Windows
        img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"{p.name}: no se pudo leer, se salta")
            continue
        danos, mascaras = analizar(img, det, seg, device, args.imgsz, det_baches)
        cv2.imencode(".jpg", dibujar(img, danos, mascaras))[1].tofile(args.salidas / f"{p.stem}.jpg")
        (args.salidas / f"{p.stem}.json").write_text(json.dumps(danos, indent=2, ensure_ascii=False),
                                                       encoding="utf-8")
        print(f"{p.name}: {len(danos)} daño(s)")
        for d in sorted(danos, key=lambda d: -d["confianza"]):
            print(f'   {d["tipo"]} {DESCRIPCION[d["tipo"]]:<12} {d["severidad"]:<5}  conf {d["confianza"]:.2f}'
                  f'  área {d["area_rel"]:.2%}  [{d["fuente"]}]')
            total[(d["tipo"], d["severidad"])] += 1

    print(f"\nResumen ({len(imagenes)} imágenes, salidas en {args.salidas}):")
    for codigo in DESCRIPCION:
        fila = {s: total[(codigo, s)] for s in COLOR}
        if any(fila.values()):
            print(f"   {codigo} {DESCRIPCION[codigo]:<12}  baja {fila['baja']:>3}  media {fila['media']:>3}"
                  f"  alta {fila['alta']:>3}")


if __name__ == "__main__":
    main()
