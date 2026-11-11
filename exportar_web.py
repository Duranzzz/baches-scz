"""Exporta los detectores a ONNX para la app web (ONNX Runtime Web) y escribe modelos.json.

    python exportar_web.py --app ../baches-scz-app          # carpeta de la app (repo aparte)
    python exportar_web.py --app RUTA --solo yolo12s_rdd2022
    python exportar_web.py --app RUTA --modelo a1=runs/a1/weights/best.pt   # sumar el modelo reentrenado

Exporta con tamaño de entrada dinámico (la app elige 640, 864 o 1280 de lado mayor según el teléfono) y
verifica, con ONNX Runtime de Python, que el ONNX da las mismas cajas que el .pt en un cuadro de prueba,
a dos tamaños. La app replica el letterbox de ultralytics: escala al lado mayor y rellena con gris 114,
centrado, hasta múltiplos de 32.
"""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import preview as pv

RAIZ = Path(__file__).resolve().parent
MODELOS = {
    # El de la app: promedio WiSE-FT (alfa 0,25) del YOLO26s descargado y el ajustado a1, con sus 4 clases
    # (entrenamiento/promediar_pesos.py). Umbrales elegidos en val, solo detector, a 640 px.
    "yolo26s_ajustado": {
        "titulo": "YOLO26s ajustado (baches y grietas)", "pesos": RAIZ / "modelos/wise4_a25/weights/best.pt",
        "uso": "vehiculo", "conf": {"bache": 0.10, "grieta": 0.25},
        "licencia": "AGPL-3.0: promedio de pesos (WiSE-FT, alfa 0,25) entre TamAko783/YOLO26s_RDD_FRDC_Distilled_v2 "
                    "y el mismo ajustado con las fotos del proyecto (ultralytics, AGPL-3.0)"},
    "rdd_yolo26s": {
        "titulo": "YOLO26s descargado (baches y grietas)", "pesos": RAIZ / "modelos/rdd_yolo26s/weights/best.pt",
        "uso": "vehiculo", "conf": {"bache": 0.15, "grieta": 0.30},
        "licencia": "AGPL-3.0, TamAko783/YOLO26s_RDD_FRDC_Distilled_v2 (entrenado con ultralytics, AGPL-3.0)"},
    "yolo12s_rdd2022": {
        "titulo": "YOLO12s RDD2022 (baches y grietas)", "pesos": RAIZ / "modelos/yolo12s_rdd2022/yolo12s_RDD2022_best.pt",
        "uso": "vehiculo", "licencia": "MIT, rezzzq/yolo12s-road-damage-rdd2022 (entrenado con ultralytics, AGPL-3.0)"},
    "parcial1_bisman": {
        "titulo": "YOLO del 1er parcial (baches de cerca)", "pesos": RAIZ / "modelos/parcial1_bisman/weights/best.pt",
        "uso": "cerca", "licencia": "modelo de Bisman Singh verificado en el 1er parcial (entrenado con ultralytics, AGPL-3.0)"},
}
NOMBRE = {"D00": "grieta_longitudinal", "D10": "grieta_transversal", "D20": "grieta_cocodrilo", "D40": "bache"}
CONF = {"bache": 0.25, "grieta": 0.40}   # si el modelo no trae los suyos; la app deja moverlos


def letterbox(img, lado):
    """Como ultralytics LetterBox(auto=True): lado mayor = `lado`, relleno 114 centrado a múltiplos de 32."""
    h, w = img.shape[:2]
    r = lado / max(h, w)
    nw, nh = round(w * r), round(h * r)
    W, H = -(-nw // 32) * 32, -(-nh // 32) * 32
    px, py = (W - nw) / 2, (H - nh) / 2
    out = np.full((H, W, 3), 114, np.uint8)
    x0, y0 = round(px - 0.1), round(py - 0.1)
    out[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, r, x0, y0


def onnx_cajas(sesion, img, lado, conf):
    """Lo mismo que hará la app: letterbox, salida [1, 4+nc, N], filtro por confianza y NMS por clase."""
    lb, r, x0, y0 = letterbox(img, lado)
    x = lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255
    out = sesion.run(None, {sesion.get_inputs()[0].name: x})[0][0]   # (4+nc, N)
    cajas, puntajes = out[:4].T, out[4:].T
    k = puntajes.argmax(1)
    s = puntajes.max(1)
    m = s >= conf
    cajas, k, s = cajas[m], k[m], s[m]
    xyxy = np.stack([cajas[:, 0] - cajas[:, 2] / 2, cajas[:, 1] - cajas[:, 3] / 2,
                     cajas[:, 0] + cajas[:, 2] / 2, cajas[:, 1] + cajas[:, 3] / 2], 1)
    xywh = np.concatenate([xyxy[:, :2], xyxy[:, 2:] - xyxy[:, :2]], 1)   # OpenCV pide x, y, ancho, alto
    idx = cv2.dnn.NMSBoxesBatched(xywh.tolist(), s.tolist(), k.tolist(), conf, 0.7) if len(s) else []
    idx = np.array(idx, dtype=int).reshape(-1)
    xyxy = (xyxy[idx] - [x0, y0, x0, y0]) / r
    return sorted(zip(k[idx].tolist(), s[idx].tolist(), xyxy.tolist()), key=lambda t: -t[1])


def verificar(pt, onnx_path, img):
    import onnxruntime as ort
    sesion = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    modelo = YOLO(str(pt))
    peor = 0.0
    for lado in (640, 864):
        ref = modelo.predict(img, imgsz=lado, conf=0.10, iou=0.7, device="cpu", verbose=False)[0]
        a = sorted(zip(ref.boxes.cls.int().tolist(), ref.boxes.conf.tolist(), ref.boxes.xyxy.tolist()),
                   key=lambda t: -t[1])
        b = onnx_cajas(sesion, img, lado, 0.10)
        if len(a) != len(b):
            raise SystemExit(f"{onnx_path.name} a {lado}: {len(a)} cajas con .pt y {len(b)} con ONNX")
        for (ka, ca, xa), (kb, cb, xb) in zip(a, b):
            if ka != kb or abs(ca - cb) > 1e-3:
                raise SystemExit(f"{onnx_path.name} a {lado}: clases o confianzas distintas")
            peor = max(peor, float(np.abs(np.array(xa) - np.array(xb)).max()))
        print(f"  {lado}: {len(a)} cajas iguales (diferencia máxima {peor:.2f} px)")
    return peor


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--app", type=Path, required=True, help="carpeta de la app; escribe en <app>/modelos/")
    ap.add_argument("--solo", nargs="*", help="exportar solo estos")
    ap.add_argument("--modelo", action="append", default=[], help="clave=pesos.pt[:uso] (uso: vehiculo o cerca)")
    ap.add_argument("--prueba", type=Path, default=RAIZ / "videos" / "scz_01_lateral_auto_hormigon.mp4",
                    help="video o imagen para verificar el ONNX")
    args = ap.parse_args()

    modelos = dict(MODELOS)
    for m in args.modelo:
        k, resto = m.split("=", 1)
        ruta, _, uso = resto.partition(":")
        modelos[k] = {"titulo": k, "pesos": Path(ruta), "uso": uso or "vehiculo", "licencia": "propio"}
    if args.solo:
        modelos = {k: v for k, v in modelos.items() if k in args.solo}

    if args.prueba.suffix.lower() in (".mp4", ".mov", ".avi"):
        cap = cv2.VideoCapture(str(args.prueba))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 120)
        ok, img = cap.read()
        cap.release()
    else:
        img = cv2.imdecode(np.fromfile(args.prueba, dtype=np.uint8), cv2.IMREAD_COLOR)

    destino = args.app / "modelos"
    destino.mkdir(parents=True, exist_ok=True)
    indice_ruta = destino / "modelos.json"
    indice = json.loads(indice_ruta.read_text(encoding="utf-8")) if indice_ruta.exists() else {"modelos": []}
    for clave, cfg in modelos.items():
        print(f"{clave}: exportando {cfg['pesos']}")
        with tempfile.TemporaryDirectory() as tmp:
            pt = Path(tmp) / f"{clave}.pt"
            shutil.copy2(cfg["pesos"], pt)
            m = YOLO(str(pt))
            onnx = Path(m.export(format="onnx", imgsz=640, dynamic=True, simplify=True, verbose=False))
            verificar(cfg["pesos"], onnx, img)
            shutil.copy2(onnx, destino / f"{clave}.onnx")
            nombres = {int(k): v for k, v in m.names.items()}
        clases = {}
        for k, nombre in nombres.items():
            c = pv.codigo_det(nombre)
            clases[str(k)] = NOMBRE[c] if c else None
        entrada = {"id": clave, "titulo": cfg["titulo"], "archivo": f"{clave}.onnx", "uso": cfg["uso"],
                   "clases": clases, "nombres_originales": {str(k): v for k, v in nombres.items()},
                   "conf": cfg.get("conf", CONF), "mb": round((destino / f"{clave}.onnx").stat().st_size / 1e6, 1),
                   "licencia": cfg["licencia"]}
        indice["modelos"] = [e for e in indice["modelos"] if e["id"] != clave] + [entrada]
        print(f"  -> {destino / entrada['archivo']} ({entrada['mb']} MB), clases {clases}")
    indice_ruta.write_text(json.dumps(indice, indent=1, ensure_ascii=False), encoding="utf-8")
    print(indice_ruta)


if __name__ == "__main__":
    main()
