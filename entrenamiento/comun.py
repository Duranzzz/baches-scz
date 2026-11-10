"""Funciones compartidas por los scripts de entrenamiento/."""
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
LADO = 1280          # lado mayor máximo de las fotos que se guardan (= imgsz de entrenamiento)
CALIDAD_JPG = 95


def raiz(arg):
    """Raíz del proyecto: la de este repo, o la que se pase (para probar en una copia)."""
    return Path(arg).resolve() if arg else RAIZ


def reducir(datos, lado=LADO, nombre="?"):
    """JPEG en bytes -> (bytes, ancho, alto originales). Si el lado mayor pasa de `lado`, lo reduce;
    si no, devuelve los mismos bytes (sin volver a comprimir)."""
    img = cv2.imdecode(np.frombuffer(datos, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"no se pudo decodificar {nombre}")
    h, w = img.shape[:2]
    if max(h, w) <= lado:
        return datos, w, h
    r = lado / max(h, w)
    img = cv2.resize(img, (round(w * r), round(h * r)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, CALIDAD_JPG])
    if not ok:
        raise ValueError(f"no se pudo codificar {nombre}")
    return buf.tobytes(), w, h
