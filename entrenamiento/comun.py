"""Funciones compartidas por los scripts de entrenamiento/."""
import os
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
LADO = 1280          # lado mayor máximo de las fotos que se guardan (= imgsz de entrenamiento)
CALIDAD_JPG = 95


def raiz(arg):
    """Raíz del proyecto: la de este repo, o la que se pase (para probar en una copia)."""
    return Path(arg).resolve() if arg else RAIZ


def procesos():
    return max(1, (os.cpu_count() or 2) - 1)


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


def guardar_foto(datos, destino, lado=LADO):
    """Escribe un JPEG reducido a `lado` si hace falta. Devuelve (ancho, alto) originales."""
    salida, w, h = reducir(datos, lado, destino.name)
    destino.write_bytes(salida)
    return w, h


def linea_yolo(clase, x1, y1, x2, y2, w, h):
    """Caja en píxeles -> línea YOLO normalizada, recortada a la imagen. None si queda degenerada."""
    x1, x2 = max(0.0, min(x1, x2)), min(float(w), max(x1, x2))
    y1, y2 = max(0.0, min(y1, y2)), min(float(h), max(y1, y2))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return (f"{clase} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
            f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
