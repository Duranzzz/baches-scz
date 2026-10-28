"""Baja el RDD2022 oficial (Figshare, 13,3 GB) y HRP4K (Zenodo, 8,7 GB), retomando si se corta y
verificando el MD5 que publica cada repositorio.

    python entrenamiento/descargar_datos.py --destino /datos

Correrlo en la máquina de GPU: allá la red es mucho más rápida que desde una casa.
"""
import argparse
import hashlib
import sys
import time
from pathlib import Path

import requests

FIGSHARE = "https://api.figshare.com/v2/articles/21431547"            # RDD2022 (CRDDC'2022)
FIGSHARE_ARCHIVO = "RDD2022_released_through_CRDDC2022.zip"
ZENODO = "https://zenodo.org/api/records/17522874"                     # HRP4K V1.00
ZENODO_ARCHIVO = "HRP4K.zip"


def md5(ruta):
    h = hashlib.md5()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 24), b""):
            h.update(bloque)
    return h.hexdigest()


def descargar(url, destino, tamano, md5_esperado, intentos=20):
    if destino.exists() and destino.stat().st_size == tamano:
        print(f"{destino.name}: ya está, verificando MD5 ...", flush=True)
    else:
        for intento in range(intentos):
            hecho = destino.stat().st_size if destino.exists() else 0
            if hecho >= tamano:
                break
            try:
                with requests.get(url, headers={"Range": f"bytes={hecho}-"}, stream=True, timeout=60) as r:
                    if r.status_code not in (200, 206):
                        raise IOError(f"HTTP {r.status_code}")
                    if r.status_code == 200 and hecho:   # el servidor ignoró el rango: empezar de cero
                        hecho = 0
                    t0, n = time.time(), 0
                    with open(destino, "ab" if hecho else "wb") as f:
                        for bloque in r.iter_content(1 << 22):
                            f.write(bloque)
                            n += len(bloque)
                            if n % (1 << 30) < (1 << 22):
                                print(f"  {destino.name}: {(hecho + n) / 1e9:.1f}/{tamano / 1e9:.1f} GB "
                                      f"({n / 1e6 / (time.time() - t0):.0f} MB/s)", flush=True)
            except (requests.RequestException, IOError) as e:
                print(f"  corte ({e}); reintento {intento + 1}/{intentos}", flush=True)
                time.sleep(min(60, 5 * (intento + 1)))
        if destino.stat().st_size != tamano:
            sys.exit(f"{destino.name}: tamaño {destino.stat().st_size} != {tamano}")
    obtenido = md5(destino)
    if obtenido != md5_esperado:
        sys.exit(f"{destino.name}: MD5 {obtenido} != {md5_esperado}. Borrarlo y volver a bajar.")
    print(f"{destino.name}: OK ({tamano / 1e9:.1f} GB, MD5 verificado)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--destino", type=Path, required=True)
    ap.add_argument("--solo", choices=["rdd2022", "hrp4k"])
    args = ap.parse_args()
    args.destino.mkdir(parents=True, exist_ok=True)

    if args.solo in (None, "rdd2022"):
        info = requests.get(FIGSHARE, timeout=60).json()
        f = next(x for x in info["files"] if x["name"] == FIGSHARE_ARCHIVO)
        descargar(f["download_url"], args.destino / f["name"], f["size"], f["computed_md5"])
    if args.solo in (None, "hrp4k"):
        info = requests.get(ZENODO, timeout=60).json()
        f = next(x for x in info["files"] if x["key"] == ZENODO_ARCHIVO)
        descargar(f["links"]["self"], args.destino / f["key"], f["size"], f["checksum"].split(":", 1)[1])


if __name__ == "__main__":
    main()
