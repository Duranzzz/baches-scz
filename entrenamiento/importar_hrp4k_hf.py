"""Suma al TRAIN el train y el valid oficiales de HRP4K, desde su copia en Hugging Face (dronefreak/HRP4K,
CC BY 4.0, la misma versión V1.00 de Zenodo). El test de HRP4K ya está en el paquete y sigue siendo test.

    python entrenamiento/importar_hrp4k_hf.py --hf /datos/_hrp4k_hf --salida /datos/ds_baches_v2

- Baja solo images/ y labels/ de train y valid (~6,4 GB) y guarda cada foto reducida a 1280 px de lado mayor
  como hrp4k_{train,valid}_<n>.jpg, con sus cajas como clase 3 (D40, bache) del esquema de etiquetas/.
- Una foto del test de HRP4K que sea casi copia (dHash <= 25, el criterio de preparar_splits.py) de una foto
  nueva deja de evaluarse, para todos los modelos por igual. El 22-nov-2026 fue 1 de 900.
- splits.csv del repo ya tiene estas filas: si ya están, no las duplica.
- Al final arma el dataset de ultralytics en --salida, con la clase llamada "Pothole (D40)" como en el
  YOLO26s destilado de partida (así ultralytics reusa su fila de bache).
"""
import argparse
import csv
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import comun  # noqa: E402
from preparar_splits import DHASH_MAX, dhash  # noqa: E402

RAIZ = comun.RAIZ


def importar(tarea):
    src_img, src_lbl, foto = tarea
    destino = RAIZ / "fotos" / f"{foto}.jpg"
    if not destino.exists():
        datos, _, _ = comun.reducir(src_img.read_bytes(), nombre=src_img.name)
        destino.write_bytes(datos)
    lineas = []
    for linea in src_lbl.read_text(encoding="utf-8").splitlines():
        p = linea.split()
        if len(p) >= 5:
            lineas.append(" ".join(["3"] + p[1:5]))   # HRP4K: clase 0 "pothole" -> 3 (D40)
    (RAIZ / "etiquetas" / f"{foto}.txt").write_text("\n".join(lineas) + ("\n" if lineas else ""), encoding="utf-8")
    return foto, len(lineas)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf", type=Path, required=True, help="carpeta para la descarga de Hugging Face")
    ap.add_argument("--salida", type=Path, required=True, help="carpeta del dataset de ultralytics a armar")
    args = ap.parse_args()

    snapshot_download("dronefreak/HRP4K", repo_type="dataset", local_dir=str(args.hf), token=False,
                      allow_patterns=["data/data.yaml", "data/images/train/*", "data/labels/train/*",
                                      "data/images/valid/*", "data/labels/valid/*"])
    tareas = []
    for split in ("train", "valid"):
        for img in sorted((args.hf / "data" / "images" / split).glob("*.jpg")):
            lbl = args.hf / "data" / "labels" / split / f"{img.stem}.txt"
            if lbl.exists():   # 2 fotos de la versión oficial no tienen etiquetas
                tareas.append((img, lbl, f"hrp4k_{split}_{img.stem}"))
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(importar, tareas, chunksize=16))
    print(f"importadas: {len(res)} | con bache: {sum(1 for _, n in res if n)} | baches: {sum(n for _, n in res)}")

    filas = list(csv.DictReader(open(RAIZ / "splits.csv", encoding="utf-8")))
    campos = list(filas[0].keys())
    ya = {r["foto"] for r in filas}
    nuevas = [f for f, _ in res]
    test = [r for r in filas if r["dataset"] == "hrp4k" and r["split"] == "test" and r["evaluable"] == "1"]
    with ProcessPoolExecutor(8) as ex:
        h_nuevas = np.unpackbits(np.stack(list(ex.map(dhash, [RAIZ / "fotos" / f"{f}.jpg" for f in nuevas],
                                                       chunksize=32))), axis=1)
        h_test = list(ex.map(dhash, [RAIZ / "fotos" / f"{r['foto']}.jpg" for r in test], chunksize=32))
    marcadas = 0
    for r, h in zip(test, h_test):
        dist = (np.unpackbits(h)[None, :] != h_nuevas).sum(1)
        j = int(dist.argmin())
        if dist[j] <= DHASH_MAX:
            r["evaluable"] = "0"
            r["motivo"] = f"casi copia de {nuevas[j]} (dhash {int(dist[j])}), del train de HRP4K"
            marcadas += 1
    filas += [{**{c: "" for c in campos}, "foto": f, "dataset": "hrp4k", "split": "train", "grupo": f,
               "evaluable": "1"} for f in nuevas if f not in ya]
    with open(RAIZ / "splits.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)
    print(f"test de HRP4K: {marcadas} fotos más dejan de evaluarse por ser casi copia de una nueva")

    subprocess.run([sys.executable, str(RAIZ / "entrenamiento" / "armar_dataset.py"), "--salida", str(args.salida)],
                   cwd=RAIZ, check=True)
    yaml = args.salida / "data.yaml"
    yaml.write_text(yaml.read_text(encoding="utf-8").replace("  0: D40\n", "  0: Pothole (D40)\n"), encoding="utf-8")
    print(f"listo: {yaml}")


if __name__ == "__main__":
    main()
