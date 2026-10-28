"""Arma la carpeta YOLO del detector de baches (1 clase) a partir de fotos/, etiquetas/ y splits.csv.

    python entrenamiento/armar_dataset.py --salida /datos/ds_baches
    python entrenamiento/armar_dataset.py --salida /datos/ds_prueba --fraccion 0.05     # una prueba chica

- Una sola clase, 0 = "D40" (bache), para que preview.py y evaluar.py la reconozcan sin cambios.
  Grietas y tapas quedan como fondo.
- train: el split train, también las fotos no evaluables (aumentadas, atardecer), que sirven para entrenar.
  val: solo val evaluable. El test no se copia: se mide aparte con evaluar.py.
- rdd2022_ (copia de dronefreak) nunca entra: su clase 3 no son solo baches.
- RDD2022 oficial: de las fotos sin bache entra 1 de cada 3 (--negativas), en train y en val, para que
  no sean la gran mayoría. En los demás datasets entran todas: son pocas y enseñan qué no es un bache
  en esa cámara.
- Las imágenes se enlazan (symlink; si no se puede, hardlink o copia), no se duplican.
La elección de negativas y de --fraccion es por hash del nombre: correrlo dos veces da lo mismo.
"""
import argparse
import csv
import hashlib
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comun  # noqa: E402

CLASE_BACHE = "3"          # en etiquetas/
SUBMUESTREO_NEGATIVAS = {"rddo"}
NUNCA = {"rdd2022"}


def azar(foto, sal):
    """Número en [0, 1) fijo para cada foto."""
    return int(hashlib.md5(f"{sal}:{foto}".encode()).hexdigest()[:8], 16) / 2**32


def enlazar(src, dst):
    try:
        os.symlink(src, dst)
    except OSError:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--salida", type=Path, required=True)
    ap.add_argument("--negativas", type=float, default=1 / 3, help="fracción de fotos sin bache de RDD2022")
    ap.add_argument("--fraccion", type=float, default=1.0, help="usar solo esta fracción (para pruebas)")
    ap.add_argument("--raiz", help="raíz del proyecto (por defecto, este repo)")
    args = ap.parse_args()

    raiz = comun.raiz(args.raiz)
    with open(raiz / "splits.csv", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))

    salida = args.salida.resolve()
    for sub in ("images", "labels"):
        shutil.rmtree(salida / sub, ignore_errors=True)
    for split in ("train", "val"):
        (salida / "images" / split).mkdir(parents=True)
        (salida / "labels" / split).mkdir(parents=True)

    stats = defaultdict(Counter)
    manifiesto = []
    for r in filas:
        foto, ds, split = r["foto"], r["dataset"], r["split"]
        if ds in NUNCA or split not in ("train", "val"):
            continue
        if split == "val" and r["evaluable"] != "1":
            continue
        if azar(foto, "fraccion") >= args.fraccion:
            continue
        lineas = [l.split() for l in (raiz / "etiquetas" / f"{foto}.txt").read_text(encoding="utf-8").splitlines()]
        baches = [" ".join(["0"] + p[1:]) for p in lineas if p and p[0] == CLASE_BACHE]
        if not baches and ds in SUBMUESTREO_NEGATIVAS and azar(foto, "negativas") >= args.negativas:
            stats[(ds, split)]["negativas descartadas"] += 1
            continue
        enlazar(raiz / "fotos" / f"{foto}.jpg", salida / "images" / split / f"{foto}.jpg")
        (salida / "labels" / split / f"{foto}.txt").write_text("\n".join(baches) + ("\n" if baches else ""),
                                                             encoding="utf-8")
        s = stats[(ds, split)]
        s["fotos"] += 1
        s["con bache"] += bool(baches)
        s["baches"] += len(baches)
        manifiesto.append((foto, ds, split, len(baches)))

    (salida / "data.yaml").write_text(
        f"# Generado por entrenamiento/armar_dataset.py\npath: {salida.as_posix()}\n"
        "train: images/train\nval: images/val\nnames:\n  0: D40\n", encoding="utf-8")
    with open(salida / "manifiesto.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["foto", "dataset", "split", "baches"])
        w.writerows(manifiesto)

    print(f"{'dataset':<8} {'split':<5} {'fotos':>6} {'con bache':>9} {'baches':>7} {'neg. descartadas':>16}")
    tot = defaultdict(Counter)
    for (ds, split), s in sorted(stats.items()):
        print(f"{ds:<8} {split:<5} {s['fotos']:>6} {s['con bache']:>9} {s['baches']:>7} "
              f"{s['negativas descartadas']:>16}")
        tot[split].update(s)
    for split, s in tot.items():
        print(f"{'TOTAL':<8} {split:<5} {s['fotos']:>6} {s['con bache']:>9} {s['baches']:>7}")
    print(f"\n{salida / 'data.yaml'}")


if __name__ == "__main__":
    main()
