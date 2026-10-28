"""Importa el train y el valid de HRP4K V1.00 (Zenodo) a fotos/ y etiquetas/ con el prefijo hrp4k_.

    python entrenamiento/importar_hrp4k.py --zip /datos/HRP4K.zip

- El test (900) ya está en fotos/ desde la copia de dronefreak: aquí solo se comprueba que sus
  etiquetas coinciden con las oficiales.
- El zip oficial trae 4.203 etiquetas de train pero solo 2.286 imágenes: se importan las que tienen ambas.
- Una sola clase en el original (0 = bache) -> 3 en el esquema común.
- Fotos 4K reducidas a --lado px en el lado mayor.
- Escribe etiquetas/_origen/hrp4k-oficial/splits.csv (foto, split), que lee preparar_splits.py.
"""
import argparse
import csv
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comun  # noqa: E402

PREFIJO = "hrp4k_"
SPLITS = {"train": "train", "valid": "val"}
_zips = {}


def _zip(ruta):
    if ruta not in _zips:
        _zips[ruta] = zipfile.ZipFile(ruta)
    return _zips[ruta]


def remapear(texto):
    lineas = []
    for l in texto.splitlines():
        p = l.split()
        if not p:
            continue
        if p[0] != "0" or len(p) != 5:
            raise ValueError(f"línea inesperada: {l!r}")
        lineas.append(" ".join(["3"] + p[1:]))
    return "\n".join(lineas) + ("\n" if lineas else "")


def iguales(a, b, tol=1e-5):
    """Mismas cajas aunque una copia redondee a 6 decimales (dronefreak) y la otra no."""
    fa, fb = [float(x) for x in a.split()], [float(x) for x in b.split()]
    return len(fa) == len(fb) and all(abs(x - y) <= tol for x, y in zip(fa, fb))


def procesar(tarea):
    ruta_zip, img, lbl, foto, fotos, etiquetas, lado = tarea
    z = _zip(ruta_zip)
    comun.guardar_foto(z.read(img), Path(fotos) / f"{foto}.jpg", lado)
    (Path(etiquetas) / f"{foto}.txt").write_text(remapear(z.read(lbl).decode("utf-8")), encoding="utf-8")
    return foto


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=Path, required=True, help="HRP4K.zip de Zenodo")
    ap.add_argument("--lado", type=int, default=comun.LADO)
    ap.add_argument("--raiz", help="raíz del proyecto (por defecto, este repo)")
    args = ap.parse_args()

    raiz = comun.raiz(args.raiz)
    fotos, etiquetas = raiz / "fotos", raiz / "etiquetas"
    origen = etiquetas / "_origen" / "hrp4k-oficial"
    origen.mkdir(parents=True, exist_ok=True)

    z = zipfile.ZipFile(args.zip)
    nombres = set(z.namelist())
    stems_test = {Path(n).stem for n in nombres if n.startswith("HRP4K/test/labels/")}
    tareas, filas, sin_imagen = [], [], 0
    for split_zip, split in SPLITS.items():
        lbls = sorted(n for n in nombres if n.startswith(f"HRP4K/{split_zip}/labels/") and n.endswith(".txt"))
        for lbl in lbls:
            stem = Path(lbl).stem
            img = f"HRP4K/{split_zip}/images/{stem}.jpg"
            if img not in nombres:
                sin_imagen += 1
                continue
            foto = PREFIJO + stem
            if stem in stems_test:
                sys.exit(f"{foto} está en {split_zip} y en test: choque de nombres")
            tareas.append((str(args.zip), img, lbl, foto, str(fotos), str(etiquetas), args.lado))
            filas.append((foto, split))
    print(f"train/valid: {len(tareas)} fotos con etiqueta; {sin_imagen} etiquetas sin imagen (se saltan)")

    t0 = time.time()
    with ProcessPoolExecutor(comun.procesos()) as pool:
        list(pool.map(procesar, tareas, chunksize=16))
    print(f"importadas en {time.time() - t0:.0f} s")

    # El test local (dronefreak) contra el oficial
    distintas = faltan = 0
    tests = sorted(n for n in nombres if n.startswith("HRP4K/test/labels/") and n.endswith(".txt"))
    for lbl in tests:
        local = etiquetas / f"{PREFIJO}{Path(lbl).stem}.txt"
        if not local.exists():
            faltan += 1
        elif not iguales(local.read_text(encoding="utf-8"), remapear(z.read(lbl).decode("utf-8"))):
            distintas += 1
        filas.append((PREFIJO + Path(lbl).stem, "test"))
    print(f"test: {len(tests)} oficiales; {faltan} faltan en fotos/; {distintas} con etiquetas distintas")

    with open(origen / "splits.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["foto", "split"])
        w.writerows(sorted(filas))
    (origen / "README.txt").write_text(
        "Fuente: HRP4K V1.00, Zenodo 10.5281/zenodo.17522874 (Chen et al., Scientific Data 13:961, 2026).\n"
        "Licencia: CC BY 4.0. El zip oficial trae 2.286 imágenes de train para 4.203 etiquetas.\n"
        "Importado con entrenamiento/importar_hrp4k.py: train y valid; el test ya estaba en fotos/.\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
