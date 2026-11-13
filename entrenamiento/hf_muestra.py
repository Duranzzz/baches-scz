"""Sube a Hugging Face una muestra del test congelado, para verla en la pestaña de datos del repositorio.
    python entrenamiento/hf_muestra.py            # la arma en salidas/hf_muestra/
    python entrenamiento/hf_muestra.py --subir    # y la sube a muestra/ de Duranishhh/baches-scz
Toma 50 fotos evaluables del test por fuente, en un orden fijo (por su md5), alternando fuentes. Solo entran
fotos con daño y con clases que su fuente marca completas. Cada fila lleva la foto (reducida a 1.280 px de lado
mayor, como en paquete/), su fuente y sus cajas en formato COCO: x, y, ancho y alto en píxeles."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image

RAIZ = Path(__file__).resolve().parents[1]
REPO = "Duranishhh/baches-scz"
SALIDA = RAIZ / "salidas" / "hf_muestra"
POR_FUENTE = 50
LADO = 1280
# fuente: (nombre, clases que marca completas). 0-2 grietas D00/D10/D20, 3 bache, 4 grieta sin tipo, 5 tapa
FUENTES = {
    "bisman": ("Bisman", {3}),
    "brazil": ("Brasil", {3, 4}),
    "bucko": ("Bucko (Eslovaquia)", {3, 5}),
    "egypt": ("Egipto", {3, 4}),
    "hrp4k": ("HRP4K", {3}),
    "rdd2022": ("RDD2022", {0, 1, 2}),
}
GRIETAS = {0, 1, 2, 4}


def cajas(etiqueta, ancho, alto):
    """Lee una etiqueta YOLO (clase, centro x, centro y, ancho, alto relativos) y la pasa a COCO en píxeles."""
    out = []
    for linea in etiqueta.read_text(encoding="utf-8").splitlines():
        partes = linea.split()
        if len(partes) != 5:
            continue
        c, (cx, cy, w, h) = int(partes[0]), map(float, partes[1:])
        x, y = max(0.0, (cx - w / 2) * ancho), max(0.0, (cy - h / 2) * alto)
        out.append((c, [round(x, 1), round(y, 1), round(w * ancho, 1), round(h * alto, 1)]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subir", action="store_true", help="subir la muestra a Hugging Face")
    args = ap.parse_args()

    fotos = {p.stem: p for p in (RAIZ / "fotos").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    candidatas = {f: [] for f in FUENTES}
    with open(RAIZ / "splits.csv", encoding="utf-8") as fh:
        for fila in csv.DictReader(fh):
            fuente, foto = fila["dataset"], fila["foto"]
            etiqueta = RAIZ / "etiquetas" / f"{foto}.txt"
            if fila["split"] != "test" or fila["evaluable"] != "1" or fuente not in FUENTES:
                continue
            if foto not in fotos or not etiqueta.exists():
                continue
            clases = {c for c, _ in cajas(etiqueta, 1, 1)}
            if clases - {5} and clases <= FUENTES[fuente][1]:
                candidatas[fuente].append(foto)
    for fuente, lista in candidatas.items():
        lista.sort(key=lambda f: hashlib.md5(f.encode()).hexdigest())
        print(f"{fuente}: {len(lista)} candidatas")
        if len(lista) < POR_FUENTE:
            raise SystemExit(f"{fuente} tiene menos de {POR_FUENTE} fotos que sirvan")

    destino = SALIDA / "muestra" / "test"
    if SALIDA.exists():
        shutil.rmtree(SALIDA)
    destino.mkdir(parents=True)
    filas = []
    for k in range(POR_FUENTE):
        for fuente, (nombre, _) in FUENTES.items():
            origen = fotos[candidatas[fuente][k]]
            archivo = f"{len(filas) + 1:03d}_{origen.name}"
            with Image.open(origen) as im:
                ancho, alto = im.size
                if max(ancho, alto) > LADO:
                    escala = LADO / max(ancho, alto)
                    ancho, alto = round(ancho * escala), round(alto * escala)
                    im.convert("RGB").resize((ancho, alto), Image.LANCZOS).save(destino / archivo, quality=90)
                else:
                    shutil.copy2(origen, destino / archivo)
            cs = cajas(RAIZ / "etiquetas" / f"{origen.stem}.txt", ancho, alto)
            filas.append({
                "file_name": archivo, "foto": origen.name, "fuente": nombre, "ancho": ancho, "alto": alto,
                "num_baches": sum(c == 3 for c, _ in cs), "num_grietas": sum(c in GRIETAS for c, _ in cs),
                "objects": {"bbox": [b for _, b in cs], "category": [c for c, _ in cs]},
            })
    with open(destino / "metadata.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for f in filas:
            fh.write(json.dumps(f, ensure_ascii=False) + "\n")
    mb = sum(p.stat().st_size for p in destino.iterdir()) / 1e6
    print(f"{len(filas)} fotos, {mb:.0f} MB en {destino}")

    if args.subir:
        from huggingface_hub import HfApi
        HfApi().upload_folder(repo_id=REPO, repo_type="dataset", folder_path=SALIDA / "muestra",
                              path_in_repo="muestra", delete_patterns="*",
                              commit_message=f"Muestra del test para la vista previa ({len(filas)} fotos)")
        print(f"subida: https://huggingface.co/datasets/{REPO}/tree/main/muestra")


if __name__ == "__main__":
    main()
