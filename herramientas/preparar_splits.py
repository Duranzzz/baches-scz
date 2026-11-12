"""Arma splits.csv: a qué partición va cada foto de fotos/, separando por grupo (video, tramo de ruta)
para que fotogramas casi iguales no queden a la vez en train y en test.

    python herramientas/preparar_splits.py
    python herramientas/preparar_splits.py --raiz /otra/copia     # para probar en otra carpeta

Lee etiquetas/_origen/ (sources.csv, listas) y escribe splits.csv con las columnas
foto, dataset, split (train | val | test | excluida), grupo, condicion, evaluable, motivo.
`evaluable` = 0 si la foto puede entrenar pero no sirve para medir (aumentada, mosaico, de noche,
casi copia de una foto de train). La semilla es fija: correrlo dos veces da el mismo archivo.
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
FOTOS = RAIZ / "fotos"
ORIGEN = RAIZ / "etiquetas" / "_origen"
SALIDA = RAIZ / "splits.csv"
SEMILLA = 2026
BLOQUE_BRASIL = 25        # fotos consecutivas de una misma ruta que van juntas
# dHash de 256 bits. Revisado a ojo el 2026-10-28: hasta 25 bits son la misma foto recortada o
# recomprimida (Bisman repite fotos entre voc_potholes y pothole_dataset_1); desde 26 son fotos
# distintas de una misma ruta de Brasil, que se parecen mucho en composición. Con 64 bits se confundían.
DHASH_LADO = 16
DHASH_MAX = 25


def leer_csv(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def repartir(grupos, peso, frac_test, frac_val, rnd):
    """Asigna grupos enteros a test y val hasta llegar a la fracción del peso total; el resto a train."""
    orden = sorted(grupos)
    rnd.shuffle(orden)
    total = sum(peso[g] for g in orden)
    split, acum = {}, {"test": 0, "val": 0}
    for g in orden:
        if peso[g] and acum["test"] < frac_test * total:
            split[g] = "test"
            acum["test"] += peso[g]
        elif peso[g] and acum["val"] < frac_val * total:
            split[g] = "val"
            acum["val"] += peso[g]
        else:
            split[g] = "train"
    return split


def filas():
    rnd = random.Random(SEMILLA)
    out = []

    def fila(foto, dataset, split, grupo, condicion="", evaluable=1, motivo=""):
        out.append({"foto": foto, "dataset": dataset, "split": split, "grupo": grupo,
                    "condicion": condicion, "evaluable": evaluable, "motivo": motivo})

    # Bisman (1er parcial): se respetan sus particiones originales
    for split, nombre in (("train", "train"), ("val", "validation"), ("test", "test")):
        for l in (ORIGEN / "bisman" / f"metadata_{nombre}.jsonl").read_text(encoding="utf-8").splitlines():
            stem = "bisman_" + Path(json.loads(l)["file_name"]).stem
            fila(stem, "bisman", split, stem)

    # RDD2022 oficial (sekilab), si ya se importó: 80/10/10 al azar por país (no hay videos ni tramos).
    # Con su propio generador, para que importarlo no cambie cómo se reparten los demás datasets.
    fuentes_rddo = ORIGEN / "rdd2022-oficial" / "fuentes.csv"
    rddo = leer_csv(fuentes_rddo) if fuentes_rddo.exists() else []
    rnd_rddo = random.Random(f"{SEMILLA}-rddo")
    por_pais = defaultdict(list)
    for r in rddo:
        por_pais[r["pais"]].append(r["foto"])
    for pais, fotos_pais in sorted(por_pais.items()):
        sp = repartir(fotos_pais, {f: 1 for f in fotos_pais}, 0.10, 0.10, rnd_rddo)
        for f in fotos_pais:
            fila(f, "rddo", sp[f], f, pais)

    # Muestra del test de dronefreak: su clase 3 mezcla baches con sendas, tapas y parches, y sus fotos
    # son copias del train oficial. Con el oficial importado, sobra.
    for p in sorted(FOTOS.glob("rdd2022_*.jpg")):
        if rddo:
            fila(p.stem, "rdd2022", "excluida", p.stem, evaluable=0, motivo="copia del RDD2022 oficial (rddo_)")
        else:
            fila(p.stem, "rdd2022", "test", p.stem)

    # HRP4K: train y valid oficiales si se importaran (no se hizo); el resto es el test oficial
    splits_hrp = ORIGEN / "hrp4k-oficial" / "splits.csv"
    split_hrp = {r["foto"]: r["split"] for r in leer_csv(splits_hrp)} if splits_hrp.exists() else {}
    for p in sorted(FOTOS.glob("hrp4k_*.jpg")):
        fila(p.stem, "hrp4k", split_hrp.get(p.stem, "test"), p.stem)

    # Bucko (dashcam, Eslovaquia): por video. Val y test solo con videos de día (clear)
    bucko = leer_csv(ORIGEN / "bucko2022-dashcam-potholes" / "sources.csv")
    cond_video = {r["video"]: r["condition"] for r in bucko}
    peso = defaultdict(int)
    for r in bucko:
        peso[r["video"]] += r["condition"] == "clear"
    sp = repartir([v for v in peso if cond_video[v] == "clear"], peso, 0.25, 0.10, rnd)
    for r in bucko:
        c = r["condition"]
        if c == "night":
            fila(r["image"], "bucko", "excluida", r["video"], c, 0, "noche: fuera de alcance")
        elif c == "clear":
            fila(r["image"], "bucko", sp[r["video"]], r["video"], c)
        else:
            fila(r["image"], "bucko", "train", r["video"], c, 0, f"{c}: fuera de alcance para medir")

    # Egipto (celular): por video. Solo los fotogramas limpios (sin aumentar) sirven para medir
    egipto = leer_csv(ORIGEN / "egypt-rdd-roboflow" / "sources.csv")
    limpias = {Path(l).stem for l in (ORIGEN / "egypt-rdd-roboflow" / "clean_valid_test.txt").read_text().split()}
    peso = defaultdict(int)
    for r in egipto:
        peso[r["video"]] += r["image"] in limpias
    sp = repartir([v for v in peso if peso[v]], peso, 0.40, 0.20, rnd)
    for r in egipto:
        s = sp.get(r["video"], "train")
        if r["image"] in limpias:
            fila(r["image"], "egypt", s, r["video"])
        elif s == "train":
            fila(r["image"], "egypt", "train", r["video"], evaluable=0, motivo="aumentada por Roboflow")
        else:
            # mosaico o aumentada de un video de val/test: ni entrena (fuga) ni mide
            fila(r["image"], "egypt", "excluida", r["video"], evaluable=0, motivo="aumentada, video de val/test")

    # Brasil (DNIT): bloques de fotos consecutivas por estado y ruta
    brasil = leer_csv(ORIGEN / "brazil-dnit-cracks-potholes" / "sources.csv")
    por_ruta = defaultdict(list)
    for r in brasil:
        por_ruta[(r["state"], r["highway"])].append(r)
    grupo_de = {}
    for (estado, ruta), rs in por_ruta.items():
        rs.sort(key=lambda r: r["source_id"])
        for i, r in enumerate(rs):
            grupo_de[r["image"]] = f"{estado}_{ruta}_{i // BLOQUE_BRASIL:03d}"
    peso = defaultdict(int)
    for g in grupo_de.values():
        peso[g] += 1
    sp = repartir(list(peso), peso, 0.15, 0.15, rnd)
    for r in brasil:
        g = grupo_de[r["image"]]
        fila(r["image"], "brazil", sp[g], g, r["state"])
    return out


def dhash(p):
    img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    peq = cv2.resize(img, (DHASH_LADO + 1, DHASH_LADO), interpolation=cv2.INTER_AREA)
    return np.packbits((peq[:, 1:] > peq[:, :-1]).flatten())


def marcar_duplicados(out):
    """Val/test con una casi copia en train (de cualquier dataset) -> evaluable = 0."""
    nombres = [r["foto"] for r in out]
    with ProcessPoolExecutor() as pool:
        hashes = dict(zip(nombres, pool.map(dhash, [FOTOS / f"{f}.jpg" for f in nombres], chunksize=64)))
    train = [r["foto"] for r in out if r["split"] == "train"]
    h_train = np.stack([hashes[f] for f in train])
    popcount = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)
    n = 0
    for r in out:
        if r["split"] not in ("val", "test") or not r["evaluable"]:
            continue
        dist = popcount[h_train ^ hashes[r["foto"]]].sum(1)
        j = int(dist.argmin())
        if dist[j] <= DHASH_MAX:
            r["evaluable"], r["motivo"] = 0, f"casi copia de {train[j]} (dhash {dist[j]})"
            n += 1
    return n


def main():
    global FOTOS, ORIGEN, SALIDA
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raiz", type=Path, default=RAIZ)
    raiz = ap.parse_args().raiz.resolve()
    FOTOS, ORIGEN, SALIDA = raiz / "fotos", raiz / "etiquetas" / "_origen", raiz / "splits.csv"
    out = filas()
    fotos = {p.stem for p in FOTOS.glob("*.jpg")}
    asignadas = [r["foto"] for r in out]
    assert len(asignadas) == len(set(asignadas)), "foto repetida en splits"
    assert set(asignadas) == fotos, f"sin asignar: {sorted(fotos - set(asignadas))[:5]}"
    dup = marcar_duplicados(out)
    with open(SALIDA, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(sorted(out, key=lambda r: r["foto"]))

    print(f"{SALIDA.name}: {len(out)} fotos, {dup} de val/test marcadas como casi copia de train")
    cuenta = defaultdict(lambda: defaultdict(int))
    for r in out:
        clave = r["split"] + ("" if r["evaluable"] else "*")
        cuenta[r["dataset"]][clave] += 1
    print("(* = no evaluable)")
    for ds, c in sorted(cuenta.items()):
        print(f"  {ds:<8}", "  ".join(f"{k} {v}" for k, v in sorted(c.items())))


if __name__ == "__main__":
    main()
