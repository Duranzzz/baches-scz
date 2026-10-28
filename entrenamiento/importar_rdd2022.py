"""Importa el RDD2022 oficial (sekilab, en Figshare) a fotos/ y etiquetas/ con el prefijo rddo_.

    python entrenamiento/importar_rdd2022.py --zip /datos/RDD2022_released_through_CRDDC2022.zip

- Solo el train de cada país: el test oficial no trae etiquetas. Sin China_Drone (vista cenital).
- Esquema común: D00 -> 0, D10 -> 1, D20 -> 2, D40 -> 3. Los otros códigos del original (D44, D50,
  Repair...) se descartan, así esas zonas quedan como fondo y el modelo aprende que una senda o una tapa
  no es un bache. Se cuentan en el resumen.
- El XML se lee con un parser y solo se toman los <object>: en Noruega hay <name>rotation</name>
  dentro de atributos, que una búsqueda de texto confundiría con una clase.
- Las fotos con lado mayor > --lado se reducen (Noruega: 4040 x 2035 -> 1280 x 645).
- Escribe etiquetas/_origen/rdd2022-oficial/fuentes.csv (foto, país, tamaño original, cajas por código).
El zip externo trae un zip por país, guardado sin comprimir; se extraen a --tmp y se borran al final.
"""
import argparse
import csv
import shutil
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comun  # noqa: E402

CODIGOS = {"D00": 0, "D10": 1, "D20": 2, "D40": 3}
EXCLUIDOS = {"China_Drone"}
PREFIJO = "rddo_"
_zips = {}


def _zip(ruta):
    if ruta not in _zips:
        _zips[ruta] = zipfile.ZipFile(ruta)
    return _zips[ruta]


def procesar(tarea):
    pais, ruta_zip, xml_nombre, fotos, etiquetas, lado = tarea
    z = _zip(ruta_zip)
    raiz_xml = ET.fromstring(z.read(xml_nombre))
    stem = Path(xml_nombre).stem
    img_nombre = f"{pais}/train/images/{stem}.jpg"
    try:
        datos = z.read(img_nombre)
    except KeyError:
        return {"pais": pais, "error": f"sin imagen: {img_nombre}"}
    foto = PREFIJO + stem
    w, h = comun.guardar_foto(datos, Path(fotos) / f"{foto}.jpg", lado)
    tam_xml = (raiz_xml.findtext("size/width"), raiz_xml.findtext("size/height"))
    cuenta, otros, lineas, degeneradas = Counter(), Counter(), [], 0
    for obj in raiz_xml.findall("object"):
        nombre = (obj.findtext("name") or "").strip()
        if nombre not in CODIGOS:
            otros[nombre] += 1
            continue
        x1, y1, x2, y2 = (float(obj.findtext(f"bndbox/{t}")) for t in ("xmin", "ymin", "xmax", "ymax"))
        linea = comun.linea_yolo(CODIGOS[nombre], x1, y1, x2, y2, w, h)
        if linea is None:
            degeneradas += 1
            continue
        lineas.append(linea)
        cuenta[nombre] += 1
    (Path(etiquetas) / f"{foto}.txt").write_text("\n".join(lineas) + ("\n" if lineas else ""), encoding="utf-8")
    return {"pais": pais, "foto": foto, "ancho": w, "alto": h, "cuenta": cuenta, "otros": otros,
            "degeneradas": degeneradas, "tam_distinto": tam_xml != (str(w), str(h))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=Path, required=True, help="RDD2022_released_through_CRDDC2022.zip")
    ap.add_argument("--tmp", type=Path, help="dónde extraer los zips por país (por defecto, junto al zip)")
    ap.add_argument("--lado", type=int, default=comun.LADO)
    ap.add_argument("--raiz", help="raíz del proyecto (por defecto, este repo)")
    args = ap.parse_args()

    raiz = comun.raiz(args.raiz)
    fotos, etiquetas = raiz / "fotos", raiz / "etiquetas"
    origen = etiquetas / "_origen" / "rdd2022-oficial"
    for d in (fotos, etiquetas, origen):
        d.mkdir(parents=True, exist_ok=True)
    tmp = args.tmp or args.zip.parent / "rdd2022_paises"
    tmp.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    tareas = []
    with zipfile.ZipFile(args.zip) as externo:
        for miembro in externo.namelist():
            pais = Path(miembro).stem
            if not miembro.endswith(".zip"):
                continue
            if pais in EXCLUIDOS:
                print(f"{pais}: se salta (vista cenital)")
                continue
            ruta = tmp / f"{pais}.zip"
            if not ruta.exists():
                print(f"extrayendo {miembro} ...", flush=True)
                with externo.open(miembro) as src, open(ruta, "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 24)
            xmls = [n for n in zipfile.ZipFile(ruta).namelist()
                    if n.startswith(f"{pais}/train/annotations/xmls/") and n.endswith(".xml")]
            print(f"{pais}: {len(xmls)} fotos de train con etiquetas", flush=True)
            tareas += [(pais, str(ruta), x, str(fotos), str(etiquetas), args.lado) for x in xmls]
    if not tareas:
        sys.exit("no se encontraron zips por país dentro del zip")

    res = []
    with ProcessPoolExecutor(comun.procesos()) as pool:
        for i, r in enumerate(pool.map(procesar, tareas, chunksize=32)):
            res.append(r)
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1}/{len(tareas)}  {time.time() - t0:.0f} s", flush=True)

    errores = [r for r in res if "error" in r]
    ok = [r for r in res if "error" not in r]
    with open(origen / "fuentes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["foto", "pais", "ancho", "alto", *CODIGOS, "otros"])
        for r in sorted(ok, key=lambda r: r["foto"]):
            w.writerow([r["foto"], r["pais"], r["ancho"], r["alto"], *(r["cuenta"][c] for c in CODIGOS),
                        " ".join(f"{k}:{v}" for k, v in sorted(r["otros"].items()))])
    (origen / "README.txt").write_text(
        "Fuente: RDD2022 released through CRDDC'2022 (sekilab), Figshare 10.6084/m9.figshare.21431547\n"
        "Cita: Arya et al., RDD2022: A multi-national image dataset for automatic road damage detection.\n"
        "Licencia: CC BY 4.0 (ficha de Figshare).\n"
        "Importado con entrenamiento/importar_rdd2022.py: solo train, sin China_Drone, solo D00/D10/D20/D40.\n",
        encoding="utf-8")

    por_pais = defaultdict(lambda: {"fotos": 0, "cuenta": Counter(), "otros": Counter(), "degeneradas": 0,
                                    "tam_distinto": 0})
    for r in ok:
        p = por_pais[r["pais"]]
        p["fotos"] += 1
        p["cuenta"] += r["cuenta"]
        p["otros"] += r["otros"]
        p["degeneradas"] += r["degeneradas"]
        p["tam_distinto"] += r["tam_distinto"]
    print(f"\n{len(ok)} fotos importadas en {time.time() - t0:.0f} s; {len(errores)} errores")
    for e in errores[:10]:
        print("  error:", e["error"])
    for pais, p in sorted(por_pais.items()):
        con_bache = sum(1 for r in ok if r["pais"] == pais and r["cuenta"]["D40"])
        print(f"{pais:<16} {p['fotos']:>6} fotos ({con_bache} con bache) | "
              + " ".join(f"{c} {p['cuenta'][c]}" for c in CODIGOS)
              + f" | descartados {dict(p['otros'])} | degeneradas {p['degeneradas']}"
              + f" | tamaño XML distinto {p['tam_distinto']}")
    if args.tmp is None:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
