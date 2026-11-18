"""Tabla para la regla del ajuste con cuadros de Santa Cruz (informes/ajuste_scz.md): AP50 de baches a 640 px en el
test congelado, por conjunto, y en las mitades A y B de HRP4K, con la misma partición por md5 que elegir_modelo.py.
Se corrió en la máquina AMD, después de evaluar cada modelo con evaluar.py --split test --det PESOS --imgsz 640:

    python entrenamiento/tabla_ajuste.py v1 v2_a025 v2_a05 v2_a075 v2_a10

Lee informes/eval_NOMBRE_test_640.json y las predicciones de salidas/cache/NOMBRE_test_640.json.
Escribe informes/ajuste_tabla_amd.json, que usa decidir_ajuste.py."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "herramientas"))
import evaluar as ev  # noqa: E402
import preview as pv  # noqa: E402


def mitades():
    with open(RAIZ / "splits.csv", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    hrp = [r["foto"] for r in filas if r["split"] == "test" and r["evaluable"] == "1" and r["dataset"] == "hrp4k"]
    return {f: "A" if int(hashlib.md5(f.encode()).hexdigest(), 16) % 2 == 0 else "B" for f in hrp}


def hrp4k(nombre, mitad):
    p = RAIZ / "salidas" / "cache" / f"{nombre}_test_640.json"
    if not p.exists():
        return None
    preds = json.loads(p.read_text(encoding="utf-8"))
    res = {}
    for m in "AB":
        sel = [q for q in preds if mitad.get(q[0]) == m]
        dataset_de = {q[0]: "hrp4k" for q in sel}
        gts_de = {f: ev.leer_etiquetas(RAIZ / "etiquetas" / f"{f}.txt", w, h) for f, w, h, _, _ in sel}
        reg, n_gt = ev.medir(sel, dataset_de, gts_de, {"conf_bache": 0.2, "conf_grieta": 0.2,
                                                        "conf_seg": pv.CONF_SEG, "conf_seg_solo": 1.01})
        res[m] = round(ev.ap50(reg[("hrp4k", "D40", "det")], n_gt[("hrp4k", "D40")]), 4)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("modelos", nargs="+", help="nombres ya evaluados con evaluar.py")
    ap.add_argument("--salida", type=Path, default=RAIZ / "informes" / "ajuste_tabla_amd.json")
    args = ap.parse_args()
    mitad = mitades()
    salida = {}
    for nombre in args.modelos:
        p = RAIZ / "informes" / f"eval_{nombre}_test_640.json"
        if not p.exists():
            print(nombre, "sin evaluación")
            continue
        t = {f["dataset"]: round(f["AP50_det"], 4) for f in json.loads(p.read_text(encoding="utf-8"))["filas"]
             if f["clase"] == "D40"}
        salida[nombre] = {"test": t, "hrp4k": hrp4k(nombre, mitad)}
        print(nombre, t, salida[nombre]["hrp4k"], flush=True)
    args.salida.write_text(json.dumps(salida, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
