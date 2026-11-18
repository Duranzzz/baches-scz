"""Aplica la regla del ajuste con cuadros de Santa Cruz (informes/ajuste_scz.md, fijada el 23-nov antes de etiquetar y
de entrenar) a v1 y a las variantes v2. Todo a 640 px; un candidato reemplaza a v1 solo si cumple las tres:
  - AP50 en los 80 cuadros de prueba de Santa Cruz >= v1 + 0,03;
  - AP50 en Egipto (test congelado) >= v1 - 0,01;
  - AP50 en HRP4K-A >= v1 - 0,005.
Entre los que cumplen gana el de mayor AP50 en Santa Cruz; si ninguno cumple, v1 sigue.

    python entrenamiento/decidir_ajuste.py

Lee informes/ajuste_tabla_amd.json (tabla_ajuste.py) y informes/scz_cuadros/eval_prueba_640*.json
(scz_cuadros.py evaluar --parte prueba). Escribe informes/ajuste_scz.json, que usa docs/armar_articulo.py."""
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
CANDIDATOS = [("v2_a025", 0.25), ("v2_a05", 0.5), ("v2_a075", 0.75), ("v2_a10", 1.0)]


def leer(ruta):
    return json.loads((RAIZ / ruta).read_text(encoding="utf-8"))


def main():
    tabla = leer("informes/ajuste_tabla_amd.json")
    p50 = leer("informes/scz_cuadros/eval_prueba_640.json")["ap50"]
    p30 = leer("informes/scz_cuadros/eval_prueba_640_iou30.json")["ap50"]

    def datos(k):
        t = tabla[k]
        return {"scz_prueba50": p50[k], "scz_prueba30": p30[k], "egipto": t["test"]["egypt"],
                "test": t["test"]["todos"], "hrp4k_a": t["hrp4k"]["A"], "hrp4k_b": t["hrp4k"]["B"]}

    v1 = datos("v1")
    candidatos = {}
    for k, alfa in CANDIDATOS:
        d = datos(k) | {"alfa": alfa}
        d["condiciones"] = {"santa_cruz": d["scz_prueba50"] >= v1["scz_prueba50"] + 0.03,
                            "egipto": d["egipto"] >= v1["egipto"] - 0.01,
                            "hrp4k_a": d["hrp4k_a"] >= v1["hrp4k_a"] - 0.005}
        d["cumple"] = all(d["condiciones"].values())
        candidatos[k] = d
    ganador = max((k for k, d in candidatos.items() if d["cumple"]), key=lambda k: candidatos[k]["scz_prueba50"],
                  default=None)
    salida = {"v1": v1, "candidatos": candidatos, "ganador": ganador}
    semillas = {k: tabla[k] for k in ("B0", "wise4_s1", "a1_s1") if k in tabla}   # estabilidad, solo se informa
    if semillas:
        salida["semillas"] = semillas
    (RAIZ / "informes" / "ajuste_scz.json").write_text(json.dumps(salida, ensure_ascii=False, indent=1) + "\n",
                                                       encoding="utf-8")
    print("ganador:", ganador or "ninguno, v1 sigue")
    for k, d in candidatos.items():
        print(k, {c: round(d[c], 3) for c in ("scz_prueba50", "egipto", "hrp4k_a")}, d["condiciones"])


if __name__ == "__main__":
    main()
