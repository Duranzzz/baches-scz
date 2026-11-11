"""WiSE-FT (Wortsman et al., 2022, "Robust fine-tuning of zero-shot models", arXiv:2109.01903): promedia los
pesos de un modelo de partida y del mismo modelo ajustado. theta(alfa) = alfa*ajustado + (1-alfa)*partida.
Suele conservar buena parte de lo ganado en los datos del ajuste y recuperar lo que el ajuste perdió fuera de
ellos (aquí: el ajuste a1 subía mucho en nuestros datasets pero bajaba en HRP4K, que no vio).

    # con las clases del ajustado (1: bache); así se compararon alfa 0,25, 0,5 y 0,75
    python entrenamiento/promediar_pesos.py PARTIDA.pt AJUSTADO.pt 0.25 salida/wise_a25.pt
    # con las 4 clases del de partida: grietas sin tocar, bache promediado (el modelo de la app)
    python entrenamiento/promediar_pesos.py PARTIDA.pt AJUSTADO.pt 0.25 salida/wise4_a25.pt --clases partida

Cuerpo y regresión de cajas: promedio tensor a tensor. Cabeza de clases: el ajustado tiene una fila (bache) y el
de partida cuatro (D00, D10, D20, D40); se emparejan por nombre de clase, la misma correspondencia que usó
ultralytics al empezar el ajuste. Los .pt son pickles: correrlo solo con pesos de confianza.
El nombre del archivo de salida debe ser propio: evaluar.py nombra su caché por el nombre del archivo.
"""
import argparse
import copy
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("partida", type=Path)
    ap.add_argument("ajustado", type=Path)
    ap.add_argument("alfa", type=float, help="peso del ajustado: 0 = partida, 1 = ajustado")
    ap.add_argument("salida", type=Path)
    ap.add_argument("--clases", choices=["ajustado", "partida"], default="ajustado",
                    help="de quién son las clases del resultado")
    args = ap.parse_args()
    a = args.alfa

    ck0 = torch.load(args.partida, map_location="cpu", weights_only=False)
    ck1 = torch.load(args.ajustado, map_location="cpu", weights_only=False)
    m0 = (ck0.get("ema") or ck0["model"]).float()
    m1 = (ck1.get("ema") or ck1["model"]).float()
    fila = {k1: next(k0 for k0, n0 in m0.names.items() if n0 == n1) for k1, n1 in m1.names.items()}
    print(f"clases del ajustado {m1.names} <- filas {fila} de {m0.names}")
    sd0, sd1 = m0.state_dict(), m1.state_dict()
    base, ck_base = (m1, ck1) if args.clases == "ajustado" else (m0, ck0)

    nuevo, por_fila = {}, []
    for k, w in base.state_dict().items():
        v0, v1 = sd0.get(k), sd1.get(k)
        if not w.is_floating_point():
            nuevo[k] = w
        elif v0 is not None and v1 is not None and v0.shape == v1.shape:
            nuevo[k] = a * v1 + (1 - a) * v0
        elif (v0 is not None and v1 is not None and v0.shape[0] == len(m0.names) and v1.shape[0] == len(m1.names)
              and v0.shape[1:] == v1.shape[1:]):
            por_fila.append(k)
            if args.clases == "ajustado":
                nuevo[k] = a * v1 + (1 - a) * v0[[fila[i] for i in sorted(fila)]]
            else:
                t = v0.clone()
                for k1, k0 in fila.items():
                    t[k0] = a * v1[k1] + (1 - a) * v0[k0]
                nuevo[k] = t
        else:
            raise SystemExit(f"tensor sin par: {k}")
    print(f"{len(nuevo)} tensores; {len(por_fila)} de la cabeza de clases, promediados por fila")

    m = copy.deepcopy(base)
    m.load_state_dict(nuevo)
    ck = dict(ck_base)
    ck.update(model=m.half(), ema=None, optimizer=None,
              sopa={"metodo": "WiSE-FT", "alfa": a, "clases": args.clases,
                    "partida": str(args.partida), "ajustado": str(args.ajustado)})
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, args.salida)
    print(f"guardado: {args.salida}")


if __name__ == "__main__":
    main()
