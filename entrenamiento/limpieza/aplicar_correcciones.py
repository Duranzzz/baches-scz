"""Aplica a etiquetas/ las 14 correcciones de la revisión de etiquetas del 1-nov-2026 (9 fotos de train y 5 de
val): cada archivo de correcciones/ reemplaza al de etiquetas/ con el mismo nombre. Quitan cajas de bache mal
puestas; las demás clases quedan. Guarda lo reemplazado en etiquetas/_antes_correcciones/.

Las 619 fotos retiradas del train (596 montajes, 11 duplicados exactos, 11 con superposiciones y 1 sin pavimento)
ya figuran en splits.csv con split "excluida"; la lista, con el motivo de cada una, está en excluir_train.csv.
Test no se tocó.

Correr una vez, después de bajar los datos (entrenamiento/hf.py):
    python entrenamiento/limpieza/aplicar_correcciones.py
"""
import shutil
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]


def main():
    etiquetas = RAIZ / "etiquetas"
    antes = etiquetas / "_antes_correcciones"
    antes.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted((AQUI / "correcciones").glob("*.txt")):
        destino = etiquetas / f.name
        if not destino.exists():
            raise SystemExit(f"falta {destino}: bajar primero los datos")
        if not (antes / f.name).exists():
            shutil.copy2(destino, antes / f.name)
        shutil.copy2(f, destino)
        n += 1
    print(f"{n} etiquetas corregidas (copia de las originales en {antes.relative_to(RAIZ)})")


if __name__ == "__main__":
    main()
