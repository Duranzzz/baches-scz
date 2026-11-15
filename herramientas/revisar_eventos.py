"""Revisión a ciegas de las marcas de bache en los videos de Santa Cruz, que no tienen etiquetas.
    python herramientas/revisar_eventos.py armar        # recortes y hojas en salidas/revision_scz/
    python herramientas/revisar_eventos.py resultados   # con los juicios, la tabla por modelo
Cada evento (un tramo de 3 cuadros o más con marca, como en comparar_videos.py) se juzga por su marca más
segura: un recorte del cuadro con la caja dibujada. Los recortes de los cinco modelos se mezclan al azar y se
numeran, así quien juzga no sabe de qué modelo es cada uno. Los juicios van en
informes/videos_scz_20261122/revision/juicios.csv (id,juicio):
  B bache · O otro daño del pavimento (grieta, parche roto, desgaste) ·
  N no es daño (junta, mancha, sombra, tapa, vehículo, marca vial) · D dudoso."""
import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
VIDEOS = RAIZ / "informes" / "videos_scz_20261122"
REVISION = VIDEOS / "revision"
HOJAS = RAIZ / "salidas" / "revision_scz"
MODELOS = {"yolo12s_rdd2022": "YOLO12s (referencia)", "rdd_yolo26s": "B0 (pesos iniciales)",
           "wise4_a25": "Baches SCZ v1 (final)", "a1_y26s": "a1 (entrenado)", "a2_y26s_hrp4k": "a2 (con HRP4K)"}
HUECO_S, MIN_CUADROS = 0.5, 3          # los mismos de comparar_videos.py
TESELA, COLUMNAS, FILAS = (320, 200), 5, 4
JUICIOS = {"B": "bache", "O": "otro daño", "N": "no es daño", "D": "dudoso"}


def tramos(marcas, fps):
    """Igual que comparar_videos.eventos, pero devuelve (inicio, fin) de cada evento."""
    hueco = max(1, round(HUECO_S * fps))
    out, inicio, ultimo = [], None, None
    for i in sorted(marcas):
        if inicio is None or i - ultimo > hueco:
            if inicio is not None and ultimo - inicio + 1 >= MIN_CUADROS:
                out.append((inicio, ultimo))
            inicio = i
        ultimo = i
    if inicio is not None and ultimo - inicio + 1 >= MIN_CUADROS:
        out.append((inicio, ultimo))
    return out


def leer_eventos():
    eventos = []
    for carpeta in sorted(p for p in VIDEOS.iterdir() if (p / "resumen.json").exists()):
        resumen = json.loads((carpeta / "resumen.json").read_text(encoding="utf-8"))
        for clave in MODELOS:
            marcas = {}
            for f in csv.DictReader(open(carpeta / f"{clave}.csv", encoding="utf-8")):
                if f["clase"] == "bache":
                    caja = [float(f[k]) for k in ("x1", "y1", "x2", "y2")]
                    marcas.setdefault(int(f["cuadro"]), []).append((float(f["confianza"]), caja))
            spans = tramos(marcas, resumen["fps"])
            esperado = resumen["modelos"][clave]["eventos_bache"]
            if len(spans) != esperado:
                raise SystemExit(f"{carpeta.name}/{clave}: {len(spans)} eventos, el informe dice {esperado}")
            for inicio, fin in spans:
                conf, caja, cuadro = max((c, k, t) for t in range(inicio, fin + 1) for c, k in marcas.get(t, []))
                eventos.append({"video": resumen["video"], "modelo": clave, "inicio": inicio, "fin": fin,
                                "cuadro": cuadro, "conf": round(conf, 3), "caja": [round(v, 1) for v in caja],
                                "cajas": {t: [k for _, k in marcas[t]] for t in range(inicio, fin + 1) if t in marcas}})
    return eventos


def tesela(img, caja, n):
    """Recorte alrededor de la caja (3 veces su tamaño, mínimo 220 x 140), con la caja y el número."""
    alto, ancho = img.shape[:2]
    x1, y1, x2, y2 = caja
    w, h = max(3 * (x2 - x1), 220), max(3 * (y2 - y1), 140)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    X1, Y1 = int(np.clip(cx - w / 2, 0, max(0, ancho - w))), int(np.clip(cy - h / 2, 0, max(0, alto - h)))
    X2, Y2 = int(min(ancho, X1 + w)), int(min(alto, Y1 + h))
    rec = img[Y1:Y2, X1:X2]
    e = min(TESELA[0] / rec.shape[1], TESELA[1] / rec.shape[0])
    rw, rh = int(rec.shape[1] * e), int(rec.shape[0] * e)
    t = np.full((TESELA[1], TESELA[0], 3), 40, np.uint8)
    ox, oy = (TESELA[0] - rw) // 2, (TESELA[1] - rh) // 2
    t[oy:oy + rh, ox:ox + rw] = cv2.resize(rec, (rw, rh), interpolation=cv2.INTER_AREA if e < 1 else cv2.INTER_CUBIC)
    p1 = (int(ox + (x1 - X1) * e), int(oy + (y1 - Y1) * e))
    p2 = (int(ox + (x2 - X1) * e), int(oy + (y2 - Y1) * e))
    cv2.rectangle(t, p1, p2, (0, 255, 255), 2)
    cv2.rectangle(t, (0, 0), (52, 22), (0, 0, 0), -1)
    cv2.putText(t, str(n), (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return t


def armar():
    eventos = leer_eventos()
    random.Random(2026).shuffle(eventos)
    for i, e in enumerate(eventos, 1):
        e["id"] = i
    teselas = {}
    for video in sorted({e["video"] for e in eventos}):
        pedidos = {}
        for e in eventos:
            if e["video"] == video:
                pedidos.setdefault(e["cuadro"], []).append(e)
        cap, t = cv2.VideoCapture(str(RAIZ / "videos" / video)), 0
        while pedidos and t <= max(pedidos):
            ok, img = cap.read()
            if not ok:
                break
            for e in pedidos.get(t, []):
                teselas[e["id"]] = tesela(img, e["caja"], e["id"])
            t += 1
        cap.release()
    HOJAS.mkdir(parents=True, exist_ok=True)
    por_hoja = COLUMNAS * FILAS
    for h in range(0, len(eventos), por_hoja):
        lote = [teselas[i] for i in range(h + 1, min(len(eventos), h + por_hoja) + 1)]
        lote += [np.full_like(lote[0], 255)] * (por_hoja - len(lote))
        filas = [np.hstack(sum([[x, np.full((TESELA[1], 4, 3), 255, np.uint8)] for x in lote[f:f + COLUMNAS]], [])[:-1])
                 for f in range(0, por_hoja, COLUMNAS)]
        hoja = np.vstack(sum([[f, np.full((4, f.shape[1], 3), 255, np.uint8)] for f in filas], [])[:-1])
        cv2.imwrite(str(HOJAS / f"hoja_{h // por_hoja + 1:02d}.jpg"), hoja, [cv2.IMWRITE_JPEG_QUALITY, 88])
    REVISION.mkdir(parents=True, exist_ok=True)
    with open(REVISION / "eventos.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "video", "modelo", "inicio", "fin", "cuadro", "conf", "x1", "y1", "x2", "y2"])
        for e in sorted(eventos, key=lambda e: e["id"]):
            w.writerow([e["id"], e["video"], e["modelo"], e["inicio"], e["fin"], e["cuadro"], e["conf"], *e["caja"]])
    (HOJAS / "cajas.json").write_text(json.dumps({e["id"]: e["cajas"] for e in eventos}), encoding="utf-8")
    print(f"{len(eventos)} eventos en {-(-len(eventos) // por_hoja)} hojas: {HOJAS}")


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def resultados():
    eventos = {int(f["id"]): f for f in csv.DictReader(open(REVISION / "eventos.csv", encoding="utf-8"))}
    juicio = {int(f["id"]): f["juicio"].strip().upper() for f in csv.DictReader(open(REVISION / "juicios.csv", encoding="utf-8"))}
    faltan = sorted(set(eventos) - set(juicio))
    if faltan:
        raise SystemExit(f"faltan juicios: {faltan[:20]}")
    cajas = {int(k): {int(t): v for t, v in c.items()} for k, c in json.loads((HOJAS / "cajas.json").read_text()).items()}
    # Un mismo bache real puede tenerlo más de un modelo: se unen los eventos "B" que se superponen en el tiempo
    # y tienen cajas que se tocan (IoU >= 0,1) en algún cuadro común.
    baches = [i for i in eventos if juicio[i] == "B"]
    padre = {i: i for i in baches}

    def raiz(i):
        while padre[i] != i:
            padre[i] = padre[padre[i]]
            i = padre[i]
        return i
    for a in baches:
        for b in baches:
            if a < b and eventos[a]["video"] == eventos[b]["video"] and eventos[a]["modelo"] != eventos[b]["modelo"]:
                comunes = set(cajas[a]) & set(cajas[b])
                if any(iou(x, y) >= 0.1 for t in comunes for x in cajas[a][t] for y in cajas[b][t]):
                    padre[raiz(a)] = raiz(b)
    grupos = {raiz(i) for i in baches}
    filas = []
    for clave, nombre in MODELOS.items():
        mios = [i for i in eventos if eventos[i]["modelo"] == clave]
        n = {j: sum(juicio[i] == j for i in mios) for j in JUICIOS}
        juzgados = n["B"] + n["O"] + n["N"]
        cubre = {raiz(i) for i in mios if juicio[i] == "B"}
        filas.append({"modelo": nombre, "eventos": len(mios), **n,
                      "precision_bache": n["B"] / juzgados if juzgados else 0,
                      "precision_dano": (n["B"] + n["O"]) / juzgados if juzgados else 0,
                      "baches_hallados": len(cubre), "cobertura": len(cubre) / len(grupos) if grupos else 0})
    (REVISION / "resultados.json").write_text(json.dumps({"baches_distintos": len(grupos), "modelos": filas},
                                                         ensure_ascii=False, indent=1), encoding="utf-8")
    pct = lambda x: f"{100 * x:.0f} %"
    l = ["# Revisión a ciegas de las marcas de bache en los videos de Santa Cruz", "",
         f"{len(eventos)} eventos de 5 modelos, juzgados sin saber de qué modelo era cada uno (ver "
         "`herramientas/revisar_eventos.py`). Hay {} baches reales distintos entre todos los modelos.".format(len(grupos)), "",
         "| Modelo | Eventos | Bache | Otro daño | No es daño | Dudoso | Precisión (bache) | Precisión (cualquier daño) | Baches hallados | Cobertura |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for f in filas:
        l.append(f"| {f['modelo']} | {f['eventos']} | {f['B']} | {f['O']} | {f['N']} | {f['D']} | {pct(f['precision_bache'])} | "
                 f"{pct(f['precision_dano'])} | {f['baches_hallados']} | {pct(f['cobertura'])} |")
    l += ["", "- **Precisión:** de los eventos juzgados (sin los dudosos), cuántos son un bache (o cualquier daño).",
          "- **Cobertura:** de los baches reales distintos que encontró al menos un modelo, cuántos encontró este. Es "
          "una cota: un bache que no marcó ningún modelo no cuenta.", "- Juzga un solo revisor, con video de WhatsApp de 848 × 478. Las hojas con cada juicio quedan en salidas/revision_scz/juzgadas/."]
    (REVISION / "RESULTADOS.md").write_text("\n".join(l) + "\n", encoding="utf-8")
    print("\n".join(l))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paso", choices=["armar", "resultados"])
    {"armar": armar, "resultados": resultados}[ap.parse_args().paso]()


if __name__ == "__main__":
    main()
