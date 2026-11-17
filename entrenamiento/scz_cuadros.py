"""Cuadros de los videos de Santa Cruz con cajas de bache, para medir los modelos en la ciudad y para ajustarlos.
    python entrenamiento/scz_cuadros.py muestrear      # elige los 200 cuadros y arma las hojas para etiquetar
    python entrenamiento/scz_cuadros.py etiquetas      # juicios -> cajas; extrae los cuadros de los videos difuminados
    python entrenamiento/scz_cuadros.py evaluar v1=modelos/baches_scz_v1.pt a1=... [--imgsz 640] [--parte prueba]
Plan, partición y regla, fijados antes de etiquetar: informes/ajuste_scz.md.
- Muestra (semilla 2026): en cada video, el 60 % inicial es "ajuste" y el 40 % final, después de 2 s, "prueba".
  En cada parte, la mitad de los cuadros es al azar y la otra mitad, de cuadros donde algún modelo marcó un
  bache; entre dos cuadros elegidos hay al menos 0,5 s.
- Hojas: cada cuadro con una grilla cada 100 px y las cajas de bache que propusieron los cinco modelos de
  comparar_videos.py, unidas y numeradas, sin decir de qué modelo es cada una.
- Juicios (informes/scz_cuadros/juicios.csv: id,aceptadas,agregadas): los números de las cajas que son baches y,
  separadas por ';', las cajas "x1 y1 x2 y2" de los baches que ninguna propuso.
- Evaluación: AP50 de baches con 101 puntos, con las mismas funciones que herramientas/evaluar.py."""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "herramientas"))
VIDEOS = RAIZ / "informes" / "videos_scz_20261122"
CARPETA = RAIZ / "informes" / "scz_cuadros"
HOJAS = RAIZ / "salidas" / "scz_cuadros" / "hojas"
DATASET = RAIZ / "salidas" / "scz_cuadros" / "dataset"
DIFUMINADOS = RAIZ / "salidas" / "difuminar"
CUADROS_POR_VIDEO = {"scz_01_lateral_auto_hormigon": 55, "scz_02_frontal_avenida": 40,
                     "scz_03_frontal_doble_via": 75, "scz_04_lateral_micro_losas": 15,
                     "scz_05_lateral_micro_hormigon": 15}
AJUSTE, SEPARACION_S, ESPACIO_S = 0.6, 2.0, 0.5
MODELOS = ["yolo12s_rdd2022", "rdd_yolo26s", "wise4_a25", "a1_y26s", "a2_y26s_hrp4k"]
POR_HOJA = 3
MARGEN_X, MARGEN_Y = 34, 22


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def leer_cuadros(video, quiero, fuente):
    """{cuadro: imagen} leyendo el video de corrido (más seguro que saltar con set)."""
    cap, t, out = cv2.VideoCapture(str(fuente)), 0, {}
    ultimo = max(quiero)
    while t <= ultimo:
        ok, img = cap.read()
        if not ok:
            break
        if t in quiero:
            out[t] = img
        t += 1
    cap.release()
    faltan = set(quiero) - set(out)
    if faltan:
        raise SystemExit(f"{video}: no se pudieron leer los cuadros {sorted(faltan)[:5]}")
    return out


def muestrear():
    rng = random.Random(2026)
    muestra, candidatos = [], {}
    for video, n in CUADROS_POR_VIDEO.items():
        r = json.loads((VIDEOS / video / "resumen.json").read_text(encoding="utf-8"))
        total, fps = r["cuadros"], r["fps"]
        marcas = {}
        for m in MODELOS:
            for f in csv.DictReader(open(VIDEOS / video / f"{m}.csv", encoding="utf-8")):
                if f["clase"] == "bache":
                    caja = [float(f[k]) for k in ("x1", "y1", "x2", "y2")]
                    marcas.setdefault(int(f["cuadro"]), []).append((float(f["confianza"]), caja))
        corte = int(total * AJUSTE)
        partes = {"ajuste": range(0, corte), "prueba": range(corte + round(SEPARACION_S * fps), total)}
        n_ajuste = round(n * AJUSTE)
        espacio = max(1, round(ESPACIO_S * fps))
        for parte, rango in partes.items():
            cuantos = n_ajuste if parte == "ajuste" else n - n_ajuste
            marcados, al_azar = [t for t in rango if t in marcas], list(rango)
            rng.shuffle(marcados)
            rng.shuffle(al_azar)
            elegidos = []
            for lista, meta in ((marcados, cuantos // 2), (al_azar, cuantos)):
                for t in lista:
                    if len(elegidos) >= meta:
                        break
                    if all(abs(t - e) >= espacio for e in elegidos):
                        elegidos.append(t)
            for t in sorted(elegidos):
                muestra.append({"video": video, "cuadro": t, "parte": parte, "marcado": int(t in marcas)})
                unidas = []
                for conf, caja in sorted(marcas.get(t, []), key=lambda x: -x[0]):
                    if all(iou(caja, u) < 0.5 for u in unidas):
                        unidas.append(caja)
                candidatos[(video, t)] = [[round(v, 1) for v in c] for c in unidas]
    muestra.sort(key=lambda f: (f["parte"] != "ajuste", f["video"], f["cuadro"]))
    for i, f in enumerate(muestra, 1):
        f["id"] = i
    CARPETA.mkdir(parents=True, exist_ok=True)
    with open(CARPETA / "muestra.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "video", "cuadro", "parte", "marcado"])
        w.writeheader()
        w.writerows(muestra)
    (CARPETA / "candidatos.json").write_text(json.dumps(
        {f["id"]: candidatos[(f["video"], f["cuadro"])] for f in muestra}), encoding="utf-8")
    # hojas: 3 cuadros por imagen, con grilla y cajas numeradas
    HOJAS.mkdir(parents=True, exist_ok=True)
    paneles = {}
    for video in CUADROS_POR_VIDEO:
        mios = [f for f in muestra if f["video"] == video]
        imgs = leer_cuadros(video, {f["cuadro"] for f in mios}, RAIZ / "videos" / f"{video}.mp4")
        for f in mios:
            paneles[f["id"]] = panel(imgs[f["cuadro"]], candidatos[(video, f["cuadro"])],
                                     f"{f['id']} | {video[:6]} cuadro {f['cuadro']} | {f['parte']}")
    for h in range(0, len(muestra), POR_HOJA):
        lote = [paneles[i] for i in range(h + 1, min(len(muestra), h + POR_HOJA) + 1)]
        ancho = max(p.shape[1] for p in lote)   # los videos miden 848 o 850 px de ancho
        lote = [cv2.copyMakeBorder(p, 0, 0, 0, ancho - p.shape[1], cv2.BORDER_CONSTANT, value=(30, 30, 30)) for p in lote]
        sep = np.full((6, ancho, 3), 255, np.uint8)
        hoja = np.vstack(sum([[p, sep] for p in lote], [])[:-1])
        cv2.imwrite(str(HOJAS / f"hoja_{h // POR_HOJA + 1:02d}.jpg"), hoja, [cv2.IMWRITE_JPEG_QUALITY, 90])
    n_aj = sum(f["parte"] == "ajuste" for f in muestra)
    print(f"{len(muestra)} cuadros ({n_aj} de ajuste, ids 1-{n_aj}; {len(muestra) - n_aj} de prueba), "
          f"{sum(len(v) for v in candidatos.values())} cajas propuestas, "
          f"{-(-len(muestra) // POR_HOJA)} hojas en {HOJAS}")


def panel(img, cajas, titulo):
    alto, ancho = img.shape[:2]
    p = np.full((alto + MARGEN_Y, ancho + MARGEN_X, 3), 30, np.uint8)
    p[MARGEN_Y:, MARGEN_X:] = img
    capa = p.copy()
    for x in range(100, ancho, 100):
        cv2.line(capa, (MARGEN_X + x, MARGEN_Y), (MARGEN_X + x, MARGEN_Y + alto), (255, 255, 255), 1)
        cv2.putText(p, str(x), (MARGEN_X + x - 12, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    for y in range(100, alto, 100):
        cv2.line(capa, (MARGEN_X, MARGEN_Y + y), (MARGEN_X + ancho, MARGEN_Y + y), (255, 255, 255), 1)
        cv2.putText(p, str(y), (2, MARGEN_Y + y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    p = cv2.addWeighted(capa, 0.3, p, 0.7, 0)
    for k, (x1, y1, x2, y2) in enumerate(cajas, 1):
        a, b = (int(MARGEN_X + x1), int(MARGEN_Y + y1)), (int(MARGEN_X + x2), int(MARGEN_Y + y2))
        cv2.rectangle(p, a, b, (0, 255, 255), 1)
        cv2.rectangle(p, (a[0], a[1] - 13), (a[0] + 14, a[1]), (0, 0, 0), -1)
        cv2.putText(p, str(k), (a[0] + 2, a[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.putText(p, titulo, (MARGEN_X + 400, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    return p


def leer_juicios():
    muestra = {int(f["id"]): f for f in csv.DictReader(open(CARPETA / "muestra.csv", encoding="utf-8"))}
    cand = {int(k): v for k, v in json.loads((CARPETA / "candidatos.json").read_text(encoding="utf-8")).items()}
    cajas = {}
    for f in csv.DictReader(open(CARPETA / "juicios.csv", encoding="utf-8")):
        i = int(f["id"])
        out = [cand[i][int(k) - 1] for k in f["aceptadas"].split()]
        out += [[float(v) for v in c.split()] for c in f["agregadas"].split(";") if c.strip()]
        cajas[i] = out
    return muestra, cajas


def etiquetas():
    muestra, cajas = leer_juicios()
    faltan = sorted(set(muestra) - set(cajas))
    if faltan:
        print(f"aviso: faltan juicios de {len(faltan)} cuadros ({faltan[:10]}...); se extraen solo los juzgados")
    with open(CARPETA / "cajas.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "video", "cuadro", "parte", "x1", "y1", "x2", "y2"])
        for i in sorted(cajas):
            for c in cajas[i]:
                w.writerow([i, muestra[i]["video"], muestra[i]["cuadro"], muestra[i]["parte"], *c])
    for video in CUADROS_POR_VIDEO:
        mios = [i for i in cajas if muestra[i]["video"] == video]
        if not mios:
            continue
        imgs = leer_cuadros(video, {int(muestra[i]["cuadro"]) for i in mios}, DIFUMINADOS / f"{video}.mp4")
        for i in mios:
            f = muestra[i]
            nombre = f"{video[:6]}_{int(f['cuadro']):05d}"
            img = imgs[int(f["cuadro"])]
            alto, ancho = img.shape[:2]
            for sub in ("images", "labels"):
                (DATASET / sub / f["parte"]).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(DATASET / "images" / f["parte"] / f"{nombre}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            lineas = [f"0 {(x1 + x2) / 2 / ancho:.6f} {(y1 + y2) / 2 / alto:.6f} {(x2 - x1) / ancho:.6f} "
                      f"{(y2 - y1) / alto:.6f}" for x1, y1, x2, y2 in cajas[i]]
            (DATASET / "labels" / f["parte"] / f"{nombre}.txt").write_text("\n".join(lineas) + ("\n" if lineas else ""))
    n = {p: sum(1 for i in cajas if muestra[i]["parte"] == p) for p in ("ajuste", "prueba")}
    b = {p: sum(len(cajas[i]) for i in cajas if muestra[i]["parte"] == p) for p in ("ajuste", "prueba")}
    print(f"cuadros: {n} | baches: {b} | dataset en {DATASET}")


def evaluar(modelos, imgsz, parte, iou_min=0.5):
    import evaluar as ev
    ev.IOU_ACIERTO = iou_min   # 0,5 es el AP50; 0,3 tolera cajas de otra extensión
    import preview as pv
    from ultralytics import YOLO
    muestra, cajas = leer_juicios()
    ids = [i for i in sorted(cajas) if parte == "todos" or muestra[i]["parte"] == parte]
    imagenes = {}
    for i in ids:
        f = muestra[i]
        imagenes[i] = cv2.imread(str(DATASET / "images" / f["parte"] / f"{f['video'][:6]}_{int(f['cuadro']):05d}.jpg"))
    n_gt = sum(len(cajas[i]) for i in ids)
    resultados = {}
    for clave, pesos in modelos.items():
        m = YOLO(str(pesos), task="detect")
        bache = {k for k, nombre in m.names.items() if pv.codigo_det(nombre) == "D40"}
        registros = []
        for i in ids:
            r = m.predict(imagenes[i], imgsz=imgsz, conf=0.001, max_det=300, verbose=False)[0].boxes
            preds = [(c, caja) for caja, c, k in zip(r.xyxy.tolist(), r.conf.tolist(), r.cls.tolist()) if int(k) in bache]
            registros += ev.emparejar(preds, cajas[i])
        resultados[clave] = round(ev.ap50(registros, n_gt), 4)
        print(f"  {clave:28s} AP (IoU >= {iou_min}) baches = {resultados[clave]:.3f}", flush=True)
    sufijo = "" if iou_min == 0.5 else f"_iou{round(iou_min * 100):02d}"
    archivo = CARPETA / f"eval_{parte}_{imgsz}{sufijo}.json"   # se suman los modelos nuevos a los ya medidos
    previos = json.loads(archivo.read_text(encoding="utf-8"))["ap50"] if archivo.exists() else {}
    salida = {"parte": parte, "imgsz": imgsz, "iou": iou_min, "cuadros": len(ids), "baches": n_gt,
              "ap50": {**previos, **resultados}}
    archivo.write_text(json.dumps(salida, indent=1), encoding="utf-8")
    print(json.dumps(salida))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paso", choices=["muestrear", "etiquetas", "evaluar"])
    ap.add_argument("modelos", nargs="*", help="para evaluar: nombre=pesos.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--parte", default="prueba", choices=["ajuste", "prueba", "todos"])
    ap.add_argument("--iou", type=float, default=0.5, help="superposición mínima para un acierto (0,5 = AP50)")
    a = ap.parse_args()
    if a.paso == "muestrear":
        muestrear()
    elif a.paso == "etiquetas":
        etiquetas()
    else:
        evaluar(dict(m.split("=", 1) for m in a.modelos), a.imgsz, a.parte, a.iou)


if __name__ == "__main__":
    main()
