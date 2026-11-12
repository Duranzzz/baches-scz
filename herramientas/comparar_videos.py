"""Compara detectores de baches y grietas sobre videos sin etiquetas.

    python herramientas/comparar_videos.py videos/scz_02_frontal_avenida.mp4 videos/scz_03_frontal_doble_via.mp4
    python herramientas/comparar_videos.py videos/scz_*.mp4 --salida salidas/videos/comparacion_20261121
    python herramientas/comparar_videos.py VIDEO --modelo a1=runs/a1/weights/best.pt     # sumar o cambiar un modelo
    python herramientas/comparar_videos.py VIDEO --max-cuadros 60                         # prueba corta

En dos máquinas (la inferencia en una GPU, el dibujo donde están los videos):
    python herramientas/comparar_videos.py videos/*.mp4 --solo rdd_yolo26s --modelo rdd_yolo26s=RUTA.pt --conf 0.10 \\
        --device 0 --solo-detectar --salida DET            # deja DET/<video>/<modelo>.csv y .json
    python herramientas/comparar_videos.py videos/*.mp4 --solo yolo12s_rdd2022,rdd_yolo26s --reusar DET \\
        --reusar salidas/videos/comparacion_20261121 --conf-modelo rdd_yolo26s=0.15 --salida SALIDA
--reusar toma las detecciones ya guardadas de un modelo si se hicieron al mismo tamaño y con una confianza
mínima menor o igual a la suya (--conf-modelo, o --conf); así no se vuelve a correr el modelo.

Por cada video deja en --salida/<video>/:
  <modelo>.mp4               el video con las marcas de ese modelo solo
  comparacion_N_paneles.mp4  original, los modelos y el acumulado, cuadro a cuadro
  <modelo>.csv               una fila por detección (cuadro, segundo, clase, confianza, caja)
  capturas/                  los paneles cada 10 s y la marca de bache más segura de cada modelo
  resumen.json
y en --salida/RESULTADOS.md y resultados.csv, la tabla de todos los videos procesados.

Sin etiquetas no hay aciertos: se cuentan cuadros con marca y "eventos" (tramos de 3 cuadros o más con
marca, uniendo huecos de hasta 0,5 s; un bache que se ve 2 s es un evento). Más marcas no es mejor: pueden
ser falsos positivos. Las confianzas de modelos distintos no están calibradas entre sí.
Si existe <--onnx>/<clave>.onnx (exportado al mismo --imgsz) se usa ese: en CPU es más rápido.
"""
import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import preview as pv

RAIZ = Path(__file__).resolve().parents[1]
MODELOS = {  # clave: (título en el video, pesos). Otros modelos, para comparar: --modelo clave=pesos.pt
    "baches_scz_v1": ("Baches SCZ v1 (baches y grietas)", RAIZ / "modelos" / "baches_scz_v1.pt"),
}
NOMBRE = {"D00": "grieta_longitudinal", "D10": "grieta_transversal", "D20": "grieta_cocodrilo", "D40": "bache"}
GRIETAS = ("grieta_longitudinal", "grieta_transversal", "grieta_cocodrilo")
ETIQUETA = {"bache": "bache", "grieta_longitudinal": "g.long", "grieta_transversal": "g.trans",
            "grieta_cocodrilo": "g.cocod", "parche": "parche"}
COLOR = {"bache": (0, 0, 230), "grieta_longitudinal": (0, 140, 255), "grieta_transversal": (230, 200, 0),
         "grieta_cocodrilo": (220, 0, 220), "parche": (150, 150, 150)}  # BGR
FONT = cv2.FONT_HERSHEY_SIMPLEX
CAB = 42            # alto de la cabecera de cada panel (fuera de la imagen)
HUECO_S = 0.5       # huecos más cortos que esto no cortan un evento
MIN_CUADROS = 3     # un evento dura al menos esto


def clase(nombre):
    c = pv.codigo_det(nombre)
    if c:
        return NOMBRE[c]
    return "parche" if nombre.lower().startswith("repair") else None


def info_video(ruta):
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream=width,height,avg_frame_rate,nb_frames", "-of", "json", str(ruta)],
                             capture_output=True, text=True, check=True)
    except FileNotFoundError:   # sin ffprobe (p. ej. la máquina de GPU): lo que dice OpenCV
        cap = cv2.VideoCapture(str(ruta))
        fps = cap.get(cv2.CAP_PROP_FPS)
        info = {"ancho": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "alto": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps_texto": f"{fps:.3f}", "fps": fps, "cuadros": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
        cap.release()
        return info
    s = json.loads(out.stdout)["streams"][0]
    num, den = (int(x) for x in s["avg_frame_rate"].split("/"))
    return {"ancho": s["width"], "alto": s["height"], "fps_texto": s["avg_frame_rate"], "fps": num / den,
            "cuadros": int(s.get("nb_frames", 0))}


def escritor(ruta, ancho, alto, fps_texto):
    """ffmpeg que recibe cuadros BGR por stdin y guarda H.264, sin audio."""
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ancho}x{alto}",
           "-r", fps_texto, "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(ruta)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def detectar(modelo, img, args, conf):
    r = modelo.predict(img, imgsz=args.imgsz, conf=conf, iou=args.iou, device=args.device, verbose=False)[0]
    polys = r.masks.xy if r.masks is not None else [None] * len(r.boxes)
    dets = []
    for caja, c, k, poly in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.int().tolist(), polys):
        cl = clase(modelo.names[k])
        if cl:
            dets.append({"clase": cl, "original": modelo.names[k], "conf": c, "caja": caja, "poly": poly})
    return dets


def texto(img, txt, xy, escala=0.5, color=(255, 255, 255)):
    cv2.putText(img, txt, xy, FONT, escala, color, 1, cv2.LINE_AA)


def panel(img, dets, titulo, linea2):
    out = img.copy()
    con_mascara = [d for d in dets if d["poly"] is not None and len(d["poly"]) >= 3]
    if con_mascara:
        capa = out.copy()
        for d in con_mascara:
            cv2.fillPoly(capa, [d["poly"].astype(np.int32)], COLOR[d["clase"]])
        out = cv2.addWeighted(capa, 0.4, out, 0.6, 0)
    for d in dets:
        x1, y1, x2, y2 = (int(round(v)) for v in d["caja"])
        col = COLOR[d["clase"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        txt = f"{ETIQUETA[d['clase']]} {d['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, FONT, 0.5, 1)
        ty = max(th + 4, y1)
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 4, ty), col, -1)
        claro = 0.114 * col[0] + 0.587 * col[1] + 0.299 * col[2] > 140
        texto(out, txt, (x1 + 2, ty - 3), color=(0, 0, 0) if claro else (255, 255, 255))
    cab = np.full((CAB, out.shape[1], 3), 32, np.uint8)
    texto(cab, titulo, (8, 17), 0.55)
    texto(cab, linea2, (8, 35), 0.48, (205, 205, 205))
    return np.vstack([cab, out])


def eventos(marcas, fps):
    """Tramos de cuadros con marca, uniendo huecos cortos; cuenta los que duran MIN_CUADROS o más."""
    hueco = max(1, round(HUECO_S * fps))
    n, inicio, ultimo = 0, None, None
    for i, m in enumerate(marcas):
        if not m:
            continue
        if inicio is None or i - ultimo > hueco:
            if inicio is not None and ultimo - inicio + 1 >= MIN_CUADROS:
                n += 1
            inicio = i
        ultimo = i
    if inicio is not None and ultimo - inicio + 1 >= MIN_CUADROS:
        n += 1
    return n


def mmss(s):
    return f"{int(s // 60):02d}:{int(s % 60):02d}"


CAMPOS_CSV = ["cuadro", "segundo", "clase", "clase_original", "confianza", "x1", "y1", "x2", "y2"]


def pasa(d, m):
    """¿La detección supera el umbral de su clase en este modelo? (bache y grietas pueden tener umbrales distintos)."""
    u = m["umbral"]
    return d["conf"] >= (u["bache"] if d["clase"] == "bache" else u["grieta"] if d["clase"] in GRIETAS else m["conf"])


def texto_umbral(m):
    u = m["umbral"]
    return f"conf >= {u['bache']}" if u["bache"] == u["grieta"] else f"bache >= {u['bache']}, grieta >= {u['grieta']}"


def fila_csv(i, fps, d):
    return [i, round(i / fps, 3), d["clase"], d["original"], round(d["conf"], 4), *(round(v, 1) for v in d["caja"])]


def guardadas(video, k, m, total, args):
    """Detecciones ya guardadas de este modelo en este video (--reusar), si sirven: mismo tamaño y NMS, todos los
    cuadros y una confianza mínima menor o igual a la de ahora. Devuelve (detecciones por cuadro, metadatos, csv)."""
    for carpeta in args.reusar:
        d = carpeta / video.stem
        ruta_csv, ruta_meta, ruta_resumen = d / f"{k}.csv", d / f"{k}.json", d / "resumen.json"
        if not ruta_csv.exists():
            continue
        if ruta_meta.exists():
            meta = json.loads(ruta_meta.read_text(encoding="utf-8"))
        elif ruta_resumen.exists() and k in json.loads(ruta_resumen.read_text(encoding="utf-8"))["modelos"]:
            r = json.loads(ruta_resumen.read_text(encoding="utf-8"))
            mr = r["modelos"][k]
            meta = {"imgsz": r["imgsz"], "iou": r["iou"], "conf": mr.get("conf", r["conf"]), "cuadros": r["cuadros"],
                    "s_por_cuadro": mr["s_por_cuadro"], "dispositivo": mr.get("dispositivo", "cpu")}
        else:
            continue
        if (list(meta["imgsz"]) != list(args.imgsz) or meta["iou"] != args.iou or meta["conf"] > m["conf"] + 1e-9
                or meta["cuadros"] < total):
            print(f"  {k}: {ruta_csv} no sirve (otro tamaño, NMS o cuadros, o confianza mínima {meta['conf']} > {m['conf']})")
            continue
        dets, grietas = [[] for _ in range(total)], False
        with open(ruta_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                i, c = int(row["cuadro"]), float(row["confianza"])
                grietas |= row["clase"] in GRIETAS
                d = {"clase": row["clase"], "original": row["clase_original"], "conf": c,
                     "caja": [float(row[x]) for x in ("x1", "y1", "x2", "y2")], "poly": None}
                if i < total and pasa(d, m):
                    dets[i].append(d)
        meta.setdefault("grietas", grietas)
        return dets, meta, ruta_csv
    return None


def detectar_todo(video, modelos, total, fps, destino, args):
    """Pasada 1: solo inferencia, un modelo a la vez sobre todo el video. Sin codificar video y sin
    intercalar modelos: varias sesiones de ONNX Runtime alternándose cuadro a cuadro se pisan los hilos
    y van 4 veces más lento. Un modelo con detecciones guardadas que sirven (--reusar) no se vuelve a correr."""
    dets_de, segundos, n = {}, {}, total
    for k, m in modelos.items():
        previas = guardadas(video, k, m, total, args)
        if previas:
            dets_de[k], meta, origen = previas
            segundos[k] = meta.get("s_por_cuadro", 0.0) * total
            m.update(dispositivo=meta.get("dispositivo", "cpu"), grietas=meta["grietas"])
            if origen.resolve() != (destino / f"{k}.csv").resolve():
                with open(destino / f"{k}.csv", "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(CAMPOS_CSV)
                    w.writerows(fila_csv(i, fps, d) for i, ds in enumerate(dets_de[k]) for d in ds)
            print(f"  {video.name} / {k}: detecciones guardadas en {origen} (confianza >= {m['conf']})", flush=True)
            continue
        if m.get("modelo") is None:
            m["modelo"] = YOLO(str(m["ruta"]), task=None)
            m["grietas"] = any(clase(nombre) in GRIETAS for nombre in m["modelo"].names.values())
        m["dispositivo"] = "cpu" if str(args.device) == "cpu" else f"gpu {args.device}"
        dets_de[k], segundos[k] = [], 0.0
        cap = cv2.VideoCapture(str(video))
        t0, i = time.time(), 0
        with open(destino / f"{k}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(CAMPOS_CSV)
            while i < total:
                ok, img = cap.read()
                if not ok:
                    break
                t = time.time()
                dets = detectar(m["modelo"], img, args, m["conf"])
                segundos[k] += time.time() - t
                # El CSV guarda todo lo que pasa la confianza mínima (sirve para --reusar con otros umbrales);
                # el video, solo lo que pasa el umbral de cada clase.
                dets_de[k].append([d for d in dets if pasa(d, m)])
                w.writerows(fila_csv(i, fps, d) for d in dets)
                i += 1
                if i % 500 == 0:
                    v = i / (time.time() - t0)
                    print(f"  {video.name} / {k}: {i}/{total}, {v:.1f} cuadros por s, faltan {mmss((total - i) / v)}",
                          flush=True)
        cap.release()
        n = min(n, i)
        (destino / f"{k}.json").write_text(json.dumps({
            "imgsz": list(args.imgsz), "iou": args.iou, "conf": m["conf"], "cuadros": i,
            "s_por_cuadro": round(segundos[k] / max(i, 1), 4), "dispositivo": m["dispositivo"],
            "grietas": m["grietas"], "pesos": str(m["ruta"])}, indent=1), encoding="utf-8")
    return {k: v[:n] for k, v in dets_de.items()}, segundos, n


def procesar(video, modelos, args, destino):
    info = info_video(video)
    destino.mkdir(parents=True, exist_ok=True)
    (destino / "capturas").mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    ancho, alto, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), info["fps"]
    total = info["cuadros"] or int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if args.max_cuadros:
        total = min(total, args.max_cuadros)
    claves = list(modelos)
    t0 = time.time()
    dets_de, segundos, n = detectar_todo(video, modelos, total, fps, destino, args)
    total = n
    if args.solo_detectar:
        print(f"  {video.name}: detecciones listas en {mmss(time.time() - t0)}", flush=True)
        return None

    marcas = {k: {"bache": [any(d["clase"] == "bache" for d in ds) for ds in dets_de[k]],
                  "grieta": [any(d["clase"] in GRIETAS for d in ds) for ds in dets_de[k]],
                  "alguna": [bool(ds) for ds in dets_de[k]]} for k in claves}
    acum = {k: {t: np.cumsum(v) for t, v in m.items()} for k, m in marcas.items()}
    obs = {k: {c: sum(d["clase"] == c for ds in dets_de[k] for d in ds) for c in ETIQUETA} for k in claves}
    conf_max = {k: {c: max((d["conf"] for ds in dets_de[k] for d in ds if d["clase"] == c), default=0.0)
                    for c in ETIQUETA} for k in claves}
    mejor = {}
    for k in claves:
        confs = [max((d["conf"] for d in ds if d["clase"] == "bache"), default=0.0) for ds in dets_de[k]]
        if confs and max(confs) > 0:
            mejor[k] = int(np.argmax(confs))

    # Pasada 2: dibujar y codificar
    alto_panel = alto + CAB
    columnas = 3
    filas_grilla = -(-(len(claves) + 2) // columnas)
    escritores = {k: escritor(destino / f"{k}.mp4", ancho, alto_panel, info["fps_texto"]) for k in claves}
    escritores["_grilla"] = escritor(destino / f"comparacion_{len(claves) + 2}_paneles.mp4", ancho * columnas,
                                     alto_panel * filas_grilla, info["fps_texto"])
    cada_captura = max(1, round(10 * fps))
    cap = cv2.VideoCapture(str(video))
    try:
        for i in range(total):
            ok, img = cap.read()
            if not ok:
                break
            seg = i / fps
            paneles = [panel(img, [], "Original", f"{video.name}  {mmss(seg)}  cuadro {i}")]
            for k in claves:
                dets = dets_de[k][i]
                n_b = sum(d["clase"] == "bache" for d in dets)
                n_g = sum(d["clase"] in GRIETAS for d in dets)
                linea2 = f"este cuadro: {n_b} bache" + (f", {n_g} grieta" if modelos[k]["grietas"] else "")
                p = panel(img, dets, f"{modelos[k]['titulo']}  {texto_umbral(modelos[k])}", linea2)
                paneles.append(p)
                escritores[k].stdin.write(p.tobytes())
                if mejor.get(k) == i:
                    b = max(d["conf"] for d in dets if d["clase"] == "bache")
                    cv2.imwrite(str(destino / "capturas" / f"mejor_bache_{k}_cuadro{i}_{b:.2f}.jpg"), p,
                                [cv2.IMWRITE_JPEG_QUALITY, 90])
            res = np.full((alto_panel, ancho, 3), 24, np.uint8)
            texto(res, f"Acumulado {mmss(seg)} de {mmss(total / fps)}  (cuadro {i + 1} de {total})", (12, 30), 0.6)
            y = 70
            for k in claves:
                cb, cg = int(acum[k]["bache"][i]), int(acum[k]["grieta"][i])
                txt = f"cuadros con bache: {cb}" + (f"   con grieta: {cg}" if modelos[k]["grietas"] else "")
                texto(res, modelos[k]["titulo"], (12, y), 0.5, (200, 200, 200))
                texto(res, txt, (24, y + 22), 0.55)
                y += 58
            x = 12
            for c, nombre in ETIQUETA.items():
                cv2.rectangle(res, (x, y + 2), (x + 16, y + 18), COLOR[c], -1)
                texto(res, nombre, (x + 22, y + 15), 0.5)
                x += 34 + cv2.getTextSize(nombre, FONT, 0.5, 1)[0][0]
            texto(res, "Sin etiquetas: son marcas, no aciertos.", (12, y + 50), 0.5, (170, 170, 170))
            texto(res, f"{args.imgsz[1]}x{args.imgsz[0]} px; confianza minima: la de cada panel", (12, y + 74), 0.5,
                  (170, 170, 170))
            paneles.append(res)
            while len(paneles) < columnas * filas_grilla:
                paneles.append(np.zeros_like(res))
            grilla = np.vstack([np.hstack(paneles[f * columnas:(f + 1) * columnas]) for f in range(filas_grilla)])
            escritores["_grilla"].stdin.write(grilla.tobytes())
            if i % cada_captura == 0:
                cv2.imwrite(str(destino / "capturas" / f"{i:05d}_{seg:05.1f}s.jpg"), grilla,
                            [cv2.IMWRITE_JPEG_QUALITY, 85])
    finally:
        cap.release()
        for e in escritores.values():
            e.stdin.close()
        for e in escritores.values():
            e.wait()
    i = total

    resumen = {"video": video.name, "cuadros": i, "fps": round(fps, 3), "segundos": round(i / fps, 1),
               "tamano": f"{ancho}x{alto}", "imgsz": list(args.imgsz), "conf": args.conf, "iou": args.iou,
               "modelos": {}}
    for k in claves:
        m = marcas[k]
        resumen["modelos"][k] = {
            "titulo": modelos[k]["titulo"], "pesos": str(modelos[k]["ruta"]), "conf": modelos[k]["conf"],
            "umbral": modelos[k]["umbral"],
            "grietas": modelos[k]["grietas"], "dispositivo": modelos[k]["dispositivo"],
            "cuadros_con_marca": sum(m["alguna"]), "cuadros_con_bache": sum(m["bache"]),
            "cuadros_con_grieta": sum(m["grieta"]), "eventos_bache": eventos(m["bache"], fps),
            "eventos_grieta": eventos(m["grieta"], fps), "observaciones": obs[k],
            "conf_max": {c: round(v, 3) for c, v in conf_max[k].items() if v},
            "s_por_cuadro": round(segundos[k] / max(i, 1), 3)}
    (destino / "resumen.json").write_text(json.dumps(resumen, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  {video.name}: listo en {mmss(time.time() - t0)}", flush=True)
    return resumen


def pct(a, b):
    return f"{100 * a / b:.0f} %" if b else "-"


def informe(salida, args):
    resumenes = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(salida.glob("*/resumen.json"))]
    if not resumenes:
        return
    claves = list(resumenes[0]["modelos"])
    m0 = resumenes[0]["modelos"]
    def umbral(m):
        u = m.get("umbral") or {"bache": m.get("conf", resumenes[0]["conf"]), "grieta": m.get("conf", resumenes[0]["conf"])}
        return f"conf ≥ {u['bache']}" if u["bache"] == u["grieta"] else f"bache ≥ {u['bache']}, grieta ≥ {u['grieta']}"

    titulo = {k: f"{m0[k]['titulo']} ({umbral(m0[k])})" for k in claves}
    con_grietas = [k for k in claves if m0[k].get("grietas", k == "yolo12s_rdd2022")]
    with open(salida / "resultados.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "modelo", "cuadros", "cuadros_con_bache", "eventos_bache", "cuadros_con_grieta",
                    "eventos_grieta", "conf_max_bache", "s_por_cuadro"])
        for r in resumenes:
            for k, m in r["modelos"].items():
                w.writerow([r["video"], k, r["cuadros"], m["cuadros_con_bache"], m["eventos_bache"],
                            m["cuadros_con_grieta"], m["eventos_grieta"], m["conf_max"].get("bache", 0),
                            m["s_por_cuadro"]])
    l = [f"# Modelos sobre videos de Santa Cruz ({time.strftime('%Y-%m-%d')})", "",
         f"Todos los cuadros, a {args.imgsz[1]}x{args.imgsz[0]} px (el tamaño de estos videos, sin agrandarlos: en la "
         f"línea base, agrandar imágenes chicas empeoró), con la confianza mínima de cada modelo (entre paréntesis; "
         f"la de la app, elegida en val) y NMS {args.iou}.",
         "Sin etiquetas: son marcas, no aciertos. Más marcas puede ser más falsos positivos. Un **evento** es un "
         f"tramo de {MIN_CUADROS} cuadros o más con marca (huecos de hasta {HUECO_S} s se unen): se parece más a "
         "\"un daño visto\" que contar cuadros.", "",
         "## Baches: % de cuadros con marca (eventos)", "",
         "| Modelo | " + " | ".join(r["video"].replace(".mp4", "") for r in resumenes) + " |",
         "|---|" + "---:|" * len(resumenes)]
    for k in claves:
        l.append(f"| {titulo[k]} | " + " | ".join(
            f"{pct(r['modelos'][k]['cuadros_con_bache'], r['cuadros'])} ({r['modelos'][k]['eventos_bache']})"
            for r in resumenes) + " |")
    if con_grietas:
        l += ["", "## Grietas (modelos que las detectan): % de cuadros con marca (eventos)", "",
              "| Modelo | " + " | ".join(r["video"].replace(".mp4", "") for r in resumenes) + " |",
              "|---|" + "---:|" * len(resumenes)]
        for k in con_grietas:
            l.append(f"| {titulo[k]} | " + " | ".join(
                f"{pct(r['modelos'][k]['cuadros_con_grieta'], r['cuadros'])} ({r['modelos'][k]['eventos_grieta']})"
                for r in resumenes) + " |")
    l += ["", "## Por video", ""]
    for r in resumenes:
        l += [f"### {r['video']} ({r['segundos']} s, {r['cuadros']} cuadros, {r['tamano']})", "",
              "| Modelo | Cuadros con bache | Eventos de bache | Conf. máx. bache | Cuadros con grieta | s por cuadro |",
              "|---|---:|---:|---:|---:|---:|"]
        for k, m in r["modelos"].items():
            l.append(f"| {m['titulo']} | {m['cuadros_con_bache']} ({pct(m['cuadros_con_bache'], r['cuadros'])}) | "
                     f"{m['eventos_bache']} | {m['conf_max'].get('bache', 0):.2f} | "
                     f"{m['cuadros_con_grieta']} | {m['s_por_cuadro']:.2f} ({m.get('dispositivo', 'cpu')}) |")
        l.append("")
    (salida / "RESULTADOS.md").write_text("\n".join(l) + "\n", encoding="utf-8")
    print(f"tabla en {salida / 'RESULTADOS.md'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--salida", type=Path, default=RAIZ / "salidas" / "videos" / f"comparacion_{time.strftime('%Y%m%d')}")
    ap.add_argument("--modelo", action="append", default=[], help="clave=pesos.pt (agrega o reemplaza)")
    ap.add_argument("--imgsz", default="512,864", help="alto,ancho de análisis (múltiplos de 32)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--onnx", type=Path, default=RAIZ / "salidas" / "onnx_864",
                    help="carpeta con <clave>.onnx exportados a --imgsz; si no está, se usa el .pt")
    ap.add_argument("--max-cuadros", type=int, default=0, help="solo los primeros N cuadros (para probar)")
    ap.add_argument("--rehacer", action="store_true", help="volver a procesar videos que ya tienen resumen.json")
    ap.add_argument("--solo", help="claves de los modelos a usar, separadas por comas, en ese orden (por defecto, todos)")
    ap.add_argument("--conf-modelo", action="append", default=[],
                    help="clave=confianza mínima de ese modelo, o clave=bache:grieta para umbrales distintos")
    ap.add_argument("--device", default="cpu", help="cpu, o 0 para la primera GPU")
    ap.add_argument("--solo-detectar", action="store_true",
                    help="solo la inferencia: deja <modelo>.csv y <modelo>.json por video, sin dibujar videos")
    ap.add_argument("--reusar", action="append", default=[], type=Path,
                    help="carpeta de una salida anterior: usa sus detecciones si sirven (ver arriba)")
    args = ap.parse_args()
    args.imgsz = tuple(int(v) for v in args.imgsz.split(","))
    if str(args.device) != "cpu":
        args.device = int(args.device)

    elegidos = {k: (t, p) for k, (t, p) in MODELOS.items()}
    for m in args.modelo:
        k, ruta = m.split("=", 1)
        elegidos[k] = (elegidos.get(k, (k, None))[0], Path(ruta))
    if args.solo:
        faltan = [k for k in args.solo.split(",") if k not in elegidos]
        if faltan:
            raise SystemExit(f"modelos desconocidos: {faltan} (agregarlos con --modelo clave=pesos.pt)")
        elegidos = {k: elegidos[k] for k in args.solo.split(",")}
    umbrales = {}
    for c in args.conf_modelo:
        k, v = c.split("=", 1)
        b, _, g = v.partition(":")
        umbrales[k] = {"bache": float(b), "grieta": float(g or b)}
    modelos = {}
    for k, (titulo, ruta) in elegidos.items():
        onnx = args.onnx / f"{k}.onnx"
        usar = onnx if onnx.exists() and not any(m.startswith(k + "=") for m in args.modelo) else ruta
        u = umbrales.get(k, {"bache": args.conf, "grieta": args.conf})
        # El modelo se abre recién si hay que correrlo (con --reusar puede no hacer falta)
        modelos[k] = {"titulo": titulo, "ruta": usar, "modelo": None, "umbral": u, "conf": min(u.values())}
        print(f"{k}: {usar} ({texto_umbral(modelos[k])})")

    for video in args.videos:
        destino = args.salida / video.stem
        if (destino / "resumen.json").exists() and not args.rehacer and not args.max_cuadros:
            print(f"{video.name}: ya estaba ({destino / 'resumen.json'})")
            continue
        destino.mkdir(parents=True, exist_ok=True)
        print(f"{video.name} ...", flush=True)
        procesar(video, modelos, args, destino)
    if not args.solo_detectar:
        informe(args.salida, args)


if __name__ == "__main__":
    main()
