"""Figuras del artículo (además de figura1_wise.py). Todos los datos salen de informes/, splits.csv, etiquetas/ y
los videos locales.
    python docs/figuras/generar_figuras.py"""
import csv
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
AZUL, NARANJA, AGUA, GRIS = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a8a"
TINTA = "#222222"
coma = FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ","))
plt.rcParams.update({"font.family": "Arial", "font.size": 8.5, "axes.edgecolor": GRIS, "axes.linewidth": 0.6,
                     "xtick.color": TINTA, "ytick.color": TINTA, "text.color": TINTA, "axes.labelcolor": TINTA})


def guardar(fig, nombre):
    fig.savefig(OUT / nombre, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(OUT / nombre)


def ap(nombre, split, lado, dataset, clase="D40"):
    r = json.loads((RAIZ / "informes" / f"eval_{nombre}_{split}_{lado}.json").read_text(encoding="utf-8"))
    return next(f["AP50_det"] for f in r["filas"] if f["dataset"] == dataset and f["clase"] == clase)


# 1) Muestras de las fuentes con sus etiquetas
def muestras():
    fotos = [("brazil_DF_00059", "Brasil, desde vehículo"), ("bucko_clear_00297", "Bucko (Eslovaquia), cámara de tablero"),
             ("egypt_01397", "Egipto, celular al frente"), ("hrp4k_30009", "HRP4K, 4K en perspectiva")]
    paneles = []
    for foto, titulo in fotos:
        img = cv2.imread(str(RAIZ / "fotos" / f"{foto}.jpg"))
        h, w = img.shape[:2]
        grosor = max(2, round(max(h, w) / 400))
        for linea in (RAIZ / "etiquetas" / f"{foto}.txt").read_text().splitlines():
            p = linea.split()
            if not p or p[0] not in ("0", "1", "2", "3", "4"):
                continue
            cx, cy, bw, bh = (float(v) for v in p[1:5])
            x1, y1, x2, y2 = int((cx - bw / 2) * w), int((cy - bh / 2) * h), int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            color = (54, 104, 235) if p[0] == "3" else (214, 120, 42)   # BGR: bache naranja, grieta azul
            cv2.rectangle(img, (x1, y1), (x2, y2), color, grosor)
        alto = 420
        img = cv2.resize(img, (round(w * alto / h), alto), interpolation=cv2.INTER_AREA)
        paneles.append((cv2.cvtColor(img, cv2.COLOR_BGR2RGB), titulo))
    fig, ejes = plt.subplots(2, 2, figsize=(6.3, 4.1))
    for ax, (img, titulo) in zip(ejes.flat, paneles):
        ax.imshow(img)
        ax.set_title(titulo, fontsize=8.5, loc="left")
        ax.axis("off")
    fig.tight_layout(h_pad=0.8, w_pad=0.6)
    guardar(fig, "figura_muestras.png")


# 2) Curvas de entrenamiento (validación de ultralytics, con la fuga de val)
def curvas():
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    for nombre, etiqueta, color, linea in (("a1_y26s", "Corrida 1: 6.507 fotografías", AZUL, "-"),
                                           ("a2_y26s_hrp4k", "Corrida 2: + HRP4K (9.693)", NARANJA, "--")):
        filas = list(csv.DictReader(open(RAIZ / "informes" / "corridas" / nombre / "results.csv", encoding="utf-8")))
        filas = [{k.strip(): v for k, v in f.items()} for f in filas]
        ep = [int(float(f["epoch"])) for f in filas]
        m = [float(f["metrics/mAP50(B)"]) for f in filas]
        ax.plot(ep, m, color=color, lw=1.4, ls=linea, label=etiqueta)
        i = int(np.argmax(m))
        ax.plot(ep[i], m[i], "o", color=color, ms=5)
        ax.annotate(f"máximo {m[i]:.3f}".replace(".", ","), (ep[i], m[i]), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=7, color="#555555")
    ax.set_xlabel("Época")
    ax.set_ylabel("mAP50 de baches en validación")
    ax.yaxis.set_major_formatter(coma)
    ax.set_ylim(0.5, 0.82)
    ax.grid(axis="y", color="#e2e2e0", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    guardar(fig, "figura_curvas.png")


# 3) AP50 por conjunto: detector de referencia, modelo inicial y modelo final
def barras():
    ds = ["bisman", "brazil", "bucko", "egypt", "hrp4k", "todos"]
    nombres = ["Bisman", "Brasil", "Bucko", "Egipto", "HRP4K", "Total"]
    series = [("Detector de referencia", "yolo12s_RDD2022_best", AGUA),
              ("Modelo inicial", "rdd_yolo26s_dist", NARANJA),
              ("Modelo final (aplicación)", "wise4_a25", AZUL)]
    fig, ax = plt.subplots(figsize=(6.3, 2.8))
    x = np.arange(len(ds))
    ancho = 0.26
    for i, (etiqueta, nombre, color) in enumerate(series):
        vals = [ap(nombre, "test", 640, d) for d in ds]
        barras_ = ax.bar(x + (i - 1) * (ancho + 0.02), vals, ancho, color=color, label=etiqueta, zorder=3)
        for b, v in zip(barras_, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}".replace(".", ","), ha="center", va="bottom",
                    fontsize=6.2, color=TINTA)
    ax.set_xticks(x)
    ax.set_xticklabels(nombres)
    ax.set_ylabel("AP50 de baches (prueba, 640 px)")
    ax.yaxis.set_major_formatter(coma)
    ax.set_ylim(0, 0.72)
    ax.grid(axis="y", color="#e2e2e0", lw=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right", ncol=3)
    guardar(fig, "figura_barras.png")


# 4) F1 en validación según el umbral de confianza (modelo final, solo detector, 640 px)
def umbral():
    conf = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
    baches = [0.365, 0.338, 0.329, 0.307, 0.294, 0.277, 0.242]
    grietas = [0.156, 0.149, 0.149, 0.144, 0.142, 0.125, 0.121]
    fig, ax = plt.subplots(figsize=(6.3, 2.5))
    ax.plot(conf, baches, color=AZUL, lw=1.6, marker="o", ms=5, label="Baches")
    ax.plot(conf, grietas, color=NARANJA, lw=1.6, ls="--", marker="s", ms=5, label="Grietas")
    for c, v, color in ((0.10, 0.338, AZUL), (0.25, 0.142, NARANJA)):
        ax.plot(c, v, "o", ms=11, mfc="none", mec=color, mew=1.4)
        ax.annotate(f"umbral de la aplicación: {c:.2f}".replace(".", ","), (c, v), xytext=(8, 10), textcoords="offset points",
                    fontsize=7, color="#555555")
    ax.set_xlabel("Umbral de confianza")
    ax.set_ylabel("F1 medio entre conjuntos (validación)")
    ax.xaxis.set_major_formatter(coma)
    ax.yaxis.set_major_formatter(coma)
    ax.set_ylim(0.08, 0.42)
    ax.grid(axis="y", color="#e2e2e0", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    guardar(fig, "figura_umbral.png")


# 5) Esquema de la aplicación
def arquitectura():
    fig, ax = plt.subplots(figsize=(6.3, 2.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis("off")
    cajas = {
        "camara": (1, 30, 21, 11, "Cámara o video\ndel teléfono"),
        "cuadro": (26, 30, 21, 11, "Cuadro escalado\na 640 px"),
        "modelo": (51, 30, 21, 11, "Modelo ONNX\n(GPU o CPU)"),
        "dets": (76, 30, 21, 11, "Cajas de baches\ny grietas"),
        "confirma": (76, 12, 21, 11, "¿Aparece en 2\nde 3 cuadros?"),
        "punto": (51, 12, 21, 11, "Punto con posición\n(GPS o recorrido)"),
        "mapa": (26, 12, 21, 11, "Mapa de\nOpenStreetMap"),
        "guardar": (1, 12, 21, 11, "Guardado en el\nteléfono; se exporta"),
    }
    for clave, (x, y, w, h, texto) in cajas.items():
        color = "#e8f0fb" if clave in ("modelo", "dets") else "#f2f2f0"
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=color, ec=GRIS, lw=0.8))
        ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=7.4)

    def flecha(a, b, de="r", a_lado="l"):
        xa, ya, wa, ha_ = cajas[a][:4]
        xb, yb, wb, hb = cajas[b][:4]
        p1 = {"r": (xa + wa + 0.6, ya + ha_ / 2), "l": (xa - 0.6, ya + ha_ / 2), "b": (xa + wa / 2, ya - 0.6)}[de]
        p2 = {"l": (xb - 0.6, yb + hb / 2), "r": (xb + wb + 0.6, yb + hb / 2), "t": (xb + wb / 2, yb + hb + 0.6)}[a_lado]
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=9, color="#555555", lw=0.9))

    flecha("camara", "cuadro")
    flecha("cuadro", "modelo")
    flecha("modelo", "dets")
    flecha("dets", "confirma", "b", "t")
    flecha("confirma", "punto", "l", "r")
    flecha("punto", "mapa", "l", "r")
    flecha("mapa", "guardar", "l", "r")
    ax.text(50, 44, "Todo ocurre en el teléfono: no se envía video a ningún servidor", ha="center", fontsize=7.6,
            color="#555555")
    guardar(fig, "figura_arquitectura.png")


# 6) Recortes de baches detectados por el modelo final en los videos de Santa Cruz (solo el pavimento)
def videos():
    casos = [("scz_01_lateral_auto_hormigon", 1060, (265.6, 377.8, 387.2, 443.6), 0.70, 2.4),
             ("scz_02_frontal_avenida", 830, (380.1, 275.1, 470.2, 313.5), 0.62, 1.15),   # sin la moto de al lado
             ("scz_03_frontal_doble_via", 809, (530.6, 397.8, 578.1, 429.6), 0.56, 2.4)]
    recortes = []
    for video, cuadro, (x1, y1, x2, y2), conf, factor in casos:
        cap = cv2.VideoCapture(str(RAIZ / "videos" / f"{video}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, cuadro)
        ok, img = cap.read()
        cap.release()
        if not ok:
            raise SystemExit(f"no pude leer {video} cuadro {cuadro}")
        h, w = img.shape[:2]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        lado = max(x2 - x1, y2 - y1) * factor
        a, b = int(max(0, cx - lado)), int(min(w, cx + lado))
        c, d = int(max(0, cy - lado * 0.55)), int(min(h, cy + lado * 0.55))
        rec = img[c:d, a:b].copy()
        cv2.rectangle(rec, (int(x1 - a), int(y1 - c)), (int(x2 - a), int(y2 - c)), (54, 104, 235), 2)
        rec = cv2.resize(rec, (round(rec.shape[1] * 260 / rec.shape[0]), 260), interpolation=cv2.INTER_CUBIC)
        recortes.append((cv2.cvtColor(rec, cv2.COLOR_BGR2RGB), f"{video.split('_')[0]}_{video.split('_')[1]}, "
                                                               f"bache {conf:.2f}".replace(".", ",")))
    fig, ejes = plt.subplots(1, 3, figsize=(6.3, 1.9))
    for ax, (img, titulo) in zip(ejes, recortes):
        ax.imshow(img)
        ax.set_title(titulo, fontsize=8)
        ax.axis("off")
    fig.tight_layout(w_pad=0.6)
    guardar(fig, "figura_videos.png")


if __name__ == "__main__":
    muestras()
    curvas()
    barras()
    umbral()
    arquitectura()
    videos()
