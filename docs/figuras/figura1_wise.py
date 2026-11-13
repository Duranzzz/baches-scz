"""Figura 1 del artículo: AP50 de baches a 640 px según la proporción alfa del modelo ajustado en el promedio
de pesos (0 = YOLO26s original, 1 = ajustado). Datos de informes/entrenamiento_amd.md (eval_*.json).
    python docs/figuras/figura1_wise.py"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ALFA = [0, 0.25, 0.5, 0.75, 1]
PANELES = [
    ("Conjuntos usados en el ajuste", [
        ("Validación, total", [0.191, 0.299, 0.478, 0.680, 0.732]),
        ("Prueba, Egipto", [0.356, 0.471, 0.534, 0.484, 0.442]),
    ]),
    ("HRP4K, no visto al entrenar", [
        ("Mitad A (elegir)", [0.034, 0.029, 0.025, 0.015, 0.011]),
        ("Mitad B (informar)", [0.065, 0.059, 0.040, 0.019, 0.019]),
    ]),
]
COLORES = ["#2a78d6", "#eb6834"]          # paleta validada (contraste y daltonismo)
ESTILOS = [("o", "-"), ("s", "--")]       # codificación secundaria para impresión en gris
TINTA, GRIS = "#222222", "#8a8a8a"
coma = FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ","))

plt.rcParams.update({"font.family": "Arial", "font.size": 8.5, "axes.edgecolor": GRIS, "axes.linewidth": 0.6,
                     "xtick.color": TINTA, "ytick.color": TINTA, "text.color": TINTA, "axes.labelcolor": TINTA})
fig, ejes = plt.subplots(1, 2, figsize=(6.3, 2.7), dpi=300)
for ax, (titulo, series) in zip(ejes, PANELES):
    ax.axvspan(0.18, 0.32, color="#e8e8e6", zorder=0)
    for (nombre, ys), color, (marca, linea) in zip(series, COLORES, ESTILOS):
        ax.plot(ALFA, ys, color=color, lw=1.6, ls=linea, marker=marca, ms=5, zorder=3, label=nombre)
    ax.set_title(titulo, fontsize=9, fontweight="bold", loc="left")
    ax.set_xticks(ALFA)
    ax.set_xticklabels(["0\noriginal", "0,25", "0,50", "0,75", "1\najustado"])
    ax.set_xlabel("Proporción α del modelo ajustado")
    ax.yaxis.set_major_formatter(coma)
    ax.grid(axis="y", color="#e2e2e0", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-0.08, 1.08)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left" if "ajuste" in titulo else "upper right")
ejes[0].set_ylabel("AP50 de baches")
ejes[1].set_ylabel("AP50 de baches")
ejes[0].set_ylim(0, 0.85)
ejes[1].set_ylim(0, 0.08)
ejes[1].axhline(0.029, color=GRIS, lw=0.8, ls=":")
ejes[1].text(1.06, 0.0305, "límite de la regla", ha="right", va="bottom", fontsize=7, color="#555555")
for ax in ejes:
    ax.text(0.25, ax.get_ylim()[1] * 0.02, "elegido", ha="center", va="bottom", fontsize=7, color="#555555")
fig.tight_layout(w_pad=2.5)
salida = Path(__file__).with_name("figura1_wise.png")
fig.savefig(salida, facecolor="white")
print(salida)
