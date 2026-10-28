"""Visor de video: prueba los modelos sobre un video y muestra las cajas (detección) y las manchas
(segmentación de baches).

    python visor.py                                          # videos en la carpeta videos/
    python visor.py --det-baches runs/a1/weights/best.pt     # suma el detector de baches reentrenado

Modos:
  Cajas        YOLO12s (RDD2022): grietas D00, D10, D20, baches D40 y parches (Repair), un color por clase.
  Manchas      segmentación de baches (YOLOv8n-seg): el contorno de cada bache.
  Juntas       las dos cosas sobre el mismo cuadro.
  Lado a lado  cajas a la izquierda, manchas a la derecha.
  Severidad    lo mismo que preview.py: combina los dos modelos y colorea por severidad
               (usa los umbrales calibrados de preview.py, no el deslizador de confianza).
Teclas: espacio = reproducir/pausa, flecha derecha = siguiente cuadro, S = captura.
"Exportar" guarda en salidas/videos/ el video anotado (1 de cada N cuadros) y un CSV con cada detección.
"""
import argparse
import csv
import ctypes
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import tkinter as tk
from collections import Counter
from pathlib import Path
from tkinter import ttk

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk
from ultralytics import YOLO

import preview as pv

VIDEOS = pv.RAIZ / "videos"
SALIDAS = pv.RAIZ / "salidas"
EXT_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".3gp"}
MODOS = ["Cajas", "Manchas", "Juntas", "Lado a lado", "Severidad"]
NOMBRE = {"D00": "longitudinal", "D10": "transversal", "D20": "cocodrilo", "D40": "bache", "Repair": "parche"}
COLOR = {  # BGR
    "D00": (255, 140, 0), "D10": (230, 220, 0), "D20": (200, 0, 200), "D40": (0, 0, 255),
    "Repair": (170, 170, 170), "nuevo": (0, 165, 255), "mancha": (0, 220, 0),
}
LEYENDA = [("D00 longitudinal", "D00"), ("D10 transversal", "D10"), ("D20 cocodrilo", "D20"),
           ("D40 bache", "D40"), ("Repair (parche)", "Repair"), ("Mancha de bache", "mancha"),
           ("Bache, modelo nuevo", "nuevo")]


def ruta_cv2(p):
    """cv2 en Windows no abre rutas con tildes o ñ: se usa el nombre corto (8.3), que es ASCII."""
    s = str(p)
    if os.name != "nt" or s.isascii():
        return s
    buf = ctypes.create_unicode_buffer(1024)
    n = ctypes.windll.kernel32.GetShortPathNameW(s, buf, 1024)
    return buf.value if n and buf.value.isascii() else s


def hex_de(bgr):
    return "#%02x%02x%02x" % (bgr[2], bgr[1], bgr[0])


def grosor_de(img):
    return max(2, round(min(img.shape[:2]) / 400))


def etiqueta(img, texto, x, y, color):
    g = grosor_de(img)
    escala = g * 0.35
    (tw, th), base = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, max(1, g // 2))
    ty = y - 4 if y - th - base - 4 > 0 else y + th + base + 4
    tx = max(0, min(x, img.shape[1] - tw - 4))
    cv2.rectangle(img, (tx, ty - th - base), (tx + tw + 4, ty + base // 2), color, -1)
    cv2.putText(img, texto, (tx + 2, ty - base // 2), cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0),
                max(1, g // 2), cv2.LINE_AA)


def dibujar_cajas(img, cajas):
    out = img.copy()
    for modelo, nombre, conf, (x1, y1, x2, y2) in cajas:
        color = COLOR["nuevo"] if modelo == "det_baches" else COLOR.get(nombre, (200, 200, 200))
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, grosor_de(out))
        texto = f"{nombre} {NOMBRE.get(nombre, '')} {conf:.2f}" + (" (nuevo)" if modelo == "det_baches" else "")
        etiqueta(out, texto, int(x1), int(y1), color)
    return out


def dibujar_manchas(img, manchas):
    capa = img.copy()
    for _, _, m in manchas:
        capa[m] = COLOR["mancha"]
    out = cv2.addWeighted(capa, 0.45, img, 0.55, 0)
    area = float(img.shape[0] * img.shape[1])
    for conf, (x1, y1, x2, y2), m in manchas:
        contornos, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contornos, -1, COLOR["mancha"], grosor_de(out))
        etiqueta(out, f"bache {conf:.2f} | {m.sum() / area:.1%} de la imagen", int(x1), int(y1), COLOR["mancha"])
    return out


def titulo(img, texto):
    out = img.copy()
    etiqueta(out, texto, 8, 8 + grosor_de(out) * 12, (255, 255, 255))
    return out


class Motor:
    """Carga los modelos una vez y procesa un cuadro según el modo."""

    def __init__(self, det_baches=None):
        pv.descargar_modelos()
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.det = YOLO(str(pv.DET_PT))
        self.seg = YOLO(str(pv.SEG_PT))
        self.det_baches = YOLO(str(det_baches)) if det_baches else None
        self.lock = threading.Lock()   # el modelo no se usa desde dos hilos a la vez

    def procesar(self, frame, modo, conf, imgsz):
        """Devuelve (imagen anotada, filas para el CSV)."""
        with self.lock:
            if modo == "Severidad":
                danos, mascaras = pv.analizar(frame, self.det, self.seg, self.device, imgsz, self.det_baches)
                filas = [{"modelo": "combinado", "clase": d["tipo"], "confianza": d["confianza"], "caja": d["caja"],
                          "area_rel": d["area_rel"], "severidad": d["severidad"]} for d in danos]
                return pv.dibujar(frame, danos, mascaras), filas
            cajas = self._cajas(frame, conf, imgsz) if modo != "Manchas" else []
            manchas = self._manchas(frame, conf, imgsz) if modo != "Cajas" else []
        area = float(frame.shape[0] * frame.shape[1])
        filas = [{"modelo": m, "clase": n, "confianza": round(c, 3), "caja": [round(v, 1) for v in b],
                  "area_rel": round(pv.area_caja(b, area), 5), "severidad": ""} for m, n, c, b in cajas]
        filas += [{"modelo": "seg", "clase": "D40", "confianza": round(c, 3), "caja": [round(v, 1) for v in b],
                   "area_rel": round(float(m.sum()) / area, 5), "severidad": ""} for c, b, m in manchas]
        if modo == "Cajas":
            return dibujar_cajas(frame, cajas), filas
        if modo == "Manchas":
            return dibujar_manchas(frame, manchas), filas
        if modo == "Juntas":
            return dibujar_cajas(dibujar_manchas(frame, manchas), cajas), filas
        izq = titulo(dibujar_cajas(frame, cajas), "Cajas: YOLO12s (RDD2022)")
        der = titulo(dibujar_manchas(frame, manchas), "Manchas: segmentacion de baches")
        return np.hstack([izq, der]), filas

    def _cajas(self, frame, conf, imgsz):
        r = self.det.predict(frame, conf=conf, imgsz=imgsz, device=self.device, verbose=False)[0]
        out = [("det", self.det.names[k], c, b) for b, c, k in
               zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.int().tolist())]
        if self.det_baches is not None:
            r = self.det_baches.predict(frame, conf=conf, imgsz=imgsz, device=self.device, verbose=False)[0]
            out += [("det_baches", "D40", c, b) for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]
        return out

    def _manchas(self, frame, conf, imgsz):
        r = self.seg.predict(frame, conf=conf, imgsz=imgsz, device=self.device, retina_masks=True, verbose=False)[0]
        if r.masks is None:
            return []
        return [(c, b, m.cpu().numpy() > 0.5) for m, b, c in
                zip(r.masks.data, r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]


def nombre_archivo(texto):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", texto).strip("_") or "video"


def limitar_ancho(img, ancho_max=1920):
    if img.shape[1] <= ancho_max:
        return img
    r = ancho_max / img.shape[1]
    return cv2.resize(img, (ancho_max, round(img.shape[0] * r) // 2 * 2), interpolation=cv2.INTER_AREA)


class Visor(tk.Tk):
    def __init__(self, det_baches):
        super().__init__()
        self.title("Visor de baches y grietas")
        # Que entre en pantallas chicas (p. ej. 1366 x 768, con la barra de tareas)
        ancho, alto = min(1400, self.winfo_screenwidth() - 40), min(820, self.winfo_screenheight() - 90)
        self.geometry(f"{ancho}x{alto}+10+10")
        self.minsize(900, 560)
        self.det_baches = det_baches
        self.motor = None
        self.video = None
        self.info = {"total": 0, "fps": 30.0}
        self.params = {"modo": "Cajas", "conf": 0.25, "imgsz": 640, "salto": 5}
        self.cmds = queue.Queue()
        self.msgs = queue.Queue()
        self.cuadros = queue.Queue(maxsize=1)
        self.reproduciendo = threading.Event()
        self.cerrando = threading.Event()
        self.exportando = False
        self.ultimo = None          # (imagen anotada, filas, idx)
        self.foto_tk = None
        self._armar()
        self.protocol("WM_DELETE_WINDOW", self.cerrar)
        self.bind("<space>", lambda e: self.alternar())
        self.bind("<Right>", lambda e: self.paso())
        self.bind("<s>", lambda e: self.captura())
        self.bind("<S>", lambda e: self.captura())
        threading.Thread(target=self.trabajador, daemon=True).start()
        self.after(30, self.sondear)

    # ---------- interfaz ----------
    def _armar(self):
        panel = ttk.Frame(self, padding=8)
        panel.pack(side="left", fill="y")
        ttk.Label(panel, text="Videos (carpeta videos/)", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.lista = tk.Listbox(panel, height=6, width=34, exportselection=False)
        self.lista.pack(fill="x")
        self.lista.bind("<<ListboxSelect>>", lambda e: self.elegir())
        fila = ttk.Frame(panel)
        fila.pack(fill="x", pady=(2, 8))
        ttk.Button(fila, takefocus=False, text="Abrir carpeta", command=self.abrir_carpeta).pack(side="left")
        ttk.Button(fila, takefocus=False, text="Actualizar", command=self.actualizar_lista).pack(side="left", padx=4)

        ttk.Label(panel, text="Qué mostrar", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.modo = tk.StringVar(value=self.params["modo"])
        for m in MODOS:
            ttk.Radiobutton(panel, text=m, value=m, variable=self.modo, command=self.cambio).pack(anchor="w")

        ttk.Label(panel, text="Confianza mínima", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(8, 0))
        self.conf = tk.Scale(panel, from_=0.05, to=0.90, resolution=0.05, orient="horizontal", length=230)
        self.conf.set(self.params["conf"])
        self.conf.pack(anchor="w")
        self.conf.bind("<ButtonRelease-1>", lambda e: self.cambio())

        ttk.Label(panel, text="Tamaño de análisis", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(4, 0))
        self.imgsz = tk.IntVar(value=self.params["imgsz"])
        fila = ttk.Frame(panel)
        fila.pack(anchor="w")
        for v, t in ((640, "640 px (rápido)"), (1280, "1280 px (baches lejanos)")):
            ttk.Radiobutton(fila, text=t, value=v, variable=self.imgsz, command=self.cambio).pack(anchor="w")

        ttk.Label(panel, text="Procesar 1 de cada N cuadros", font=("Segoe UI", 10, "bold")).pack(anchor="w",
                                                                                                 pady=(4, 0))
        self.salto = tk.Scale(panel, from_=1, to=30, orient="horizontal", length=230)
        self.salto.set(self.params["salto"])
        self.salto.pack(anchor="w")
        self.salto.bind("<ButtonRelease-1>", lambda e: self.cambio(refrescar=False))

        fila = ttk.Frame(panel)
        fila.pack(fill="x", pady=8)
        self.boton_play = ttk.Button(fila, takefocus=False, text="▶ Reproducir", command=self.alternar)
        self.boton_play.pack(side="left")
        ttk.Button(fila, takefocus=False, text="⏭ +1 cuadro", command=self.paso).pack(side="left", padx=4)
        ttk.Button(fila, takefocus=False, text="⏮", command=lambda: self.buscar(0), width=3).pack(side="left")
        fila = ttk.Frame(panel)
        fila.pack(fill="x")
        ttk.Button(fila, takefocus=False, text="Captura (S)", command=self.captura).pack(side="left")
        self.boton_export = ttk.Button(fila, takefocus=False, text="Exportar video", command=self.exportar)
        self.boton_export.pack(side="left", padx=4)
        self.barra = ttk.Progressbar(panel, length=230, maximum=1.0)
        self.barra.pack(anchor="w", pady=4)


        derecha = ttk.Frame(self, padding=(0, 8, 8, 8))
        derecha.pack(side="left", fill="both", expand=True)
        self.vista = tk.Label(derecha, bg="#202020", fg="white", text="Cargando modelos…", font=("Segoe UI", 14))
        self.vista.pack(fill="both", expand=True)
        self.linea = tk.Scale(derecha, from_=0, to=1, orient="horizontal", showvalue=0)
        self.linea.pack(fill="x")
        self.linea.bind("<ButtonRelease-1>", lambda e: self.buscar(int(self.linea.get())))
        self.estado = ttk.Label(derecha, text="", anchor="w")
        self.estado.pack(fill="x")
        self.detalle = ttk.Label(derecha, text="", anchor="w", foreground="#555")
        self.detalle.pack(fill="x")
        leyenda = ttk.Frame(derecha)
        leyenda.pack(fill="x", pady=(2, 0))
        for texto, clave in LEYENDA:
            if clave == "nuevo" and not self.det_baches:
                continue
            tk.Label(leyenda, text=f" {texto} ", bg=hex_de(COLOR[clave]), fg="black").pack(side="left", padx=(0, 4))
        ttk.Label(leyenda, text="Severidad: verde baja, amarillo media, rojo alta").pack(side="left", padx=6)
        self.actualizar_lista()

    def actualizar_lista(self):
        VIDEOS.mkdir(exist_ok=True)
        self.videos = sorted(p for p in VIDEOS.iterdir() if p.suffix.lower() in EXT_VIDEO)
        self.lista.delete(0, "end")
        for p in self.videos:
            self.lista.insert("end", p.name)
        if not self.videos:
            self.lista.insert("end", "(vacía: copia videos en la carpeta)")

    def abrir_carpeta(self):
        VIDEOS.mkdir(exist_ok=True)
        try:
            os.startfile(VIDEOS)
        except AttributeError:
            subprocess.Popen(["xdg-open", str(VIDEOS)])

    def elegir(self):
        sel = self.lista.curselection()
        if not sel or not self.videos:
            return
        self.reproduciendo.clear()
        self.boton_play.config(text="▶ Reproducir")
        self.video = self.videos[sel[0]]
        self.cmds.put(("abrir", self.video))

    def cambio(self, refrescar=True):
        self.focus_set()   # que espacio y flecha vuelvan a ser atajos y no muevan el deslizador
        self.params = {"modo": self.modo.get(), "conf": float(self.conf.get()), "imgsz": int(self.imgsz.get()),
                       "salto": int(self.salto.get())}
        if refrescar and not self.reproduciendo.is_set():
            self.cmds.put(("refrescar",))

    def alternar(self):
        if self.video is None or self.exportando:
            return
        if self.reproduciendo.is_set():
            self.reproduciendo.clear()
            self.boton_play.config(text="▶ Reproducir")
        else:
            self.reproduciendo.set()
            self.boton_play.config(text="⏸ Pausa")

    def paso(self):
        if self.video is not None and not self.exportando:
            self.reproduciendo.clear()
            self.boton_play.config(text="▶ Reproducir")
            self.cmds.put(("paso",))

    def buscar(self, idx):
        self.focus_set()
        if self.video is not None and not self.exportando:
            self.cmds.put(("buscar", max(0, idx)))

    def captura(self):
        if self.ultimo is None:
            return
        img, _, idx = self.ultimo
        carpeta = SALIDAS / "capturas"
        carpeta.mkdir(parents=True, exist_ok=True)
        destino = carpeta / f"{nombre_archivo(self.video.stem)}_{idx:06d}_{nombre_archivo(self.params['modo'])}.jpg"
        cv2.imencode(".jpg", img)[1].tofile(destino)
        self.detalle.config(text=f"captura guardada: {destino.relative_to(pv.RAIZ)}")

    def cerrar(self):
        self.cerrando.set()
        self.destroy()

    # ---------- hilo que lee el video y corre los modelos ----------
    def trabajador(self):
        try:
            self.motor = Motor(self.det_baches)
        except Exception as e:  # noqa: BLE001
            self.msgs.put(("error", f"no se pudieron cargar los modelos: {e!r}"))
            return
        self.msgs.put(("listo", None))
        cap, idx, ultimo_frame = None, 0, None
        while not self.cerrando.is_set():
            try:
                cmd = self.cmds.get(timeout=0.02)
            except queue.Empty:
                cmd = None
            frame = None
            if cmd and cmd[0] == "abrir":
                if cap is not None:
                    cap.release()
                cap = cv2.VideoCapture(ruta_cv2(cmd[1]))
                if not cap.isOpened():
                    self.msgs.put(("error", f"no se pudo abrir {cmd[1].name}"))
                    cap = None
                    continue
                info = {"total": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
                        "ancho": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "alto": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
                self.msgs.put(("info", info))
                ok, frame = cap.read()
                idx = 0
            elif cmd and cmd[0] == "buscar" and cap is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, cmd[1])
                ok, frame = cap.read()
                idx = cmd[1]
            elif cmd and cmd[0] == "paso" and cap is not None:
                frame, idx = self._siguiente(cap, idx)
            elif cmd and cmd[0] == "refrescar":
                frame = ultimo_frame
            elif cmd is None and self.reproduciendo.is_set() and cap is not None:
                frame, idx = self._siguiente(cap, idx)
                if frame is None:
                    self.reproduciendo.clear()
                    self.msgs.put(("fin", None))
            if frame is None:
                continue
            ultimo_frame = frame
            p = dict(self.params)
            t0 = time.time()
            try:
                img, filas = self.motor.procesar(frame, p["modo"], p["conf"], p["imgsz"])
            except Exception as e:  # noqa: BLE001
                self.msgs.put(("error", repr(e)))
                continue
            dato = (img, filas, idx, (time.time() - t0) * 1000, p)
            try:
                self.cuadros.get_nowait()   # solo importa el último cuadro
            except queue.Empty:
                pass
            self.cuadros.put(dato)

    def _siguiente(self, cap, idx):
        salto = self.params["salto"]
        for _ in range(salto - 1):
            if not cap.grab():
                return None, idx
        ok, frame = cap.read()
        return (frame if ok else None), idx + salto

    # ---------- exportar ----------
    def exportar(self):
        if self.video is None or self.exportando or self.motor is None:
            return
        self.reproduciendo.clear()
        self.boton_play.config(text="▶ Reproducir")
        self.exportando = True
        self.boton_export.config(state="disabled")
        threading.Thread(target=self._exportar, args=(self.video, dict(self.params)), daemon=True).start()

    def _exportar(self, video, p):
        try:
            cap = cv2.VideoCapture(ruta_cv2(video))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            carpeta = SALIDAS / "videos"
            carpeta.mkdir(parents=True, exist_ok=True)
            nombre = nombre_archivo(f"{video.stem}_{p['modo']}_{p['imgsz']}px_1de{p['salto']}")
            tmp, final, ruta_csv = carpeta / f"{nombre}_tmp.mp4", carpeta / f"{nombre}.mp4", carpeta / f"{nombre}.csv"
            escritor, idx, n = None, 0, 0
            t0 = time.time()
            with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["video", "cuadro", "tiempo_s", "modelo", "clase", "confianza", "x1", "y1", "x2", "y2",
                            "area_rel", "severidad"])
                while not self.cerrando.is_set():
                    if idx % p["salto"]:
                        if not cap.grab():
                            break
                        idx += 1
                        continue
                    ok, frame = cap.read()
                    if not ok:
                        break
                    img, filas = self.motor.procesar(frame, p["modo"], p["conf"], p["imgsz"])
                    img = limitar_ancho(img)
                    img = img[: img.shape[0] // 2 * 2, : img.shape[1] // 2 * 2]   # H.264 pide lados pares
                    if escritor is None:
                        escritor = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"),
                                                   max(1.0, fps / p["salto"]), (img.shape[1], img.shape[0]))
                    escritor.write(img)
                    for d in filas:
                        w.writerow([video.name, idx, round(idx / fps, 3), d["modelo"], d["clase"], d["confianza"],
                                    *d["caja"], d["area_rel"], d["severidad"]])
                    n += 1
                    idx += 1
                    self.msgs.put(("progreso", (idx / total, n, time.time() - t0)))
            cap.release()
            if escritor is None:
                raise IOError("el video no tiene cuadros legibles")
            escritor.release()
            # H.264 con ffmpeg, si está, para que el video se vea en cualquier reproductor
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg and subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264",
                                          "-pix_fmt", "yuv420p", "-crf", "23", str(final)]).returncode == 0:
                tmp.unlink()
            else:
                tmp.replace(final)
            self.msgs.put(("exportado", (final, ruta_csv, n, time.time() - t0)))
        except Exception as e:  # noqa: BLE001
            self.msgs.put(("error", f"exportar: {e!r}"))
            self.msgs.put(("exportado", None))

    # ---------- bucle de la interfaz ----------
    def sondear(self):
        while True:
            try:
                tipo, dato = self.msgs.get_nowait()
            except queue.Empty:
                break
            if tipo == "listo":
                self.vista.config(text="Elige un video de la lista" if self.videos else
                                  "Copia videos en la carpeta videos/ y pulsa Actualizar")
                if self.videos:
                    self.lista.selection_set(0)
                    self.elegir()
            elif tipo == "info":
                self.info = dato
                self.linea.config(to=max(1, dato["total"] - 1))
                self.detalle.config(text=f"{self.video.name}: {dato['ancho']}×{dato['alto']}, {dato['fps']:.0f} fps, "
                                         f"{dato['total'] / dato['fps']:.0f} s")
            elif tipo == "fin":
                self.boton_play.config(text="▶ Reproducir")
                self.detalle.config(text="fin del video")
            elif tipo == "error":
                self.detalle.config(text=f"error: {dato}")
            elif tipo == "progreso":
                frac, n, seg = dato
                self.barra["value"] = frac
                self.estado.config(text=f"exportando… {frac:.0%} ({n} cuadros procesados, {seg:.0f} s)")
            elif tipo == "exportado":
                self.exportando = False
                self.boton_export.config(state="normal")
                if dato:
                    final, ruta_csv, n, seg = dato
                    self.barra["value"] = 1.0
                    self.estado.config(text=f"exportado: {final.relative_to(pv.RAIZ)} y {ruta_csv.name} "
                                            f"({n} cuadros en {seg:.0f} s)")
        try:
            img, filas, idx, ms, p = self.cuadros.get_nowait()
        except queue.Empty:
            pass
        else:
            self.ultimo = (img, filas, idx)
            self.mostrar(img)
            self.linea.set(idx)
            fps = self.info["fps"]
            cuenta = Counter(("mancha" if d["modelo"] == "seg" else
                              "nuevo" if d["modelo"] == "det_baches" else d["clase"]) for d in filas)
            resumen = ", ".join(f"{NOMBRE.get(k, k)} {v}" for k, v in sorted(cuenta.items())) or "nada"
            nota = " (umbrales de preview.py)" if p["modo"] == "Severidad" else ""
            if not self.exportando:
                self.estado.config(text=f"cuadro {idx}/{self.info['total']}  ({idx / fps:.1f} s)  |  "
                                        f"{ms:.0f} ms por cuadro a {p['imgsz']} px  |  {p['modo']}{nota}: {resumen}")
        if not self.cerrando.is_set():
            self.after(30, self.sondear)

    def mostrar(self, img):
        ancho = max(200, self.vista.winfo_width() - 4)
        alto = max(150, self.vista.winfo_height() - 4)
        r = min(ancho / img.shape[1], alto / img.shape[0])
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((max(1, int(img.shape[1] * r)), max(1, int(img.shape[0] * r))),
                                          Image.BILINEAR)
        self.foto_tk = ImageTk.PhotoImage(pil)
        self.vista.config(image=self.foto_tk, text="")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--det-baches", type=Path, help="pesos de un detector solo de baches (p. ej. el reentrenado)")
    args = ap.parse_args()
    Visor(args.det_baches).mainloop()


if __name__ == "__main__":
    main()
