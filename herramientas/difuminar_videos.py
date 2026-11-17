"""Difumina caras y placas en los videos de Santa Cruz antes de publicarlos, y les quita el audio.
    python herramientas/difuminar_videos.py                        # videos/scz_*.mp4 -> salidas/difuminar/
    python herramientas/difuminar_videos.py videos/scz_02_frontal_avenida.mp4 --max-cuadros 150
    python herramientas/difuminar_videos.py --retocar herramientas/difuminar_retoques.json   # zonas puestas a mano
Detectores, en salidas/difuminar/modelos/ (se bajan solos):
- placas: morsetechlab/yolov11-license-plate-detection, v1n en ONNX (AGPL-3.0). Corre en el cuadro entero a
  1280 px y, para cada vehículo de 60 px o más de alto, en un recorte de ese vehículo;
- caras: YuNet de OpenCV (ONNX), con el cuadro ampliado al doble;
- personas y vehículos: YOLO26n de Ultralytics entrenado con COCO. De cada persona de 60 px o más se difumina
  la cabeza. Es un pickle: se revisa con picklescan antes de cargarlo.
Cada zona se difumina también en los 3 cuadros anteriores y siguientes, por si un detector la pierde en alguno.
Por video deja el mp4 difuminado (H.264, sin audio) y un JSON con las zonas de cada cuadro. Los retoques agregan,
sobre el video ya difuminado, zonas fijas que el control a ojo encontró sin difuminar (por ejemplo, un espejo)."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "salidas" / "difuminar"
MODELOS = SALIDA / "modelos"
VECINOS = 3          # cuadros antes y después en que también se difumina cada zona
MIN_ALTO = 60        # px: en personas y vehículos más chicos, caras y placas no se leen a 848 x 478
PERSONA, VEHICULOS = 0, [1, 2, 3, 5, 7]   # COCO: bicicleta, auto, moto, bus y camión
# Margen alrededor de cada zona detectada (fracción del ancho, del alto)
MARGEN = {"placa": (0.25, 0.45), "cara": (0.30, 0.35), "cabeza": (0.10, 0.10), "zona": (0.03, 0.03)}


def bajar_modelos():
    from huggingface_hub import hf_hub_download
    from ultralytics.utils.downloads import attempt_download_asset
    MODELOS.mkdir(parents=True, exist_ok=True)
    placas = Path(hf_hub_download("morsetechlab/yolov11-license-plate-detection", "license-plate-finetune-v1n.onnx",
                                  local_dir=MODELOS))
    caras = Path(hf_hub_download("opencv/face_detection_yunet", "face_detection_yunet_2023mar.onnx", local_dir=MODELOS))
    coco = MODELOS / "yolo26n.pt"
    if not coco.exists():
        attempt_download_asset(str(coco))
    r = subprocess.run([str(Path(sys.executable).with_name("picklescan")), "--path", str(coco)],
                       capture_output=True, text=True)
    if "Dangerous globals: 0" not in r.stdout or "Infected files: 0" not in r.stdout:
        raise SystemExit(f"picklescan no aprobó {coco.name}:\n{r.stdout}{r.stderr}")
    return placas, caras, coco


def ampliar(x1, y1, x2, y2, fx, fy, ancho, alto):
    w, h = x2 - x1, y2 - y1
    return (max(0, x1 - w * fx), max(0, y1 - h * fy), min(ancho, x2 + w * fx), min(alto, y2 + h * fy))


class Detectores:
    def __init__(self):
        from ultralytics import YOLO
        placas, caras, coco = bajar_modelos()
        self.placas = YOLO(str(placas), task="detect")
        self.coco = YOLO(str(coco))
        self.caras = cv2.FaceDetectorYN.create(str(caras), "", (320, 320), 0.6, 0.3, 5000)

    def zonas(self, img):
        """Zonas a difuminar en un cuadro: (tipo, x1, y1, x2, y2) en píxeles del cuadro."""
        alto, ancho = img.shape[:2]
        zonas, vehiculos = [], []
        r = self.coco.predict(img, imgsz=960, conf=0.25, classes=[PERSONA, *VEHICULOS], verbose=False)[0]
        for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()):
            if y2 - y1 < MIN_ALTO:
                continue
            if int(c) == PERSONA:   # la cabeza: el 28 % de arriba de la persona
                zonas.append(("cabeza", x1, y1, x2, y1 + 0.28 * (y2 - y1)))
            else:
                vehiculos.append((x1, y1, x2, y2))
        for x1, y1, x2, y2 in self.placas.predict(img, imgsz=1280, conf=0.15, verbose=False)[0].boxes.xyxy.tolist():
            zonas.append(("placa", x1, y1, x2, y2))
        for v in vehiculos:   # segunda pasada: la placa de cada vehículo cercano, en un recorte
            X1, Y1, X2, Y2 = (int(round(c)) for c in ampliar(*v, 0.10, 0.10, ancho, alto))
            recorte = img[Y1:Y2, X1:X2]
            if min(recorte.shape[:2]) < 8:
                continue
            for x1, y1, x2, y2 in self.placas.predict(recorte, imgsz=640, conf=0.15, verbose=False)[0].boxes.xyxy.tolist():
                zonas.append(("placa", x1 + X1, y1 + Y1, x2 + X1, y2 + Y1))
        doble = cv2.resize(img, (ancho * 2, alto * 2), interpolation=cv2.INTER_LINEAR)
        self.caras.setInputSize((ancho * 2, alto * 2))
        _, caras = self.caras.detect(doble)
        for x, y, w, h in ([] if caras is None else caras[:, :4].tolist()):
            zonas.append(("cara", x / 2, y / 2, (x + w) / 2, (y + h) / 2))
        return [(t, *(round(c, 1) for c in caja)) for t, *caja in zonas]


def difuminar(img, zonas):
    """Pixela y suaviza cada zona, con bordes difusos: rectángulo para placas, elipse para caras y cabezas."""
    alto, ancho = img.shape[:2]
    for tipo, *caja in zonas:
        x1, y1, x2, y2 = (int(round(c)) for c in ampliar(*caja, *MARGEN[tipo], ancho, alto))
        w, h = x2 - x1, y2 - y1
        if w < 3 or h < 3:
            continue
        roi = img[y1:y2, x1:x2].astype(np.float32)
        k = max(3, min(w, h) // 5)
        chico = cv2.resize(roi, (max(1, w // k), max(1, h // k)), interpolation=cv2.INTER_AREA)
        borroso = cv2.GaussianBlur(cv2.resize(chico, (w, h), interpolation=cv2.INTER_LINEAR), (0, 0), max(1.0, k / 2))
        mascara = np.zeros((h, w), np.float32)
        b = max(1, min(w, h) // 8)
        if tipo in ("placa", "zona"):
            cv2.rectangle(mascara, (b, b), (w - 1 - b, h - 1 - b), 1.0, -1)
        else:
            cv2.ellipse(mascara, (w // 2, h // 2), (max(1, w // 2 - b), max(1, h // 2 - b)), 0, 0, 360, 1.0, -1)
        mascara = cv2.GaussianBlur(mascara, (0, 0), b / 2 + 0.5)[..., None]
        img[y1:y2, x1:x2] = np.clip(borroso * mascara + roi * (1 - mascara), 0, 255).astype(np.uint8)
    return img


def procesar(video, det, max_cuadros=None):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    ancho, alto = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_cuadros:
        total = min(total, max_cuadros)
    regiones, t0 = [], time.time()
    while len(regiones) < total:   # 1.ª pasada: detectar
        ok, img = cap.read()
        if not ok:
            break
        regiones.append(det.zonas(img))
        if len(regiones) % 250 == 0:
            s = (time.time() - t0) / len(regiones)
            print(f"  {video.stem}: {len(regiones)}/{total} cuadros, {s:.2f} s/cuadro, faltan {s * (total - len(regiones)) / 60:.0f} min",
                  flush=True)
    cap.release()
    n = len(regiones)
    salida = SALIDA / f"{video.stem}.mp4"
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ancho}x{alto}",
         "-r", f"{fps:.3f}", "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(salida)], stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(str(video))
    for t in range(n):   # 2.ª pasada: difuminar cada cuadro con sus zonas y las de sus vecinos
        ok, img = cap.read()
        if not ok:
            break
        zonas = [z for u in range(max(0, t - VECINOS), min(n, t + VECINOS + 1)) for z in regiones[u]]
        ffmpeg.stdin.write(difuminar(img, zonas).tobytes())
    cap.release()
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise SystemExit(f"ffmpeg falló con {video.name}")
    conteo = {t: sum(z[0] == t for r in regiones for z in r) for t in MARGEN}
    resumen = {"video": video.name, "cuadros": n, "fps": round(fps, 3), "tamano": f"{ancho}x{alto}",
               "detecciones": conteo, "cuadros_con_zonas": sum(bool(r) for r in regiones),
               "vecinos": VECINOS, "s_por_cuadro": round((time.time() - t0) / max(1, n), 3), "zonas": regiones}
    (SALIDA / f"{video.stem}.json").write_text(json.dumps(resumen, ensure_ascii=False), encoding="utf-8")
    print(f"{video.name}: {n} cuadros, {conteo}, -> {salida}", flush=True)


def retocar(retoques):
    """Agrega zonas fijas por rango de cuadros a videos ya difuminados y los vuelve a codificar."""
    for video, zonas in json.loads(Path(retoques).read_text(encoding="utf-8")).items():
        entrada, temporal = SALIDA / f"{video}.mp4", SALIDA / f"{video}_retoque.mp4"
        cap = cv2.VideoCapture(str(entrada))
        fps = cap.get(cv2.CAP_PROP_FPS)
        ancho, alto = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ffmpeg = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ancho}x{alto}",
             "-r", f"{fps:.3f}", "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporal)], stdin=subprocess.PIPE)
        t = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            activas = [(z["tipo"], *z["caja"]) for z in zonas if z["desde"] <= t <= z["hasta"]]
            ffmpeg.stdin.write(difuminar(img, activas).tobytes())
            t += 1
        cap.release()
        ffmpeg.stdin.close()
        if ffmpeg.wait() != 0:
            raise SystemExit(f"ffmpeg falló con {video}")
        temporal.replace(entrada)
        print(f"{video}: {t} cuadros, {len(zonas)} retoques", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("videos", nargs="*", type=Path, help="por defecto, videos/scz_*.mp4")
    ap.add_argument("--max-cuadros", type=int, help="solo los primeros N cuadros (para probar)")
    ap.add_argument("--retocar", type=Path, help="JSON con zonas fijas para agregar a videos ya difuminados")
    args = ap.parse_args()
    if args.retocar:
        retocar(args.retocar)
        return
    videos = args.videos or sorted((RAIZ / "videos").glob("scz_*.mp4"))
    SALIDA.mkdir(parents=True, exist_ok=True)
    det = Detectores()
    for v in videos:
        procesar(v, det, args.max_cuadros)


if __name__ == "__main__":
    main()
