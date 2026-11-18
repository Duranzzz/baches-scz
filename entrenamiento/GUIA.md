# Guía: entrenar el detector de baches en AMD Developer Cloud

Una corrida (A1 de `PLAN.md`) en una MI300X, con las fotos y etiquetas que ya están en el repositorio
privado de Hugging Face `Duranishhh/baches-scz` (subidas el 28-oct con `hf.py paquete`).
Probado en CPU el 21-nov: `armar_dataset.py` da 7.126 fotos de train (5.315 con bache) y 1.222 de val,
y `entrenar.py --prueba` corre completo.

> **Corrida real del 22-nov-2026 (`informes/entrenamiento_amd.md`), con estos cambios a lo de abajo:**
> - **Modelo de partida:** el 22-nov fueron unos pesos públicos preentrenados (ver `MODEL_CARD.md`); desde ahora, por defecto, `modelos/baches_scz_v1.pt`. La clase del `data.yaml` se llama
>   "Pothole (D40)" para que ultralytics reuse su fila de bache. Desde el 23-nov, `armar_dataset.py` la escribe así:
>   - con "D40", el log dice "Transferred 696/708" y la fila queda al azar;
>   - con el nombre correcto dice 708/708.
> - **Caché:** `--cache-disco`. Con `--cache-ram`, Python 3.14 copia la caché en cada worker y se acaba la RAM
>   (235 GB).
> - **Tope de horas:** `--horas` es un freno de seguridad. El `time` de ultralytics estiraba las épocas para
>   llenar el tiempo.
> - **Aumentos:** `--aumentos-scz` imita video de celular por WhatsApp.
> - **Velocidad:** ~49 s por época con 6.507 fotos a 1280 px. a1 hizo 91 épocas en 1,2 h.
> - **Limpieza:** antes de armar el dataset, `limpieza/aplicar_correcciones.py`.
> - **HRP4K:** para sumar su train, `importar_hrp4k_hf.py` (copia de Hugging Face; Zenodo bajaba muy lento).

**Reglas de plata**

1. La máquina cobra **mientras exista**, aunque esté apagada. Al terminar, **destruirla**.
2. Los créditos vencen 30 días después del depósito.
3. Si se acaban los créditos sin tarjeta, la máquina se borra con todo. Por eso el paso 4 sube los
   resultados a Hugging Face antes de terminar.

| Paso | Tiempo | Costo (1,99 USD/h) |
|---|---|---|
| 1-2. Crear la máquina, entrar, bajar el paquete (1,8 GB) | ~15 min | ~0,5 USD |
| 3. Prueba de 2 épocas | 10 a 20 min | ~0,5 USD |
| 4. Corrida, medición y subida | hasta ~4 h (se elige con la prueba) | ~8 USD |

## 1. Crear la máquina y entrar

1. Si no hay clave SSH: `ssh-keygen -t ed25519` (Enter a todo). La pública queda en `~/.ssh/id_ed25519.pub`.
2. En la consola de AMD Developer Cloud: **GPU Droplet → MI300X × 1 → imagen Quick Start "PyTorch"
   (ROCm)**, y pegar la clave SSH pública. Anotar la IP. Desde aquí, cobra.
3. Entrar:

```bash
ssh root@LA_IP
docker ps                      # el contenedor de PyTorch suele llamarse "rocm"
docker exec -it rocm bash      # todo lo que sigue va dentro del contenedor
```

## 2. Bajar el paquete y preparar

```bash
export HF_TOKEN=hf_...            # el token de Hugging Face con el que se subió el paquete
mkdir -p /datos/scz && cd /datos/scz
pip install -q huggingface_hub
python -c "from huggingface_hub import snapshot_download as s; s('Duranishhh/baches-scz', repo_type='dataset', local_dir='/datos/_hf', allow_patterns=['paquete/*'])"
for t in codigo etiquetas fotos; do tar -xf /datos/_hf/paquete/$t.tar; done
pip install -q "ultralytics==8.4.191" dill
python -c "import cv2" 2>/dev/null || { pip uninstall -y -q opencv-python; pip install -q opencv-python-headless; }
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))"   # hip no puede salir None
python entrenamiento/armar_dataset.py --salida /datos/ds_baches     # tiene que decir 7126 train / 1222 val
```

Si `hip` sale `None`, pip cambió el torch de ROCm: recrear el contenedor.

## 3. Prueba de 2 épocas

Mide cuánto tarda una época en esta máquina:

```bash
python entrenamiento/entrenar.py --datos /datos/ds_baches/data.yaml --modelo baches-scz-v1 --nombre a1 --prueba --cache-disco
```

Al final imprime "~X s por época completa". Elegir **EPOCAS = 14400 / X** (para que entre en ~4 h),
con un máximo de 120. Si salen pérdidas `nan` o mAP 0, repetir con `--sin-amp`.

## 4. Corrida, medición y subida, todo de una vez

Se puede cerrar la terminal: sigue corriendo.

```bash
cat > a1.sh <<'EOF'
set -e
cd /datos/scz
python entrenamiento/entrenar.py --datos /datos/ds_baches/data.yaml --modelo baches-scz-v1 --nombre a1 --cache-disco --epocas $EPOCAS
B=runs/a1/weights/best.pt
for s in 1280 640; do
  python herramientas/evaluar.py --split val  --imgsz $s --barrer
  python herramientas/evaluar.py --split test --imgsz $s
  python herramientas/evaluar.py --split val  --imgsz $s --barrer --det-baches $B
  python herramientas/evaluar.py --split test --imgsz $s --det-baches $B
done
python entrenamiento/hf.py subir --repo Duranishhh/baches-scz --runs a1
python -c "from huggingface_hub import HfApi; HfApi().upload_folder(repo_id='Duranishhh/baches-scz', repo_type='dataset', folder_path='salidas/cache', path_in_repo='resultados/cache')"
echo "TODO LISTO: ya puedes destruir la maquina"
EOF
EPOCAS=120 nohup bash a1.sh > a1.log 2>&1 &      # cambiar 120 por el número del paso 3
tail -f a1.log                                     # Ctrl+C sale de la vista
```

Mide la línea base y A1 en la misma máquina, con las mismas fotos, a 1280 y a 640 px (el tamaño
del teléfono). Para ver el avance desde otra terminal: `tail -3 runs/a1/results.csv` y `rocm-smi`.
Si se corta (red, reinicio): `python entrenamiento/entrenar.py --nombre a1 --reanudar`.

## 5. Destruir la máquina

Cuando el log diga "TODO LISTO": consola → el droplet → **Destroy**. Apagarla no alcanza.

## 6. Decidir y llevar el modelo a la app

La regla se fija antes de ver los resultados: A1 reemplaza al detector actual para baches solo si su AP50
de test lo supera en las cámaras de vehículo (Bucko, Egipto y HRP4K). Si no, se presenta la línea base
con este resultado. En la laptop:

```powershell
.venv\Scripts\python entrenamiento\hf.py bajar --repo Duranishhh/baches-scz --que resultados
.venv\Scripts\python herramientas\exportar_web.py --app ..\baches-scz-app --modelo a1=salidas\_hf\resultados\a1\weights\best.pt   # si gana
```

Y copiar `best.pt` a `modelos/a1/weights/best.pt` para que quede en el repo.

## Problemas comunes

| Síntoma | Qué hacer |
|---|---|
| `hip` sale `None` o PyTorch no ve la GPU | Usar la imagen Quick Start "PyTorch", dentro del contenedor `rocm` |
| `cv2` no carga | El paso 2 cambia solo a `opencv-python-headless` |
| `nan` en las pérdidas o mAP 0 | `--sin-amp` |
| Falta memoria de GPU | `--batch 16` |
| Falta RAM con `--cache-ram` | Quitar `--cache-ram` (ocupa ~27 GB con estas fotos) |
| Se cortó la conexión SSH | Los procesos con `nohup` siguen; volver a entrar y `tail -f a1.log` |
| Hay que cambiar fotos o etiquetas | En la PC: `python entrenamiento/hf.py paquete --repo Duranishhh/baches-scz` y volver al paso 2 |
