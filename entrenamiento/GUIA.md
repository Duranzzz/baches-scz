# Guía: entrenar el detector de baches en AMD Developer Cloud

> **Nota del 2026-11-21:** la ruta completa de más abajo todavía no se siguió en una GPU (ver la nota
> de `PLAN.md`). Para presentar al día siguiente, usar la **ruta rápida**.

## Ruta rápida: una corrida con los datos que ya están en Hugging Face

Lo mismo que A1 (YOLO12s de RDD2022, 1 clase "D40", 1280 px), pero sin bajar RDD2022 ni HRP4K: solo las
12.204 fotos del paquete. Queda **HRP4K fuera del entrenamiento**: su test mide si el modelo generaliza a una
cámara y un país que no vio. Probado en CPU el 21-nov: `armar_dataset.py` da 7.126 fotos de train
(5.315 con bache) y 1.222 de val, y `entrenar.py --prueba` corre completo.

| Paso | Tiempo | Costo (1,99 USD/h) |
|---|---|---|
| Crear la máquina, entrar, bajar el paquete (1,8 GB) | ~15 min | ~0,5 USD |
| Prueba de 2 épocas | 10 a 20 min | ~0,5 USD |
| Corrida + medición + subida | hasta ~4 h (se elige con la prueba) | ~8 USD |

**1. Crear la máquina y entrar:** pasos 1 y 2 de la ruta completa (imagen Quick Start "PyTorch", contenedor `rocm`).

**2. Bajar el paquete y preparar** (dentro del contenedor):

```bash
export HF_TOKEN=hf_...            # el token de Hugging Face con el que subiste el paquete
mkdir -p /datos/scz && cd /datos/scz
pip install -q huggingface_hub
python -c "from huggingface_hub import snapshot_download as s; s('Duranishhh/baches-scz', repo_type='dataset', local_dir='/datos/_hf', allow_patterns=['paquete/*'])"
for t in codigo etiquetas fotos; do tar -xf /datos/_hf/paquete/$t.tar; done
pip install -q "ultralytics==8.4.191" dill
python -c "import cv2" 2>/dev/null || { pip uninstall -y -q opencv-python; pip install -q opencv-python-headless; }
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))"   # hip no puede salir None
python entrenamiento/armar_dataset.py --salida /datos/ds_baches     # tiene que decir 7126 train / 1222 val
```

**3. Prueba de 2 épocas** (mide cuánto tarda una época en esta máquina):

```bash
python entrenamiento/entrenar.py --datos /datos/ds_baches/data.yaml --modelo yolo12s-rdd --nombre a1 --prueba --cache-ram
```

Al final imprime "~X s por época completa". Elegir **EPOCAS = 14400 / X** (para que entre en ~4 h), con un
máximo de 120. Si salen pérdidas `nan` o mAP 0, repetir con `--sin-amp`.

**4. Corrida, medición y subida, todo de una vez** (se puede cerrar la terminal; sigue corriendo):

```bash
cat > a1.sh <<'EOF'
set -e
cd /datos/scz
python entrenamiento/entrenar.py --datos /datos/ds_baches/data.yaml --modelo yolo12s-rdd --nombre a1 --cache-ram --epocas $EPOCAS
B=runs/a1/weights/best.pt
for s in 1280 640; do
  python evaluar.py --split val  --imgsz $s --barrer
  python evaluar.py --split test --imgsz $s
  python evaluar.py --split val  --imgsz $s --barrer --det-baches $B
  python evaluar.py --split test --imgsz $s --det-baches $B
done
python entrenamiento/hf.py subir --repo Duranishhh/baches-scz --runs a1
python -c "from huggingface_hub import HfApi; HfApi().upload_folder(repo_id='Duranishhh/baches-scz', repo_type='dataset', folder_path='salidas/cache', path_in_repo='resultados/cache')"
echo "TODO LISTO: ya puedes destruir la maquina"
EOF
EPOCAS=120 nohup bash a1.sh > a1.log 2>&1 &      # cambiar 120 por el número del paso 3
tail -f a1.log                                     # Ctrl+C sale de la vista
```

Mide la línea base y A1 en la misma máquina, con las mismas fotos, a 1280 y a 640 (el tamaño del teléfono).

**5. Cuando el log diga "TODO LISTO": destruir la máquina** (consola → el droplet → Destroy).

**6. Decidir, con una regla fijada antes de ver los resultados:** A1 reemplaza a YOLO12s para baches solo
si su AP50 de test supera al de YOLO12s en las cámaras de vehículo (Bucko, Egipto y HRP4K). Si no, se
presenta la línea base con este resultado. En la laptop:

```powershell
.venv\Scripts\python entrenamiento\hf.py bajar --repo Duranishhh/baches-scz --que resultados
.venv\Scripts\python exportar_web.py --app ..\baches-scz-app --modelo a1=salidas\_hf\resultados\a1\weights\best.pt   # si gana
```

## Ruta completa (plan original del 28-oct)

Paso a paso, para una MI300X con los 100 USD de crédito. El porqué de cada decisión está en `PLAN.md`.

**Tres reglas de plata:**
1. La máquina cobra **mientras exista**, aunque esté apagada. Al terminar, **destruirla**.
2. Los créditos vencen **30 días después del depósito**. No activarlos hasta tener el paquete subido (paso 0).
3. Si se acaban los créditos sin tarjeta, la máquina se borra con todo. Subir los resultados a Hugging
   Face después de cada corrida (`hf.py subir`).

## Tiempo y costo esperados

A 1,99 USD por hora (confirmar el precio en la consola). Las horas de entrenamiento son una
estimación: la prueba del paso 6 mide el número real y con eso se decide el resto.

| Paso | Dónde | Tiempo | Costo |
|---|---|---|---|
| 0. Subir el paquete (~2,5 GB) | tu PC | 30 a 90 min, según tu internet | 0 |
| 1-2. Crear la máquina y entrar | GPU | 10 min | 0,3 USD |
| 3. Preparar: bajar 22 GB, importar, splits, dataset | GPU | 45 a 90 min | 1,5 a 3 USD |
| 4. Guardar los datos preparados | GPU | 10 a 20 min | 0,5 USD |
| 5. Línea base sobre el test final | GPU | 15 a 30 min | 0,5 a 1 USD |
| 6. Prueba de 2 épocas | GPU | 10 a 20 min | 0,5 USD |
| 7. **A1**: YOLO12s-RDD a 1280 px | GPU | 6 a 14 h | 12 a 28 USD |
| 8. Medir A1 y subir resultados | GPU | 20 min | 0,7 USD |
| **Primera tanda (sin A2)** | | **~8 a 17 h** | **~16 a 34 USD** |
| 9. A2 (control YOLO26s), opcional | GPU | 6 a 14 h | 12 a 28 USD |
| **Reserva para reentrenar con video de Santa Cruz** | | | **≥ 40 USD** |

Prioridad del gasto: A1, después la ronda con datos de Santa Cruz, y A2 solo si sobra.

## 0. En tu PC, antes de prender nada (gratis)

1. Crear un token de Hugging Face con permiso de escritura: https://huggingface.co/settings/tokens
2. En la carpeta del proyecto:

```powershell
.venv\Scripts\hf auth login                       # pegar el token
.venv\Scripts\python entrenamiento\hf.py paquete --repo TU_USUARIO/baches-scz
```

Arma tres archivos (fotos reducidas a 1280 px, etiquetas, código y modelos) y los sube a un
repositorio **privado**. Si se corta, volver a correr el mismo comando: lo ya subido no se repite.

3. Crear una clave SSH, si no tienes: `ssh-keygen -t ed25519` (Enter a todo). La pública queda en
   `C:\Users\Duran\.ssh\id_ed25519.pub`.

## 1. Crear la máquina (desde aquí, cobra)

En la consola de AMD Developer Cloud: **GPU Droplet → MI300X × 1 → imagen Quick Start "PyTorch"
(ROCm)**, y pegar la clave SSH pública. Anotar la IP.

## 2. Entrar

```bash
ssh root@LA_IP
docker ps                      # el contenedor de PyTorch suele llamarse "rocm"
docker exec -it rocm bash      # todo lo que sigue va dentro del contenedor
df -h /                        # tiene que haber más de 80 GB libres
```

## 3. Bajar el paquete y preparar los datos (~1 h)

```bash
export HF_TOKEN=hf_tu_token
pip install -q huggingface_hub
mkdir -p /datos/scz && cd /datos/scz
python -c "from huggingface_hub import snapshot_download as s; s('TU_USUARIO/baches-scz', repo_type='dataset', local_dir='/datos/_hf', allow_patterns=['paquete/*'])"
for t in codigo etiquetas fotos; do tar -xf /datos/_hf/paquete/$t.tar; done
DATOS=/datos nohup bash entrenamiento/preparar_maquina.sh > preparar.log 2>&1 &
tail -f preparar.log           # Ctrl+C sale de la vista; el proceso sigue
```

El script verifica la GPU, instala ultralytics sin tocar el PyTorch de ROCm, baja RDD2022 y HRP4K
(verificando el MD5), los importa, rehace los splits y arma el dataset. Si algo falla, se detiene
con un mensaje. **Revisar al final:**

- RDD2022: fotos y baches por país, y los códigos descartados (D43, D44, D50, Repair).
- HRP4K: "0 con etiquetas distintas" en el test.
- Dataset: cerca de 20.000 fotos de train y 3.500 de val.

## 4. Guardar los datos preparados (para no repetir el paso 3 en otra sesión)

```bash
python entrenamiento/hf.py datos --repo TU_USUARIO/baches-scz
```

## 5. Línea base sobre el test final

Con RDD2022 y HRP4K importados, el test cambió: hay que volver a medir los modelos públicos para
comparar con justicia.

```bash
python evaluar.py --split val --barrer --imgsz 1280
python evaluar.py --split test --imgsz 640
python evaluar.py --split test --imgsz 1280
```

## 6. Prueba de 2 épocas (obligatoria)

```bash
python entrenamiento/entrenar.py --datos /datos/ds_baches/data.yaml --modelo yolo12s-rdd --nombre a1 --prueba
```

Al final imprime la **proyección** de horas y dólares. Decidir con ella:

| Proyección de A1 | Qué hacer |
|---|---|
| hasta 10 h | A1 completo (120 épocas) |
| 10 a 18 h | A1 con `--epocas 80`; no hacer A2 |
| más de 18 h | `--imgsz 1024 --epocas 80` (~35 % menos de cómputo) |

Si aparecen pérdidas `nan` o mAP 0: repetir con `--sin-amp`.

## 7. Corrida A1

```bash
nohup python entrenamiento/entrenar.py --datos /datos/ds_baches/data.yaml --modelo yolo12s-rdd --nombre a1 > a1.log 2>&1 &
tail -f a1.log                 # progreso
tail -3 runs/a1/results.csv    # métricas por época
rocm-smi                       # uso de la GPU (desde otra terminal)
```

Si se corta (red, reinicio): `python entrenamiento/entrenar.py --nombre a1 --reanudar`.
Ultralytics se detiene solo si val no mejora en 30 épocas.

## 8. Medir A1 y subir resultados

```bash
python evaluar.py --split val --barrer --imgsz 1280 --det-baches runs/a1/weights/best.pt
# Usar los umbrales que imprime el barrido:
python evaluar.py --split test --imgsz 1280 --det-baches runs/a1/weights/best.pt --conf-bache X --conf-seg-solo Y
python entrenamiento/hf.py subir --repo TU_USUARIO/baches-scz --runs a1
```

El test se mide **una sola vez**, con los umbrales elegidos en val. Comparar por dataset contra el
paso 5: lo que importa son Bucko, HRP4K, Egipto y RDD2022 (cámara frontal).

## 9. A2, solo si sobra presupuesto

Igual que 6 a 8, con `--modelo yolo26s --nombre a2`. Gana el que tenga mejor resultado en val de
cámara frontal.

## 10. Destruir la máquina

Consola → el droplet → **Destroy**. Apagarla no alcanza.

En tu PC, para bajar los resultados y probarlos:

```powershell
.venv\Scripts\python entrenamiento\hf.py bajar --repo TU_USUARIO/baches-scz --que resultados
.venv\Scripts\python preview.py --det-baches salidas\_hf\resultados\a1\weights\best.pt --imgsz 1280
```

## Problemas comunes

| Síntoma | Qué hacer |
|---|---|
| `PyTorch no ve la GPU` o `no es la versión ROCm` | Usar la imagen Quick Start "PyTorch", dentro del contenedor |
| `pip cambió torch` | Recrear el contenedor o reinstalar el torch de ROCm de la imagen |
| `cv2` no carga | El script cambia solo a `opencv-python-headless` |
| `nan` en las pérdidas o mAP 0 | `--sin-amp` |
| Falta memoria de GPU | `--batch 16` |
| La GPU está ociosa (`rocm-smi` < 70 %) | `--cache-ram` (usa ~110 GB de RAM) |
| Se cortó la conexión SSH | Los procesos con `nohup` siguen; volver a entrar y `tail -f` |

## Una sesión nueva (por ejemplo, con video de Santa Cruz)

En vez del paso 3, bajar lo ya preparado y armar el dataset:

```bash
python -c "from huggingface_hub import snapshot_download as s; s('TU_USUARIO/baches-scz', repo_type='dataset', local_dir='/datos/_hf', allow_patterns=['paquete/codigo.tar','datos/*'])"
cd /datos/scz && tar -xf /datos/_hf/paquete/codigo.tar && tar -xf /datos/_hf/datos/preparados.tar
pip install -q "ultralytics==8.4.191" dill
python entrenamiento/armar_dataset.py --salida /datos/ds_baches
```
