#!/usr/bin/env bash
# Prepara la máquina de GPU: verifica ROCm, instala dependencias, baja RDD2022 y HRP4K, los importa,
# rehace los splits y arma el dataset de entrenamiento. Corre dentro del contenedor de PyTorch (ROCm),
# desde la raíz del proyecto (donde está preview.py):
#
#     DATOS=/datos nohup bash entrenamiento/preparar_maquina.sh > preparar.log 2>&1 &
#     tail -f preparar.log
#
# Se puede volver a correr: las descargas se retoman y lo ya hecho se rehace igual.
set -euo pipefail
cd "$(dirname "$0")/.."
DATOS=${DATOS:-/datos}
PY=$(command -v python || command -v python3)
paso() { echo; echo "=== [$(date +%H:%M:%S)] $*"; }

paso "1/7 GPU y PyTorch (ROCm)"
"$PY" - <<'EOF'
import torch
assert torch.cuda.is_available(), "PyTorch no ve la GPU"
assert torch.version.hip, "este torch no es la versión ROCm"
p = torch.cuda.get_device_properties(0)
print(f"torch {torch.__version__} | HIP {torch.version.hip} | {p.name} | {p.total_memory / 1e9:.0f} GB")
EOF
mkdir -p "$DATOS"
LIBRE=$(df -BG --output=avail "$DATOS" | tail -1 | tr -dc 0-9)
echo "disco libre en $DATOS: ${LIBRE} GB"
[ "$LIBRE" -ge 80 ] || { echo "hacen falta ~80 GB libres"; exit 1; }

paso "2/7 Dependencias (sin tocar el torch de ROCm)"
TORCH_ANTES=$("$PY" -c "import torch; print(torch.__version__)")
"$PY" -m pip install -q "ultralytics==8.4.191" huggingface_hub dill pytest requests
TORCH_DESPUES=$("$PY" -c "import torch; print(torch.__version__)")
if [ "$TORCH_ANTES" != "$TORCH_DESPUES" ]; then
    echo "pip cambió torch ($TORCH_ANTES -> $TORCH_DESPUES): reinstalar el de ROCm antes de seguir"; exit 1
fi
if ! "$PY" -c "import cv2" 2>/dev/null; then
    echo "cv2 no carga (falta libGL en el contenedor): se cambia a opencv-python-headless"
    "$PY" -m pip uninstall -y -q opencv-python
    "$PY" -m pip install -q opencv-python-headless
fi
"$PY" -c "import cv2, ultralytics; print('opencv', cv2.__version__, '| ultralytics', ultralytics.__version__)"
"$PY" -m pytest -q tests

paso "3/7 Descargar RDD2022 (13,3 GB) y HRP4K (8,7 GB), con verificación MD5"
"$PY" entrenamiento/descargar_datos.py --destino "$DATOS"

paso "4/7 Importar RDD2022 oficial (solo train, sin China_Drone, solo D00/D10/D20/D40)"
"$PY" entrenamiento/importar_rdd2022.py --zip "$DATOS/RDD2022_released_through_CRDDC2022.zip"

paso "5/7 Importar HRP4K (train y valid oficiales)"
"$PY" entrenamiento/importar_hrp4k.py --zip "$DATOS/HRP4K.zip"

paso "6/7 Splits (por grupo, sin casi copias entre train y val/test)"
"$PY" preparar_splits.py

paso "7/7 Dataset de entrenamiento del detector de baches"
"$PY" entrenamiento/armar_dataset.py --salida "$DATOS/ds_baches"

paso "Listo. Siguiente: entrenamiento/GUIA.md, paso 5"
