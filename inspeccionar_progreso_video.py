import json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from comparar_video_local import MODELS,panel

out=Path(__file__).resolve().parent/'salidas/videos/whatsapp_20261104_comparacion_20261105'
meta=json.loads((out/'preparacion.json').read_text(encoding='utf-8'))
records={}
for key in MODELS:
    rows=[]
    for line in (out/(key+'_detecciones.jsonl')).read_text(encoding='utf-8').splitlines():
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:pass
    records[key]=rows
    print(json.dumps(dict(model=key,frames=len(rows),classes=dict(Counter(d['class'] for r in rows for d in r['detections'])))),flush=True)
idx=max(0,((min(len(r) for r in records.values())-1)//150)*150)
cap=cv2.VideoCapture(meta['video']);cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,frame=cap.read();assert ok;cap.release()
panels={key:panel(frame,rows[idx]['detections'],MODELS[key]['title'],idx,meta['fps'],meta['conf'],meta['imgsz'],MODELS[key]['kind']=='segment') for key,rows in records.items()}
base=panel(frame,[],'Original',idx,meta['fps'],meta['conf'],meta['imgsz'])
joined=np.vstack([np.hstack([base,panels['yolo12s_rdd2022']]),np.hstack([panels['farzad_segmentacion'],panels['piloto_v1_previo']])])
target=out/f'avance_comparacion_{idx:05d}.jpg'
cv2.imencode('.jpg',joined,[cv2.IMWRITE_JPEG_QUALITY,94])[1].tofile(target)
print(json.dumps(dict(preview=str(target),frame=idx,seconds=idx/meta['fps'])),flush=True)
