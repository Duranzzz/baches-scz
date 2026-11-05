"""Reexporta los fotogramas ya analizados sin repetir inferencia."""
import csv,json,time
from pathlib import Path
from collections import Counter
import cv2
from comparar_video_local import MODELS,encode,probe,digest,sheets

def main():
    out=Path(__file__).resolve().parent/'salidas/videos/whatsapp_20261104_comparacion_20261105'
    info=json.loads((out/'preparacion.json').read_text(encoding='utf-8'))
    original=Path(info['video']);n=info['frames'];fps=info['fps']
    assert digest(original)==info['video_sha256']
    for key,spec in MODELS.items():
        target=out/(key+'.mp4');raw=out/(key+'_temporal.mp4')
        raw_meta=probe(raw)
        stream=next(s for s in raw_meta['streams'] if s['codec_type']=='video')
        assert int(stream['nb_read_frames'])==n
        rows=[json.loads(line) for line in (out/(key+'_detecciones.jsonl')).read_text(encoding='utf-8').splitlines()]
        assert len(rows)==n and [r['frame'] for r in rows]==list(range(n))
        with (out/(key+'_detecciones.csv')).open(encoding='utf-8-sig') as f:csv_rows=list(csv.DictReader(f))
        assert len(csv_rows)==sum(len(r['detections']) for r in rows)
        if not target.exists():encode(raw,original,target)
        meta=probe(target)
        final_stream=next(s for s in meta['streams'] if s['codec_type']=='video')
        assert int(final_stream['nb_read_frames'])==n
        assert abs(float(meta['format']['duration'])-float(info['metadata']['format']['duration']))<.15
        assert any(s['codec_type']=='audio' for s in meta['streams'])
        totals=Counter();frame_counts=Counter();maximum={}
        for row in rows:
            frame_counts.update({d['class'] for d in row['detections']})
            for d in row['detections']:
                totals[d['class']]+=1
                maximum[d['class']]=max(maximum.get(d['class'],0),d['confidence'])
        cap=cv2.VideoCapture(str(target));images=[];labels=[]
        for s in [0,10,20,30,40,50,60,70,80,90,98]:
            idx=int(s*fps);cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,img=cap.read();assert ok
            images.append(img);labels.append(f'{idx/fps:.1f}s | {len(rows[idx]["detections"])} marcas')
        cap.release();sheets(images,labels,out/(key+'_muestra.jpg'))
        checkpoint=dict(spec,path=str(spec['path']),sha256=digest(spec['path']))
        progress=json.loads((out/(key+'_progreso.json')).read_text(encoding='utf-8'))
        result=dict(model=key,checkpoint=checkpoint,classes=info['models'][key]['names'],video=str(original),output=str(target),frames_processed=n,frames_with_detections=sum(bool(r['detections']) for r in rows),detection_observations_by_class=dict(totals),frames_by_class=dict(frame_counts),max_confidence=maximum,imgsz=1280,confidence=.25,iou=.7,device='cpu',threads=4,backend='PyTorch FP32, checkpoint original; exportación ONNX de diagnóstico no utilizada',inference_elapsed_seconds_at_frame_2950=progress['elapsed_seconds'],input_sha256=info['video_sha256'],output_sha256=digest(target),output_metadata=meta,note='Observaciones repetidas por fotograma, no daños únicos ni precisión. Severidad sin evaluar. Se corrigió solo la exportación audiovisual para conservar el último fotograma.')
        (out/(key+'_resumen.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(dict(model=key,status='verified',frames=n,counts=dict(totals),output=str(target))),flush=True)

if __name__=='__main__':main()
