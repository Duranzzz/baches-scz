"""Compara exclusivamente checkpoints locales, sin entrenar ni descargar modelos."""
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parent
ARTIFACTS=Path(r'C:\Users\Duran\.codex\visualizations\2026\10\30\01a18e32-66bd-7b23-88db-280c28d470dc')
MODELS={
    'yolo12s_rdd2022':dict(path=ROOT/'modelos/yolo12s_rdd2022/yolo12s_RDD2022_best.pt',title='YOLO12s RDD2022',kind='detect',scope='Grietas, baches y clase Repair del checkpoint publico.'),
    'farzad_segmentacion':dict(path=ROOT/'modelos/farzad_pothole_seg/best.pt',title='Farzad YOLOv8n-seg',kind='segment',scope='Segmentacion publica de baches.'),
    'piloto_v1_previo':dict(path=ARTIFACTS/'entrenamiento_v1/resultado_piloto_01/baches_ajustado.pt',title='Piloto v1 previo - referencia',kind='detect',scope='Checkpoint del piloto anterior de dos epocas, no un entrenamiento nuevo ni un modelo final.'),
}
COLORS={'bache':(20,65,245),'grieta_longitudinal':(0,170,255),'grieta_transversal':(240,210,0),'grieta_cocodrilo':(210,50,200),'parche':(170,170,170)}
SHORT={'bache':'bache','grieta_longitudinal':'grieta longitudinal','grieta_transversal':'grieta transversal','grieta_cocodrilo':'grieta cocodrilo','parche':'Repair / parche'}

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def probe(path):
    return json.loads(subprocess.check_output([shutil.which('ffprobe'),'-v','error','-count_frames','-show_streams','-show_format','-of','json',str(path)],text=True,encoding='utf-8'))

def canonical(name,model_key):
    value=str(name).lower().replace('_',' ')
    if model_key in ('farzad_segmentacion','piloto_v1_previo'):return 'bache'
    if value.startswith('d00'):return 'grieta_longitudinal'
    if value.startswith('d10'):return 'grieta_transversal'
    if value.startswith('d20'):return 'grieta_cocodrilo'
    if value.startswith('d40'):return 'bache'
    if 'repair' in value:return 'parche'
    return str(name)

def panel(frame,items,title,idx,fps,conf,imgsz,polygons=False):
    out=frame.copy()
    if polygons:
        layer=out.copy()
        for d in items:
            points=np.asarray(d.get('polygon',[]),dtype=np.int32)
            if len(points)>=3:cv2.fillPoly(layer,[points],COLORS.get(d['class'],(180,180,180)))
        out=cv2.addWeighted(layer,0.32,out,0.68,0)
        for d in items:
            points=np.asarray(d.get('polygon',[]),dtype=np.int32)
            if len(points)>=3:cv2.polylines(out,[points],True,COLORS.get(d['class'],(180,180,180)),2,cv2.LINE_AA)
    for d in sorted(items,key=lambda d:d['confidence']):
        x1,y1,x2,y2=map(lambda x:int(round(x)),d['box'])
        color=COLORS.get(d['class'],(180,180,180))
        cv2.rectangle(out,(x1,y1),(x2,y2),color,2,cv2.LINE_AA)
        label=f"{SHORT.get(d['class'],d['class'])} {d['confidence']:.2f}"
        (tw,th),baseline=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.47,1)
        x=max(0,min(x1,out.shape[1]-tw-8))
        y=max(th+6,min(y1-4,out.shape[0]-baseline-2))
        cv2.rectangle(out,(x,y-th-5),(x+tw+7,y+baseline),color,-1)
        cv2.putText(out,label,(x+3,y-2),cv2.FONT_HERSHEY_SIMPLEX,.47,(0,0,0),1,cv2.LINE_AA)
    header=np.zeros((58,frame.shape[1],3),dtype=np.uint8)
    cv2.putText(header,title,(12,23),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(header,f'{idx/fps:05.1f}s | conf >= {conf:.2f} | imgsz {imgsz} | severidad: sin evaluar',(12,46),cv2.FONT_HERSHEY_SIMPLEX,.43,(205,205,205),1,cv2.LINE_AA)
    footer=np.full((28,frame.shape[1],3),22,dtype=np.uint8)
    counts=Counter(d['class'] for d in items)
    text=' | '.join(f'{SHORT.get(k,k)}: {v}' for k,v in counts.items()) or 'Sin detecciones sobre el umbral'
    cv2.putText(footer,text,(12,19),cv2.FONT_HERSHEY_SIMPLEX,.43,(230,230,230),1,cv2.LINE_AA)
    return np.vstack([header,out,footer])

def encode(raw,original,destination):
    command=[shutil.which('ffmpeg'),'-hide_banner','-loglevel','error','-n','-i',str(raw),'-i',str(original),'-map','0:v:0','-map','1:a?',
             '-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-c:a','copy','-map_metadata','-1','-movflags','+faststart',str(destination)]
    subprocess.run(command,check=True)

def sheets(frames,names,output,columns=4):
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    thumbw=430
    thumbh=290
    canvas=Image.new('RGB',(columns*thumbw,((len(frames)+columns-1)//columns)*thumbh),'white')
    draw=ImageDraw.Draw(canvas)
    for i,frame in enumerate(frames):
        rgb=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        rgb.thumbnail((thumbw-8,thumbh-32))
        x=(i%columns)*thumbw;y=(i//columns)*thumbh
        canvas.paste(rgb,(x+(thumbw-rgb.width)//2,y))
        draw.text((x+5,y+thumbh-28),names[i],font=font,fill='black')
    canvas.save(output,quality=92)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--model',choices=list(MODELS))
    parser.add_argument('--probe-only',action='store_true')
    parser.add_argument('--combine',action='store_true')
    parser.add_argument('--benchmark',action='store_true')
    parser.add_argument('--imgsz',type=int,default=1280)
    parser.add_argument('--conf',type=float,default=.25)
    parser.add_argument('--threads',type=int,default=8)
    args=parser.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(args.threads)
    args.video=args.video.resolve();args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
    metadata=probe(args.video)
    video_stream=next(s for s in metadata['streams'] if s['codec_type']=='video')
    cap=cv2.VideoCapture(str(args.video))
    assert cap.isOpened()
    fps=cap.get(cv2.CAP_PROP_FPS)
    n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.benchmark:
        ok,frame=cap.read();assert ok;cap.release()
        torch.set_num_threads(1)
        model=YOLO(str(MODELS['yolo12s_rdd2022']['path']))
        for _ in range(2):model.predict(frame,imgsz=args.imgsz,conf=args.conf,device='cpu',verbose=False)
        for threads in [1,2,4]:
            torch.set_num_threads(threads)
            started=time.perf_counter()
            for _ in range(3):model.predict(frame,imgsz=args.imgsz,conf=args.conf,device='cpu',verbose=False)
            print(json.dumps(dict(threads=threads,seconds_per_frame=(time.perf_counter()-started)/3)),flush=True)
        return
    if args.probe_only:
        images=[];labels=[]
        for second in [0,10,20,30,45,60,75,90]:
            cap.set(cv2.CAP_PROP_POS_MSEC,second*1000);ok,frame=cap.read();assert ok
            images.append(frame);labels.append(f'{second}s')
        sheets(images,labels,args.out/'video_original_muestra.jpg')
        timings={}
        for key,spec in MODELS.items():
            model=YOLO(str(spec['path']))
            start=time.perf_counter()
            for frame in images[:2]:
                model.predict(frame,imgsz=args.imgsz,conf=args.conf,device='cpu',verbose=False)
            timings[key]=dict(names=model.names,task=model.task,two_frame_seconds=time.perf_counter()-start,path=str(spec['path']))
            del model
        cap.release()
        info=dict(video=str(args.video),video_sha256=digest(args.video),metadata=metadata,fps=fps,frames=n,width=w,height=h,models=timings,imgsz=args.imgsz,conf=args.conf)
        (args.out/'preparacion.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(info,ensure_ascii=False,indent=2),flush=True)
        return
    if args.combine:
        cap.release()
        captures={k:cv2.VideoCapture(str(args.out/(k+'.mp4'))) for k in MODELS}
        assert all(c.isOpened() for c in captures.values())
        original=cv2.VideoCapture(str(args.video))
        raw=args.out/'comparacion_4_paneles_temporal.mp4'
        dest=args.out/'comparacion_4_paneles.mp4'
        writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w*2,(h+86)*2))
        assert writer.isOpened()
        count=0;images=[];labels=[]
        picked={int(s*fps) for s in [0,10,20,30,40,50,60,70,80,90,98]}
        while True:
            ok,frame=original.read()
            if not ok:break
            processed={}
            for key,c in captures.items():
                ret,img=c.read();assert ret,(key,count)
                processed[key]=img
            basepanel=panel(frame,[],'Original',count,fps,args.conf,args.imgsz)
            cv2.rectangle(basepanel,(0,h+58),(w,h+86),(22,22,22),-1)
            cv2.putText(basepanel,'Video original | Las marcas son predicciones, no etiquetas verificadas',(12,h+77),cv2.FONT_HERSHEY_SIMPLEX,.42,(230,230,230),1,cv2.LINE_AA)
            joined=np.vstack([np.hstack([basepanel,processed['yolo12s_rdd2022']]),np.hstack([processed['farzad_segmentacion'],processed['piloto_v1_previo']])])
            writer.write(joined)
            if count in picked:
                images.append(joined);labels.append(f'{count/fps:.1f}s')
                captures_dir=args.out/'capturas_comparacion'
                captures_dir.mkdir(exist_ok=True)
                cv2.imencode('.jpg',joined,[cv2.IMWRITE_JPEG_QUALITY,94])[1].tofile(captures_dir/f'{count:05d}_{count/fps:05.1f}s.jpg')
            count+=1
        writer.release();original.release()
        for c in captures.values():
            assert not c.read()[0], 'Sobran fotogramas en un resultado'
            c.release()
        assert count==n
        encode(raw,args.video,dest)
        sheets(images,labels,args.out/'resumen_comparacion.jpg',columns=3)
        print(json.dumps(dict(comparison=str(dest),frames=count,metadata=probe(dest)),ensure_ascii=False),flush=True)
        return
    assert args.model
    key=args.model;spec=MODELS[key]
    spec=dict(spec,path=str(spec['path']),sha256=digest(spec['path']))
    model=YOLO(spec['path'])
    assert (model.task=='segment')==(spec['kind']=='segment')
    raw=args.out/(key+'_temporal.mp4');dest=args.out/(key+'.mp4')
    assert not dest.exists(), 'El resultado ya existe; usar una carpeta de salida nueva.'
    writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h+86))
    assert writer.isOpened()
    totals=Counter();frames_by_class=Counter();max_conf={};frames_with=0;idx=0;started=time.perf_counter()
    sample_images=[];sample_labels=[]
    sample_frames={int(s*fps) for s in [0,10,20,30,40,50,60,70,80,90,98]}
    json_path=args.out/(key+'_detecciones.jsonl')
    csv_path=args.out/(key+'_detecciones.csv')
    with json_path.open('w',encoding='utf-8') as jf,csv_path.open('w',encoding='utf-8-sig',newline='') as cf:
        cw=csv.DictWriter(cf,['frame','seconds','class','source_class','confidence','x1','y1','x2','y2','severity'])
        cw.writeheader()
        while True:
            ok,frame=cap.read()
            if not ok:break
            result=model.predict(frame,imgsz=args.imgsz,conf=args.conf,iou=.7,device='cpu',verbose=False,retina_masks=(spec['kind']=='segment'))[0]
            items=[]
            boxes=result.boxes.xyxy.cpu().tolist();scores=result.boxes.conf.cpu().tolist();classes=result.boxes.cls.int().cpu().tolist()
            polygons=result.masks.xy if result.masks is not None else []
            for j,(box,score,cls) in enumerate(zip(boxes,scores,classes)):
                source_class=str(model.names[cls]);label=canonical(source_class,key)
                item=dict(class_id=cls,source_class=source_class,**{'class':label},confidence=round(score,6),box=[round(v,3) for v in box],severity=None)
                if spec['kind']=='segment' and j<len(polygons):item['polygon']=np.round(polygons[j],2).tolist()
                items.append(item)
                totals[label]+=1;max_conf[label]=max(max_conf.get(label,0),score)
                cw.writerow(dict(frame=idx,seconds=round(idx/fps,6),**{'class':label},source_class=source_class,confidence=round(score,6),**dict(zip(['x1','y1','x2','y2'],item['box'])),severity='sin_evaluar'))
            frames_by_class.update(set(d['class'] for d in items))
            frames_with+=bool(items)
            jf.write(json.dumps(dict(frame=idx,seconds=round(idx/fps,6),detections=items),ensure_ascii=False)+'\n')
            out=panel(frame,items,spec['title'],idx,fps,args.conf,args.imgsz,spec['kind']=='segment')
            writer.write(out)
            if idx in sample_frames:sample_images.append(out);sample_labels.append(f'{idx/fps:.1f}s | {len(items)} marcas')
            idx+=1
            if idx%50==0:
                elapsed=time.perf_counter()-started
                progress=dict(model=key,frames=idx,total=n,elapsed_seconds=round(elapsed,1),estimated_remaining_seconds=round((n-idx)*elapsed/idx,1))
                (args.out/(key+'_progreso.json')).write_text(json.dumps(progress),encoding='utf-8')
                print(json.dumps(progress),flush=True)
    cap.release();writer.release()
    assert idx==n,(idx,n)
    encode(raw,args.video,dest)
    output_meta=probe(dest)
    out_stream=next(s for s in output_meta['streams'] if s['codec_type']=='video')
    assert int(out_stream['nb_read_frames'])==n
    assert any(s['codec_type']=='audio' for s in output_meta['streams'])==any(s['codec_type']=='audio' for s in metadata['streams'])
    assert abs(float(output_meta['format']['duration'])-float(metadata['format']['duration']))<.15
    assert digest(args.video)==json.loads((args.out/'preparacion.json').read_text(encoding='utf-8'))['video_sha256']
    sheets(sample_images,sample_labels,args.out/(key+'_muestra.jpg'))
    summary=dict(model=key,checkpoint=spec,classes=model.names,video=str(args.video),output=str(dest),frames_processed=idx,frames_with_detections=frames_with,detection_observations_by_class=dict(totals),frames_by_class=dict(frames_by_class),max_confidence=max_conf,imgsz=args.imgsz,confidence=args.conf,iou=.7,device='cpu',threads=args.threads,elapsed_seconds=round(time.perf_counter()-started,2),input_sha256=digest(args.video),output_sha256=digest(dest),output_metadata=output_meta,note='Observaciones repetidas por fotograma, no daños únicos ni medición de precisión. Severidad sin evaluar.')
    (args.out/(key+'_resumen.json')).write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
