"""Convierte el mismo YOLO12s a FP32 y comprueba concordancia con fotogramas ya calculados."""
import ast
import json
import shutil
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import onnxruntime as ort
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.engine.results import Results
from ultralytics.utils import nms,ops

class LocalOnnxDetector:
    task='detect'
    def __init__(self,path,threads=4):
        self.path=Path(path)
        options=ort.SessionOptions();options.intra_op_num_threads=threads;options.inter_op_num_threads=1
        options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        self.names=ast.literal_eval(self.session.get_modelmeta().custom_metadata_map['names'])
        self.shape=self.session.get_inputs()[0].shape
        self.transform=LetterBox(new_shape=tuple(self.shape[2:]),auto=False,stride=32)
    def predict(self,frame,conf=.25,iou=.7,**kwargs):
        im=self.transform(image=frame)
        im=np.ascontiguousarray(im[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255.
        raw=self.session.run(None,{self.session.get_inputs()[0].name:im})[0]
        pred=nms.non_max_suppression(torch.from_numpy(raw),conf_thres=conf,iou_thres=iou,nc=len(self.names),max_det=300)[0]
        if len(pred):pred[:,:4]=ops.scale_boxes(tuple(self.shape[2:]),pred[:,:4],frame.shape)
        return [Results(orig_img=frame,path='video',names=self.names,boxes=pred)]

def main():
    root=Path(__file__).resolve().parent
    out=root/'salidas/videos/whatsapp_20261104_comparacion_20261105'
    folder=out/'motor_onnx';folder.mkdir(exist_ok=True)
    checkpoint=folder/'yolo12s_rdd2022.pt'
    if not checkpoint.exists():shutil.copy2(root/'modelos/yolo12s_rdd2022/yolo12s_RDD2022_best.pt',checkpoint)
    path=checkpoint.with_suffix('.onnx')
    torch.set_num_threads(4);cv2.setNumThreads(1)
    if not path.exists():YOLO(str(checkpoint)).export(format='onnx',imgsz=[736,1280],batch=1,dynamic=False,simplify=False,opset=17,device='cpu')
    model=LocalOnnxDetector(path,4)
    cached={}
    for line in (out/'yolo12s_rdd2022_detecciones.jsonl').read_text(encoding='utf-8').splitlines():
        try:
            r=json.loads(line);cached[r['frame']]=r['detections']
        except json.JSONDecodeError:pass
    chosen=sorted(set([0,1,2,30,60,90,120,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900])&set(cached))
    cap=cv2.VideoCapture(str(root/'videos/WhatsApp Video 2026-11-04 at 08.45.28.mp4'))
    diffs=[];started=time.perf_counter()
    for idx in chosen:
        cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,frame=cap.read();assert ok
        pred=model.predict(frame)[0]
        found=[dict(class_id=int(k),box=b,confidence=float(c)) for b,c,k in zip(pred.boxes.xyxy.tolist(),pred.boxes.conf.tolist(),pred.boxes.cls.tolist())]
        expected=cached[idx]
        assert len(expected)==len(found),(idx,expected,found)
        for a,b in zip(sorted(expected,key=lambda x:(x['class_id'],-x['confidence'])),sorted(found,key=lambda x:(x['class_id'],-x['confidence']))):
            assert a['class_id']==b['class_id']
            boxdiff=max(abs(x-y) for x,y in zip(a['box'],b['box']))
            confdiff=abs(a['confidence']-b['confidence'])
            assert boxdiff<.1 and confdiff<.0001,(idx,boxdiff,confdiff)
            diffs.append((boxdiff,confdiff))
    cap.release()
    report=dict(status='passed',same_checkpoint=True,quantization=False,precision='FP32',frames_checked=chosen,max_box_difference_pixels=max([d[0] for d in diffs] or [0]),max_conf_difference=max([d[1] for d in diffs] or [0]),seconds_per_frame=(time.perf_counter()-started)/len(chosen),path=str(path))
    (out/'concordancia_onnx.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
