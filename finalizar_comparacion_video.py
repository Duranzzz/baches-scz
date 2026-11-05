"""Espera los resultados locales, compone la comparación y verifica los videos entregables."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
import zipfile
from datetime import datetime,timezone
from pathlib import Path

from comparar_video_local import MODELS, digest, probe

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--wait',action='store_true')
    args=p.parse_args()
    out=args.out.resolve();video=args.video.resolve()
    ready={k:out/(k+'_resumen.json') for k in MODELS}
    last=None
    start=time.monotonic()
    while not all(p.is_file() for p in ready.values()):
        if not args.wait:raise RuntimeError('Faltan resultados de algunos modelos.')
        done=[k for k,v in ready.items() if v.is_file()]
        value=','.join(done)
        if value!=last or not last:
            print(json.dumps({'videos_terminados':done,'estado':'esperando inferencia'}),flush=True)
            last=value
        if time.monotonic()-start>10800:raise TimeoutError('La comparación superó tres horas de espera.')
        time.sleep(10)
    if not (out/'comparacion_4_paneles.mp4').exists():
        subprocess.run([sys.executable,str(Path(__file__).with_name('comparar_video_local.py')),'--video',str(video),'--out',str(out),'--combine'],check=True)
    summaries={k:json.loads(path.read_text(encoding='utf-8')) for k,path in ready.items()}
    initial=json.loads((out/'preparacion.json').read_text(encoding='utf-8'))
    assert digest(video)==initial['video_sha256']
    n=initial['frames'];duration=float(initial['metadata']['format']['duration'])
    outputs={}
    for k,path in {**{k:out/(k+'.mp4') for k in MODELS},'comparacion':out/'comparacion_4_paneles.mp4'}.items():
        meta=probe(path)
        stream=next(s for s in meta['streams'] if s['codec_type']=='video')
        assert int(stream['nb_read_frames'])==n
        assert stream['codec_name']=='h264' and stream['pix_fmt']=='yuv420p'
        assert any(s['codec_type']=='audio' for s in meta['streams'])
        assert abs(float(meta['format']['duration'])-duration)<.15
        outputs[k]=dict(file=str(path),sha256=digest(path),bytes=path.stat().st_size,frames=n,duration=float(meta['format']['duration']),width=stream['width'],height=stream['height'],audio='AAC original conservado')
        if k in summaries:
            s=summaries[k]
            assert digest(s['checkpoint']['path'])==s['checkpoint']['sha256']
            assert s['frames_processed']==n
            lines=(out/(k+'_detecciones.jsonl')).read_text(encoding='utf-8').splitlines()
            assert len(lines)==n
            records=[json.loads(line) for line in lines]
            assert [r['frame'] for r in records]==list(range(n))
            assert all(d['severity'] is None for r in records for d in r['detections'])
            assert sum(len(r['detections']) for r in records)==sum(s['detection_observations_by_class'].values())
    rows=[]
    for k,s in summaries.items():
        counts=s['detection_observations_by_class']
        rows.append(dict(modelo=s['checkpoint']['title'],fotogramas=n,fotogramas_con_marcas=s['frames_with_detections'],observaciones_bache=counts.get('bache',0),observaciones_grieta_longitudinal=counts.get('grieta_longitudinal',0),observaciones_grieta_transversal=counts.get('grieta_transversal',0),observaciones_grieta_cocodrilo=counts.get('grieta_cocodrilo',0),observaciones_parche=counts.get('parche',0),archivo=k+'.mp4'))
    with (out/'comparacion_resumen.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)
    tables='\n'.join('| '+r['modelo']+' | '+str(r['fotogramas_con_marcas'])+' | '+str(r['observaciones_bache'])+' | '+str(r['observaciones_grieta_longitudinal']+r['observaciones_grieta_transversal']+r['observaciones_grieta_cocodrilo'])+' | '+str(r['observaciones_parche'])+' |' for r in rows)
    report=f'''# Comparación del video de WhatsApp — 5 de noviembre de 2026

Se procesó el video completo con los tres checkpoints locales disponibles, sin descargar otros modelos ni ejecutar entrenamiento. Duración original: {duration:.3f} s. {n} fotogramas, {initial['width']} × {initial['height']} píxeles. Cada modelo analizó todos los fotogramas.

## Videos

- [Comparación de cuatro paneles](comparacion_4_paneles.mp4): original arriba izquierda, YOLO12s arriba derecha, Farzad abajo izquierda y piloto previo abajo derecha.
- [YOLO12s RDD2022](yolo12s_rdd2022.mp4): grietas, baches y su clase Repair.
- [Farzad YOLOv8n-seg](farzad_segmentacion.mp4): baches con contornos y cajas.
- [Piloto v1 previo](piloto_v1_previo.mp4): referencia del modelo anterior de dos épocas, ya existente; no es el nuevo piloto v2 ni un modelo final.

Todos los MP4 finales están en H.264 y conservan el audio original AAC y la duración. La cabecera y el pie se añadieron fuera de la imagen de la carretera; el video original no se modificó. No se saltaron fotogramas, no se usó interpolación de detecciones ni se aplicó seguimiento o suavizado temporal.

## Configuración común

Tamaño de inferencia 1280, umbral de confianza 0,25, NMS IoU 0,70, FP32 en CPU. Estos umbrales permiten una comparación visual con parámetros comunes; las confianzas de modelos distintos no están calibradas entre sí. El tamaño original del video se conserva al dibujar.

Rojo: bache; naranja: grieta longitudinal; cian: transversal; magenta: cocodrilo; gris: Repair/parche. En Farzad el área coloreada es la predicción de su máscara; no es área física medida.

## Observaciones por fotograma

| Modelo | Fotogramas con alguna marca | Observaciones de bache | Observaciones de grieta | Observaciones de Repair |
|---|---:|---:|---:|---:|
{tables}

Una lesión puede aparecer cientos de veces al continuar visible en el video. Estos números **no son el conteo de daños únicos ni una medida de precisión**. Un mayor número de marcas puede incluir más falsos positivos. No se dispone de etiquetas exhaustivas de este video para calcular precisión, recall o mAP.

La severidad queda **sin evaluar** en todos los resultados; no se deduce del tamaño del cuadro, de la máscara ni de la confianza. Hay CSV por modelo con coordenadas/confianza y JSONL por cada fotograma, incluidos los que no tienen detecciones. Los resúmenes JSON incluyen nombres originales de clases y huellas de cada checkpoint para reproducir la prueba.

Se verificaron los cuatro videos finales: {n} fotogramas cada uno, audio presente, duración coincidente y decodificación completa con ffprobe. Se comprobaron la integridad del original y la de los pesos. El hash del video original es `{initial['video_sha256']}`.
'''
    (out/'RESULTADOS.md').write_text(report,encoding='utf-8')
    manifest=dict(status='verified',verified_at_utc=datetime.now(timezone.utc).isoformat(),input_sha256=initial['video_sha256'],original_unchanged=True,frames_per_model=n,trained_new_model=False,outputs=outputs)
    (out/'verificacion_final.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    package=out/'videos_y_detecciones.zip'
    with zipfile.ZipFile(package,'w') as z:
        for file in sorted(out.iterdir()):
            if not file.is_file() or file==package or '_temporal' in file.name or '_progreso' in file.name:continue
            if file.suffix not in {'.mp4','.csv','.json','.jsonl','.md','.jpg'}:continue
            z.write(file,file.name,compress_type=zipfile.ZIP_STORED if file.suffix=='.mp4' else zipfile.ZIP_DEFLATED,compresslevel=3)
    with zipfile.ZipFile(package) as z:assert z.testzip() is None
    print(json.dumps(dict(status='completo',outputs=outputs,zip=str(package),rows=rows),ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
