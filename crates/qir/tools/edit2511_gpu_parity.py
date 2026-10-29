"""Build deterministic ~1MP graph and worker requests; replay ComfyUI and compare RGB."""
import argparse, copy, json, time, urllib.request, urllib.parse
from pathlib import Path
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'target/edit2511/evidence'
COMFY = Path('C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI')
WORKFLOWS = Path('C:/Users/5090/Documents/PhotoshopEX/crates/genai/workflows')
TEMPLATES = ['edit-lightning-8','fill-lightning-8','remove-lightning-8','expand-lightning-8','fill-ref-lightning-8','image-ref','angles','edit']

def fixtures():
    OUT.mkdir(parents=True, exist_ok=True)
    # A real installed ComfyUI input, held fixed for all routes.
    source = Image.open(COMFY / 'input/bench-lighthouse.png').convert('RGB').resize((1024,1024),Image.Resampling.LANCZOS)
    reference = source.resize((768,1024),Image.Resampling.LANCZOS)
    reference2 = source.resize((1024,640),Image.Resampling.LANCZOS)
    mask = Image.new('RGB',source.size); ImageDraw.Draw(mask).ellipse((280,240,744,780),fill=(255,255,255))
    for name,im in [('picture',source),('reference',reference),('reference2',reference2),('mask',mask)]:
        im.save(OUT / (name+'.png')); (OUT / (name+'.rgb')).write_bytes(im.tobytes())
    import numpy as np
    crop=np.asarray(source)[128:896,128:896]
    canvas=Image.fromarray(np.pad(crop,((128,128),(128,128),(0,0)),mode='edge'))
    emask=Image.new('RGB',source.size,(255,255,255)); ImageDraw.Draw(emask).rectangle((128,128,895,895),fill=(0,0,0))
    for name,im in [('expand',canvas),('expand-mask',emask)]:
        im.save(OUT/(name+'.png'));(OUT/(name+'.rgb')).write_bytes(im.tobytes())
    cases=[]
    for suffix in TEMPLATES:
        meta=json.loads((WORKFLOWS / ('qwen-edit-2511-'+suffix+'.json')).read_text())['meta']
        names=[]
        def blob(name):
            im=Image.open(OUT/(name+'.png')); idx=len(names); names.append(str(OUT/(name+'.rgb')))
            return dict(width=im.width,height=im.height,channels=3,index=idx)
        req=dict(model='qwen-edit-2511/'+suffix,task=meta['task'],prompt='Change the sky to a warm sunset with soft pink clouds. Keep the lighthouse and landscape unchanged.',negative='',seed=43,steps=meta['defaults']['steps'],guidance=meta['defaults']['guidance'],width=1024,height=1024,params={})
        if suffix=='remove-lightning-8': req['prompt']='Remove the lighthouse and fill its place with natural landscape.'
        if suffix=='angles': req['prompt']='<sks> front view eye-level shot medium shot'
        if suffix=='image-ref': req['references']=[blob('picture'),blob('reference'),blob('reference2')]
        else: req['image']=blob('expand' if suffix.startswith('expand') else 'picture')
        if meta.get('needsMask'): req['mask']=blob('expand-mask' if suffix.startswith('expand') else 'mask')
        if suffix=='fill-ref-lightning-8': req['references']=[blob('reference'),blob('reference2')]
        if suffix.startswith('expand'): req['params']=dict(ref_x=128,ref_y=128,ref_w=768,ref_h=768)
        cases.append(dict(request=req,blobs=names))
    (OUT/'manifest.json').write_text(json.dumps(dict(models=str(COMFY/'models'),out=str(OUT/'native'),cases=cases),indent=2))

def api(path,data=None):
    raw=None if data is None else json.dumps(data).encode()
    return json.load(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8188'+path,data=raw,headers={'Content-Type':'application/json'})))

def replay(only):
    cases=json.loads((OUT/'manifest.json').read_text())['cases']; dest=OUT/'comfy'; dest.mkdir(exist_ok=True)
    for case in cases:
        req=case['request']; suffix=req['model'].split('/')[1]
        if only and suffix not in only.split(','): continue
        template=json.loads((WORKFLOWS/('qwen-edit-2511-'+suffix+'.json')).read_text())
        graph=template['graph']; meta=template['meta']
        vals=dict(prompt=req['prompt'],negative='',steps=req['steps'],cfg=req['guidance'],width=1024,height=1024,sampler='euler',scheduler='simple',denoise=1.0,**req['params'])
        for model in meta['models']: vals[model['placeholder']]=model['default']
        # LoadImage accepts absolute input paths; all writes stay in the worktree.
        vals.update(image=('expand.png' if suffix.startswith('expand') else 'picture.png'),mask=('expand-mask.png' if suffix.startswith('expand') else 'mask.png'),image1='picture.png',image2='reference.png',image3='reference2.png')
        refs=meta.get('references',{})
        placeholders=refs.get('placeholders',[])
        present=3 if suffix=='image-ref' else (2 if suffix=='fill-ref-lightning-8' else 0)
        remove=[]
        for ph in placeholders[present:]:
            remove.extend(refs.get('nodes',{}).get(ph,[]))
        for key in remove: graph.pop(key,None)
        for node in graph.values():
            for key,v in list(node['inputs'].items()):
                if isinstance(v,list) and v[0] in remove: del node['inputs'][key]
        for warm in range(2):
            vals.update(seed=42+warm,prefix='edit2511-'+suffix+'-'+str(warm))
            def subst(v):
                if isinstance(v,str) and v.startswith('{{') and v.endswith('}}'): return vals[v[2:-2]]
                if isinstance(v,dict): return {k:subst(x) for k,x in v.items()}
                if isinstance(v,list): return [subst(x) for x in v]
                return v
            begin=time.perf_counter(); response=api('/prompt',{'prompt':subst(graph)})
            ident=response['prompt_id']
            while True:
                hist=api('/history/'+ident).get(ident)
                if hist: break
                time.sleep(.1)
            seconds=time.perf_counter()-begin
            (dest/(suffix+'-'+str(warm)+'.json')).write_text(json.dumps(dict(seconds=seconds,history=hist),indent=2))
            if hist['status']['status_str']!='success': raise RuntimeError(hist['status'])
            entry=hist['outputs'][meta['saveNode']]['images'][0]
            url='http://127.0.0.1:8188/view?'+urllib.parse.urlencode(entry)
            (dest/(suffix+'-'+str(warm)+'.png')).write_bytes(urllib.request.urlopen(url).read())
            print(suffix,warm,seconds,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['fixtures','comfy','native','compare']);p.add_argument('--case');p.add_argument('--binary',type=Path);p.add_argument('--native-dir',type=Path);p.add_argument('--baseline',type=Path);p.add_argument('--phase-profile',action='store_true');p.add_argument('--profile',action='store_true');a=p.parse_args()
    if a.action=='fixtures': fixtures()
    elif a.action=='comfy': replay(a.case)
    elif a.action=='native':
        import subprocess, os
        env=os.environ.copy();env.update(QIR_PROFILE='1' if a.profile else '0',QIR_PHASE_PROFILE='1' if a.phase_profile else '0',RAYON_NUM_THREADS='8',OMP_NUM_THREADS='8')
        env.setdefault('RUST_LOG','info' if a.profile else 'warn')
        if a.case:env['CASE']=a.case
        else:env.pop('CASE',None)
        samples=[]
        manifest=OUT/'manifest.json'
        if a.native_dir:
            value=json.loads(manifest.read_text());value['out']=str(a.native_dir.resolve());manifest=OUT/'replay-manifest.json';manifest.write_text(json.dumps(value))
        with (OUT/'native-all.log').open('w') as log:
            proc=subprocess.Popen([str(a.binary.resolve() if a.binary else ROOT/'target/edit2511/debug/examples/edit2511_parity.exe'),str(manifest)],env=env,stdout=log,stderr=log)
            while proc.poll() is None:
                raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
                memory,util=map(int,raw.split(','));samples.append(dict(time=time.time(),memory_mib=memory,utilization=util));time.sleep(1)
        (OUT/'native-vram.json').write_text(json.dumps(dict(peak_mib=max(s['memory_mib'] for s in samples),samples=samples)))
        raise SystemExit(proc.returncode)
    else:
        import numpy as np
        rows=[]
        failed=False
        for suffix in TEMPLATES:
            if a.case and suffix not in a.case.split(','): continue
            n=(a.native_dir or OUT/'native')/f'{suffix}-1.json'; c=OUT/'comfy'/f'{suffix}-1.png'
            if not n.exists() or not c.exists(): raise SystemExit(f'missing comparison artifact: {n} or {c}')
            info=json.loads(n.read_text()); im=info['done']['images'][0]
            native=np.frombuffer((n.with_suffix('.rgba')).read_bytes(),np.uint8).reshape(im['height'],im['width'],4)[:,:,:3].astype(float)/255
            oracle=np.asarray(Image.open(c).convert('RGB')).astype(float)/255
            d=native-oracle; mse=np.mean(d*d)
            row=dict(template=suffix,mean_abs_rgb=float(np.mean(abs(d))),psnr=float(-10*np.log10(mse)),native_seconds=info['seconds'],comfy_seconds=json.loads(c.with_suffix('.json').read_text())['seconds'])
            if a.baseline:
                before=np.frombuffer((a.baseline/f'{suffix}-1.rgba').read_bytes(),np.uint8).reshape(im['height'],im['width'],4)[:,:,:3].astype(float)/255
                delta=abs(native-before)
                old_mse=np.mean((before-oracle)**2)
                row.update(before_after_mean_abs_rgb=float(delta.mean()),byte_identical=bool(np.array_equal(native,before)),baseline_psnr=float(-10*np.log10(old_mse)))
                row['gate_pass']=row['before_after_mean_abs_rgb']<=1/255 and row['psnr']+1e-9>=row['baseline_psnr']
                failed=failed or not row['gate_pass']
            rows.append(row);print(row)
        (OUT/'comparison.json').write_text(json.dumps(rows,indent=2))
        if failed: raise SystemExit('2511 parity gate failed')
