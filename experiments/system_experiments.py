"""Bounded workstation benchmarks and local two-process proof of concept.
Synthetic amplification preserves the observed district-year support and is
only a computational workload, never additional epidemiological evidence.
"""
from __future__ import annotations
import argparse, hashlib, io, json, os, platform, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'/'revision';OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
from pipeline.core import RunContext
from pipeline.modules.space_time_scan import SpaceTimeScanModule
from revision_experiments import load
threadpool_limits(1)

def dump(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2,default=str))
def hashes():
    paths=[ROOT/'pipeline/core.py',*sorted((ROOT/'pipeline/modules').glob('*.py')),ROOT/'pipeline/covariates.py',ROOT/'etl/normalize.py',ROOT/'api/main.py']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
def worker(mult):
    ev=load('anthrax').dropna(subset=['latitude','longitude','year','district_raion'])
    ev=pd.concat([ev]*mult,ignore_index=True)
    base=psutil.Process().memory_info().rss;peak=[base];stop=threading.Event()
    def mem():
        while not stop.wait(.005):peak[0]=max(peak[0],psutil.Process().memory_info().rss)
    th=threading.Thread(target=mem);th.start();t=time.perf_counter()
    r=SpaceTimeScanModule().run(RunContext('anthrax',ev,{'n_permutations':199,'random_seed':42}))
    elapsed=time.perf_counter()-t;stop.set();th.join()
    print(json.dumps({'multiplier':mult,'n':len(ev),'seconds':elapsed,'events_per_second':len(ev)/elapsed,
        'rss_before_mib':base/2**20,'peak_rss_mib':peak[0]/2**20,'extra_peak_mib':(peak[0]-base)/2**20,
        'n_districts':r.summary['n_districts'],'n_windows':r.summary['n_candidate_windows'],'n_zones':r.summary['n_candidate_zones']}))
def main():
    before=hashes();rows=[]
    for mult in [1,2,5,10]:
        for rep in range(3):
            p=subprocess.run([sys.executable,__file__,'--worker',str(mult)],capture_output=True,text=True,check=True)
            row=json.loads(p.stdout.strip().splitlines()[-1]);row['repeat']=rep;rows.append(row)
            pd.DataFrame(rows).to_csv(OUT/'workstation_scaling.csv',index=False)
            print('scale',mult,rep,round(row['seconds'],2),flush=True)
    # Use private upload directory, keeping the original API source unchanged.
    import api.main as api
    import tempfile;api.UPLOAD_DIR=Path(tempfile.mkdtemp(prefix='api_uploads_'))  # uploaded registry copies stay out of the repository
    import uvicorn,httpx
    server=uvicorn.Server(uvicorn.Config(api.app,host='127.0.0.1',port=18761,log_level='error'))
    api_thread=threading.Thread(target=server.run,daemon=True);api_thread.start()
    for _ in range(100):
        if server.started:break
        time.sleep(.05)
    if not server.started:raise RuntimeError('local API failed to start')
    web=subprocess.Popen([sys.executable,'-m','http.server','18762','--bind','127.0.0.1','--directory',str(ROOT/'web')],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                wr=httpx.get('http://127.0.0.1:18762',timeout=1)
                if wr.status_code==200:break
            except httpx.HTTPError:pass
            time.sleep(.05)
        base='http://127.0.0.1:18761';data=load('rabies')
        cases=[]
        variants=[('original',data.copy(),200),('reordered_columns',data[data.columns[::-1]],200),
            ('extra_column',data.assign(unused_field='ignored'),200),('numeric_strings',data.astype({'latitude':str,'longitude':str,'year':str}),200),
            ('missing_required_column',data.drop(columns=['year']),422),('unknown_disease_value',data.copy(),200)]
        def upload(d,disease='rabies'):
            return httpx.post(base+'/datasets',data={'disease':disease},files={'file':('input.csv',d.to_csv(index=False).encode(),'text/csv')},timeout=30)
        for name,d,status in variants:
            resp=upload(d,'test_disease' if name=='unknown_disease_value' else 'rabies')
            cases.append({'case':name,'expected_status':status,'actual_status':resp.status_code,'passed':resp.status_code==status})
        for name,col in [('unparseable_latitude','latitude'),('missing_year_value','year')]:
            d=data.copy();d[col]=d[col].astype(object);d.loc[0,col]='bad-number'
            resp=upload(d);jid=resp.json()['dataset_id']
            run=httpx.post(base+'/runs',data={'dataset_id':jid,'params':json.dumps({'n_permutations':19,'max_spatial_fraction':.1,'max_temporal_fraction':.1})}).json()
            for _ in range(600):
                rr=httpx.get(base+'/runs/'+run['run_id']).json()
                if rr['status'] in ['done','failed']:break
                time.sleep(.02)
            cases.append({'case':name,'status':rr['status'],'n_events_used':rr.get('summary',{}).get('n_events_used'),
                'passed':rr['status']=='done' and rr['summary']['n_events_used']==362})
        bad=httpx.post(base+'/datasets',data={'disease':'rabies'},files={'file':('bad.csv',data.to_csv(index=False,sep=';').encode(),'text/csv')})
        cases.append({'case':'unsupported_semicolon_delimiter','actual_status':bad.status_code,'passed':bad.status_code==422})
        pd.DataFrame(cases).to_csv(OUT/'schema_robustness.csv',index=False)
        # Update handled as a new uploaded version; no in-place persistence claim.
        first=upload(data).json();second=upload(pd.concat([data,data.iloc[:1]],ignore_index=True)).json()
        update={'old_rows':first['row_count'],'new_rows':second['row_count'],'different_id':first['dataset_id']!=second['dataset_id']}
        dataset_id=first['dataset_id'];concurrent=[]
        def request_run(_):
            r=httpx.post(base+'/runs',data={'dataset_id':dataset_id,'params':json.dumps({'n_permutations':199,'random_seed':42})},timeout=30)
            return r.status_code,r.json()['run_id']
        for n in [1,2,4]:
            proc=psutil.Process();peak=proc.memory_info().rss;t=time.perf_counter()
            with ThreadPoolExecutor(max_workers=n) as pool:jobs=list(pool.map(request_run,range(n)))
            statuses={}
            for _ in range(6000):
                statuses={rid:httpx.get(base+'/runs/'+rid,timeout=30).json() for _,rid in jobs}
                peak=max(peak,proc.memory_info().rss)
                if all(r['status'] in ['done','failed'] for r in statuses.values()):break
                time.sleep(.025)
            elapsed=time.perf_counter()-t
            successful=sum(r['status']=='done' for r in statuses.values())
            geo_ok=sum(httpx.get(base+'/runs/'+rid+'/geojson').status_code==200 for rid,r in statuses.items() if r['status']=='done')
            concurrent.append({'concurrent_requests':n,'successful':successful,'geojson_ok':geo_ok,'wall_seconds':elapsed,'runs_per_second':successful/elapsed,'peak_process_rss_mib':peak/2**20})
        pd.DataFrame(concurrent).to_csv(OUT/'local_api_concurrency.csv',index=False)
        dump('local_poc.json',{'api_http_status':httpx.get(base+'/modules').status_code,'web_http_status':wr.status_code,
            'web_contains_api_base_control':'api' in wr.text.lower(),'api_modules':httpx.get(base+'/modules').json(),
            'update_upload':update,'core_source_unchanged':before==hashes(),'n_robustness_passed':sum(c['passed'] for c in cases),'n_robustness_cases':len(cases),
            'scope':'Local HTTP API plus independent static web process; no browser interaction, Docker, database, distributed services or durability test.'})
    finally:
        server.should_exit=True;api_thread.join(timeout=5);web.terminate();web.wait(timeout=5)
    dump('system_source_hashes.json',{'before':before,'after':hashes()})
    print('SYSTEM DONE',flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker',type=int);a=ap.parse_args()
    if a.worker:worker(a.worker)
    else:main()
