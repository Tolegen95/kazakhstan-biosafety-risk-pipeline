"""Reproducible reviewer experiments; original inputs and results are read only.
Run with the original scientific environment (Python 3.13), not document Python.
"""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, os, platform, resource, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'revision';OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT))
from pipeline.core import RunContext
from pipeline.covariates import load_covariate_stack_from_file
from pipeline.modules.maxent import _feature_matrix, _block_ids
from pipeline.modules.space_time_scan import SpaceTimeScanModule
from benchmarks.maxent_reference import fit_maxent
threadpool_limits(limits=1)
KEY=['region_oblast','district_raion','settlement_name','event_date','year','species']
LEGACY=[c for c in KEY if c!='year']

def write(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf8')
def load(d):
    ev=pd.read_csv(ROOT/f'data/normalized/{d}_normalized.csv')
    for c in ['latitude','longitude','year','animal_count']:
        ev[c]=pd.to_numeric(ev[c],errors='coerce')
    return ev
def covariates():
    raw=load_covariate_stack_from_file(str(ROOT/'data/covariates/kazakhstan_bioclim.tif'))
    return raw,raw.masked_to_boundary(str(ROOT/'data/boundaries/kazakhstan.geojson'))
def fit(p,b,C=1):
    m,s=p.mean(0),p.std(0); s[s==0]=1
    clf=LogisticRegression(C=C,max_iter=2000)
    clf.fit(_feature_matrix(np.r_[p,b],m,s),np.r_[np.ones(len(p)),np.zeros(len(b))])
    return lambda x:clf.predict_proba(_feature_matrix(x,m,s))[:,1]
def score(f,p,b):
    return float(roc_auc_score(np.r_[np.ones(len(p)),np.zeros(len(b))],np.r_[f(p),f(b)]))
def sample(ev,cov):
    ev=ev.dropna(subset=['latitude','longitude']).copy()
    x=cov.sample(ev.longitude.to_numpy(),ev.latitude.to_numpy())
    ok=np.isfinite(x).all(1)
    return ev.loc[ok].reset_index(drop=True),x[ok]
def folds_for(ev,blon,blat,repeats=5):
    pb=_block_ids(ev.longitude.to_numpy(),ev.latitude.to_numpy(),2)
    bb=_block_ids(blon,blat,2)
    # Fixed master grid, so filtering records never changes block assignment.
    master=np.array([i*100000+j for i in range(22,46) for j in range(18,30)])
    rng=np.random.default_rng(20261028)
    folds=[]
    for rep in range(repeats):
        lookup=dict(zip(master,rng.permutation(len(master))%5))
        pf=np.array([lookup[int(x)] for x in pb]);bf=np.array([lookup[int(x)] for x in bb])
        folds.extend((rep,k,pf!=k,pf==k,bf!=k,bf==k) for k in range(5))
    return folds
def evaluate(ev,px,bx,blon,blat,C=1,cols=None,repeats=5):
    cols=np.arange(px.shape[1]) if cols is None else np.array(cols)
    p,b=px[:,cols],bx[:,cols]
    rows=[]
    for rep,k,ptr,pte,btr,bte in folds_for(ev,blon,blat,repeats):
        if pte.sum()==0 or bte.sum()==0 or ptr.sum()<5:continue
        f=fit(p[ptr],b[btr],C)
        rows.append({'repeat':rep,'fold':k,'auc':score(f,p[pte],b[bte]),'n_test':int(pte.sum())})
    tab=pd.DataFrame(rows); means=tab.groupby('repeat').auc.mean()
    return {'n_presence':len(p),'mean':float(means.mean()),'sd_repeat':float(means.std(ddof=0)),
            'sd_fold':float(tab.auc.std(ddof=0))},tab

def models():
    raw,masked=covariates(); anth=load('anthrax');rab=load('rabies')
    rng=np.random.default_rng(42)
    blon,blat=masked.random_background(5000,rng);bx=masked.sample(blon,blat)
    ev,px=sample(anth,masked)
    baseline,bt=evaluate(ev,px,bx,blon,blat)
    bt.to_csv(OUT/'baseline_block_folds.csv',index=False)
    audit={}
    for disease in ['anthrax','rabies','fmd']:
        d=load(disease);coords=d.dropna(subset=['latitude','longitude'])
        used,x=sample(d,masked)
        scan=d.dropna(subset=['latitude','longitude','year','district_raion'])
        audit[disease]={'normalized':len(d),'valid_coordinates':len(coords),'valid_coordinates_and_year':len(coords.dropna(subset=['year'])),
            'scan_used':len(scan),'masked_presence':len(used),'raster_or_mask_excluded':len(coords)-len(used),
            'year_min':int(d.year.min()),'year_max':int(d.year.max()),'legacy_duplicate_candidates':int(d.duplicated(LEGACY).sum()),
            'year_aware_duplicate_candidates':int(d.duplicated(KEY).sum()),
            'conservative_duplicate_candidates':int(d.duplicated(KEY+['latitude','longitude','animal_count']).sum()),
            'year_only_records':int(d.event_date.isna().sum())}
    write('audit.json',audit)
    quality=[]
    for tag,data in [('baseline',anth),('year_aware_candidates_removed',anth.drop_duplicates(KEY)),
                     ('legacy_key_removed',anth.drop_duplicates(LEGACY))]:
        e,x=sample(data,masked);s,_=evaluate(e,x,bx,blon,blat);quality.append({'scenario':tag,**s})
    # Factorial isolates point retention from changing the background extent.
    eall,xall=sample(anth,raw)
    rblo,rbla=raw.random_background(5000,np.random.default_rng(42));rbx=raw.sample(rblo,rbla)
    for tag,e,x,b,lo,la in [('retained_presence_bbox_background',ev,px,rbx,rblo,rbla),
                           ('restored_presence_national_background',eall,xall,bx,blon,blat),
                           ('restored_presence_bbox_background',eall,xall,rbx,rblo,rbla)]:
        s,_=evaluate(e,x,b,lo,la);quality.append({'scenario':tag,**s})
    pd.DataFrame(quality).to_csv(OUT/'quality_auc.csv',index=False)
    from shapely.geometry import shape,Point
    gj=json.loads((ROOT/'data/boundaries/kazakhstan.geojson').read_text())
    from shapely.ops import unary_union
    poly=unary_union([shape(f['geometry']) for f in gj['features']])
    xv=masked.sample(eall.longitude.to_numpy(),eall.latitude.to_numpy())
    excluded=eall[~np.isfinite(xv).all(1)]
    audit['boundary_exclusions']={'n':len(excluded),'inside_vector_polygon':sum(poly.covers(Point(r.longitude,r.latitude)) for r in excluded.itertuples()),'outside_vector_polygon':sum(not poly.covers(Point(r.longitude,r.latitude)) for r in excluded.itertuples())}
    write('audit.json',audit)
    temp=[i for i,n in enumerate(masked.names) if n.startswith('bio') and int(n[3:])<=11]
    precip=[i for i,n in enumerate(masked.names) if n.startswith('bio') and int(n[3:])>=12]
    elev=[i for i,n in enumerate(masked.names) if n=='elevation']
    assert len(temp)==11 and len(precip)==8 and len(elev)==1
    groups={'all':list(range(20)),'temperature_only':temp,
            'precipitation_only':precip,'bioclim_without_elevation':temp+precip,
            'without_temperature':precip+elev,'without_precipitation':temp+elev,
            'elevation_only':elev}
    ab=[]
    for name,cols in groups.items():
        s,_=evaluate(ev,px,bx,blon,blat,cols=cols);ab.append({'group':name,**s})
    pd.DataFrame(ab).to_csv(OUT/'predictor_ablation.csv',index=False)
    grid=masked.data.reshape(20,-1).T;grid=grid[np.isfinite(grid).all(1)]
    corr=spearmanr(grid).statistic
    pd.DataFrame(corr,index=masked.names,columns=masked.names).to_csv(OUT/'predictor_spearman.csv')
    pairs=[{'a':masked.names[i],'b':masked.names[j],'rho':float(corr[i,j])} for i in range(20) for j in range(i+1,20) if abs(corr[i,j])>=.8]
    write('predictor_correlations.json',{'n_valid_cells':len(grid),'n_pairs':190,'n_abs_rho_ge_08':len(pairs),'pairs':pairs})
    jk=[]
    for i,name in enumerate(masked.names):
        s,t=evaluate(ev,px,bx,blon,blat,cols=[j for j in range(20) if i!=j])
        paired=bt[['repeat','fold','auc']].merge(t[['repeat','fold','auc']],on=['repeat','fold'],suffixes=('_full','_without'))
        paired['delta']=paired.auc_full-paired.auc_without
        ds=paired.groupby('repeat').delta.mean()
        jk.append({'predictor':name,'auc_without':s['mean'],'delta_auc':float(ds.mean()),'sd_delta_across_repeats':float(ds.std(ddof=0))})
    pd.DataFrame(jk).to_csv(OUT/'jackknife_paired.csv',index=False)
    reg=[]
    for tag,d in [('anthrax',anth),('rabies_livestock',rab[rab.reservoir_category=='livestock']),('rabies_companion',rab[rab.reservoir_category=='companion'])]:
        e,x=sample(d,masked)
        for C in [.01,.1,1,10,100]:
            s,_=evaluate(e,x,bx,blon,blat,C=C,repeats=3);reg.append({'dataset':tag,'C':C,**s})
    pd.DataFrame(reg).to_csv(OUT/'regularization_logistic.csv',index=False)
    # Retrospective temporal transfer; fixed modern climate, NOT a historical real-time forecast.
    temporal=[]
    for tag,d,cuts in [('anthrax',anth,[2000,2005,2010]),('rabies',rab,[2015,2016,2017])]:
        e,x=sample(d.dropna(subset=['year']),masked)
        for cut in cuts:
            tr=e.year.to_numpy()<=cut;te=~tr
            if te.sum()<2:continue
            trainbg=bx[:2500];testbg=bx[2500:]
            f=fit(x[tr],trainbg)
            area_scores=f(testbg);test_scores=f(x[te]);thr=np.quantile(f(bx),.8)
            blockp=_block_ids(e.loc[te,'longitude'].to_numpy(),e.loc[te,'latitude'].to_numpy(),2)
            blockb=_block_ids(blon[2500:],blat[2500:],2)
            blocks=np.unique(np.r_[blockp,blockb]);rngb=np.random.default_rng(123)
            aucs=[]
            for _ in range(500):
                draw=rngb.choice(blocks,len(blocks),replace=True)
                pp=np.concatenate([test_scores[blockp==j] for j in draw]);bs=np.concatenate([area_scores[blockb==j] for j in draw])
                if len(pp) and len(bs):aucs.append(roc_auc_score(np.r_[np.ones(len(pp)),np.zeros(len(bs))],np.r_[pp,bs]))
            temporal.append({'dataset':tag,'train_through':cut,'test_from':cut+1,'test_through':int(e.year.max()),
                'n_train':int(tr.sum()),'n_test':int(te.sum()),'auc':score(f,x[te],testbg),'block_bootstrap_lo':float(np.quantile(aucs,.025)),
                'block_bootstrap_hi':float(np.quantile(aucs,.975)),'fraction_test_in_top20_background':float((test_scores>=thr).mean()),'n_valid_bootstraps':len(aucs)})
    pd.DataFrame(temporal).to_csv(OUT/'temporal_transfer.csv',index=False)
    priors=[('uniform',1,1),('Jeffreys',.5,.5),('fitted_reproduction_only',24.97,184.29)]
    for mean in [.01,.05,.10,.28]:
        for strength in [2,20,100,209.26]:priors.append((f'mean_{mean}_strength_{strength}',mean*strength,(1-mean)*strength))
    bay=[]
    for name,a,b in priors:
        post=beta(a+35,b+894-35)
        bay.append({'prior':name,'prior_mean':a/(a+b),'prior_strength':a+b,'alpha':a,'beta':b,'posterior_mean':post.mean(),'low':post.ppf(.025),'high':post.ppf(.975)})
    pd.DataFrame(bay).to_csv(OUT/'prior_sensitivity.csv',index=False)
    write('models_summary.json',{'baseline':baseline,'covariate_names':masked.names,'finished':True})
    print('MODELS DONE',flush=True)

def cells(row):
    return {(d,y) for d in row.districts.split('; ') for y in range(int(row.start_year),int(row.end_year)+1)}
def jac(a,b):return len(a&b)/len(a|b) if a|b else 1.
def scanrun(ev,disease,tag,sp=.5,tp=.5,R=999):
    t=time.perf_counter()
    r=SpaceTimeScanModule().run(RunContext(disease,ev,{'max_spatial_fraction':sp,'max_temporal_fraction':tp,'n_permutations':R,'random_seed':42,'max_clusters_reported':200}))
    tab=r.tables['clusters'];tab.to_csv(OUT/f'scan_{tag}.csv',index=False)
    write(f'scan_{tag}.json',{**r.summary,'seconds':time.perf_counter()-t})
    print(tag,round(time.perf_counter()-t,2),r.summary['n_significant_clusters_p<0.05'],flush=True)
    return tab,r.summary
def scans():
    ev=load('anthrax'); rows=[]
    baseline=pd.read_csv(ROOT/'data/normalized/results/anthrax/clusters.csv')
    base_primary=cells(next(baseline.itertuples()))
    base_union=set().union(*(cells(r) for r in baseline[baseline.p_value<.05].itertuples()))
    for tag,d in [('year_aware_dedup',ev.drop_duplicates(KEY)),('legacy_dedup',ev.drop_duplicates(LEGACY))]:
        if (OUT/f'scan_{tag}.json').exists():
            tab=pd.read_csv(OUT/f'scan_{tag}.csv');s=json.loads((OUT/f'scan_{tag}.json').read_text())
        else:tab,s=scanrun(d,'anthrax',tag)
        sig=tab[tab.p_value<.05]
        rows.append({'scenario':tag,'n_events':s['n_events_used'],'significant':len(sig),'primary_llr':s['observed_max_llr'],'primary_p':s['observed_max_llr_p_value'],
             'primary_jaccard':jac(base_primary,cells(next(tab.itertuples()))),'union_jaccard':jac(base_union,set().union(*(cells(r) for r in sig.itertuples())))})
    pd.DataFrame(rows).to_csv(OUT/'quality_clusters.csv',index=False)
    grid=[]
    for sp in [.05,.1,.2,.3,.5]:
        for tp in [.05,.1,.5]:
            tag=f'grid_{sp:g}_{tp:g}';tab,s=scanrun(ev,'anthrax',tag,sp,tp,999)
            sig=tab[tab.p_value<.05];first=next(tab.itertuples())
            grid.append({'spatial_pct':sp*100,'temporal_pct':tp*100,'R':999,'n_significant':len(sig),'primary_llr':s['observed_max_llr'],
                 'primary_p':s['observed_max_llr_p_value'],'primary_start':first.start_year,'primary_end':first.end_year,
                 'primary_jaccard_to_50_50':jac(base_primary,cells(first)),
                 'union_jaccard_to_50_50':jac(base_union,set().union(*(cells(r) for r in sig.itertuples())))})
            pd.DataFrame(grid).to_csv(OUT/'scan_grid.csv',index=False)
    # Error recorded in source: isolate most geographically remote rabies point by district.
    rab=load('rabies')
    from pipeline.modules.space_time_scan import _haversine_km
    distances=[]
    for _,g in rab.groupby(['region_oblast','district_raion']):
        if len(g)<3:continue
        la,lo=g.latitude.median(),g.longitude.median()
        for ix,r in g.iterrows():
            km=float(_haversine_km(np.array([r.latitude]),np.array([r.longitude]),np.array([la]),np.array([lo]))[0])
            distances.append((km,ix))
    km,idx=max(distances)
    tab,s=scanrun(rab.drop(index=idx),'rabies','rabies_outlier_removed')
    orig=pd.read_csv(ROOT/'data/normalized/results/rabies/clusters.csv')
    u=set().union(*(cells(r) for r in orig[orig.p_value<.05].itertuples()))
    v=set().union(*(cells(r) for r in tab[tab.p_value<.05].itertuples()))
    write('rabies_error_sensitivity.json',{'focus_id':rab.loc[idx,'focus_id'],'distance_to_district_median_km':km,'union_jaccard':jac(u,v),'summary':s})
    print('SCANS DONE',flush=True)

def maxent_regularization():
    import elapid
    raw,masked=covariates();rng=np.random.default_rng(42)
    blon,blat=masked.random_background(5000,rng);bx=masked.sample(blon,blat)
    e,px=sample(load('anthrax'),masked);rows=[]
    # Three repetitions of five identical spatial folds, feature classes fixed.
    for mult in [.5,1,2,4]:
        t=time.perf_counter();vals=[]
        for rep,k,tr,te,btr,bte in folds_for(e,blon,blat,repeats=3):
            model=elapid.MaxentModel(feature_types=['linear','quadratic','hinge','product'],beta_multiplier=mult,transform='cloglog',random_state=42)
            x=pd.DataFrame(np.r_[px[tr],bx[btr]]);y=np.r_[np.ones(tr.sum()),np.zeros(btr.sum())]
            model.fit(x,y)
            f=lambda a:np.asarray(model.predict(pd.DataFrame(a))).ravel()
            vals.append({'repeat':rep,'fold':k,'auc':score(f,px[te],bx[bte])})
        tab=pd.DataFrame(vals);tab.to_csv(OUT/f'elapid_beta_{mult:g}_folds.csv',index=False)
        means=tab.groupby('repeat').auc.mean()
        rows.append({'beta_multiplier':mult,'mean':means.mean(),'sd_repeat':means.std(ddof=0),'seconds':time.perf_counter()-t})
        pd.DataFrame(rows).to_csv(OUT/'regularization_elapid.csv',index=False)
        print('elapid',rows[-1],flush=True)
    print('ELAPID DONE',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('part',choices=['models','scans','elapid']);args=ap.parse_args()
    write(f'environment_{args.part}.json',{'python':sys.version,'platform':platform.platform(),
        'packages':{x:importlib.metadata.version(x) for x in ['numpy','pandas','scipy','scikit-learn','rasterio','elapid']},
        'seed':42,'analysis_seed':20261028,'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'data/normalized').glob('*_normalized.csv'))}})
    {'models':models,'scans':scans,'elapid':maxent_regularization}[args.part]()
