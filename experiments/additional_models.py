from revision_experiments import *
from sklearn.model_selection import train_test_split
raw,cov=covariates();blon,blat=cov.random_background(5000,np.random.default_rng(42));bx=cov.sample(blon,blat)
rows=[]
for tag,data in [('anthrax',load('anthrax')),('rabies_livestock',load('rabies').query("reservoir_category=='livestock'")),('rabies_companion',load('rabies').query("reservoir_category=='companion'")),('rabies_wildlife',load('rabies').query("reservoir_category=='wildlife'"))]:
 e,x=sample(data,cov);rng=np.random.default_rng(20261028);aa=[]
 for _ in range(100):
  p,q=train_test_split(np.arange(len(x)),train_size=.75,random_state=int(rng.integers(1<<30)))
  a,b=train_test_split(np.arange(len(bx)),train_size=.75,random_state=int(rng.integers(1<<30)))
  aa.append(score(fit(x[p],bx[a]),x[q],bx[b]))
 s,t=evaluate(e,x,bx,blon,blat);t.to_csv(OUT/f'updated_{tag}_folds.csv',index=False)
 rows.append({'dataset':tag,'random_mean':np.mean(aa),'random_sd':np.std(aa),**s})
pd.DataFrame(rows).to_csv(OUT/'updated_main_auc.csv',index=False)
strict=load('anthrax').drop_duplicates(KEY+['latitude','longitude','animal_count'])
e,x=sample(strict,cov);s,t=evaluate(e,x,bx,blon,blat)
write('conservative_duplicates_auc.json',s)
tab,ss=scanrun(strict,'anthrax','conservative_dedup')
base=pd.read_csv(ROOT/'data/normalized/results/anthrax/clusters.csv')
u=set().union(*(cells(r) for r in base[base.p_value<.05].itertuples()));v=set().union(*(cells(r) for r in tab[tab.p_value<.05].itertuples()))
write('conservative_duplicates_clusters.json',{'summary':ss,'primary_jaccard':jac(cells(next(base.itertuples())),cells(next(tab.itertuples()))),'union_jaccard':jac(u,v)})
print('ADDITIONAL DONE',flush=True)
