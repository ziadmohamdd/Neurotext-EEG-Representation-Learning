import importlib.util,json,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[2]).resolve()
s=importlib.util.spec_from_file_location('p',root/'scripts/analyze_glim_pairs.py');p=importlib.util.module_from_spec(s);s.loader.exec_module(p)
from sentence_transformers import SentenceTransformer
plan=json.loads(p.PLAN.read_text(encoding='utf-8'))['plan']
recs=[r for x in plan['cohort']['manifests'] for r in p.read_manifest(root/x)]
zone=p.excluded_positions(recs,plan['seed'],**plan['protected']['v2_test_zone'])
_,_,closed,v1=p.bridge_scope(recs,plan['seed'],zone)
prot={p.squash(r.text) for r in recs if (r.dataset,int(r.sentence_id)) in closed|v1}
out={n:p.read_rows(Path(sys.argv[1])/f) for n,f in p.FILES.items()}
base=out['glim'];keep=[i for i,r in enumerate(base) if p.squash(r['Input Text']) not in prot]
m=SentenceTransformer(p.MODEL,revision=p.REVISION,device='cpu')
enc=lambda t:p.normalize(m.encode(t,batch_size=64,show_progress_bar=False,convert_to_numpy=True).astype(np.float64))
sent=sorted({base[i]['Input Text'] for i in keep});pl={t:i for i,t in enumerate(sent)};bank=enc(sent)
own=np.array([pl[base[i]['Input Text']] for i in keep])
g={pl[base[i]['Input Text']]:base[i]['Dataset']+'_'+base[i]['Raw Task Key'] for i in keep};gs=np.array([g[i] for i in range(len(sent))])
res={}
for n,t in out.items():
    v=enc([t[i]['Generated Text'] for i in keep]);mat=p.sentence_matrix(p.pair_wins(v,own,bank),own,len(sent))
    names=sorted(set(gs));res[n]={}
    for a in names:
        for b in names:
            if a<b:
                x=np.ix_(gs==a,gs==b);y=np.ix_(gs==b,gs==a)
                res[n][a+'|'+b]=round(float((mat[x].sum()+mat[y].sum())/(mat[x].size+mat[y].size)),4)
    print(n,res[n])
json.dump(res,open(root/'results/glim_pair_check_20261106/group_pairs.json','w'),indent=1)
