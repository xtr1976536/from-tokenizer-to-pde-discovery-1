import json,gzip,collections,csv
from pathlib import Path
import numpy as np
ROOT=Path('/data1/users/liugang/src/eq-bpe/project');S=ROOT/'restarts/stage01_20261113/stage1';O=S.parent/'paper_stage01_v2_20261114';O.mkdir(exist_ok=True);out=[];scores=collections.defaultdict(list)
jobs=json.loads((S/'downstream_jobs.json').read_text());keys=sorted({(j['arm'],j['sampling'],j['case']) for j in jobs})
for arm,mode,case in keys:
 def path(scale):return S/'search/results'/str(scale)/arm/mode/'525'/case
 small=set()
 with gzip.open(path(212)/'candidates.jsonl.gz','rt') as f:
  for l in f:
   r=json.loads(l)
   if r.get('effective_structure_id'):small.add(r['effective_structure_id'])
 p=path(636);final=set(json.loads((p/'history.json').read_text())['history'][-1]['candidate_ids']);rr=[]
 with gzip.open(p/'candidates.jsonl.gz','rt') as f:
  for l in f:
   r=json.loads(l)
   if r['status']=='evaluated':rr.append((r['effective_structure_id'],r['reward'],r.get('effective_classification')=='correct',r['candidate_id'] in final))
 new=[r for r in rr if r[0] not in small];truth=[r[1] for r in rr if r[2]];cut=max(truth) if truth else None;high=[r for r in new if not r[2] and cut is not None and r[1]>cut+1e-9]
 out.append(dict(arm=arm,sampling=mode,case=case,seed=525,scored=len(rr),new_scored=len(new),unique_scored=len({r[0] for r in rr}),new_unique=len({r[0] for r in new}),new_final=sum(r[3] for r in new),new_final_unique=len({r[0] for r in new if r[3]}),new_correct=sum(r[2] for r in new),new_final_correct=sum(r[2] and r[3] for r in new),reference_available=cut is not None,high_wrong_count=len(high),high_wrong_unique=len({r[0] for r in high}),high_wrong_final=sum(r[3] for r in high),high_wrong_any=bool(high)))
 v=np.array([r[1] for r in rr]);order=np.argsort(v);rank=np.empty(len(v));i=0
 while i<len(order):
  j=i+1
  while j<len(order) and v[order[j]]==v[order[i]]:j+=1
  rank[order[i:j]]=((i+j-1)/2+.5)/len(v);i=j
 if arm in ['ast_bpe_300','ast_bpe_450'] and mode=='mix20':
  for i,r in enumerate(rr):scores['new' if r[0] not in small else 'seen'].append(float(rank[i]))
with (O/'seed525_novelty.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
np.savez_compressed(O/'seed525_rank_scores.npz',**{k:np.array(v) for k,v in scores.items()});print('SEED525_ANALYSIS_DONE',len(out))
