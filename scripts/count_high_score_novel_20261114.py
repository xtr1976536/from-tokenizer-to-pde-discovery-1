import json,gzip,collections,csv,time
from pathlib import Path
ROOT=Path('/data1/users/liugang/src/eq-bpe/project');S=ROOT/'restarts/stage01_20261113/stage1';O=S.parent/'closing/task3/questions';O.mkdir(exist_ok=True);jobs=json.loads((S/'downstream_jobs.json').read_text());keys=sorted({(j['arm'],j['sampling'],j['case']) for j in jobs});out=[]
def dest(scale,arm,mode,seed,case):return S/'search/results'/str(scale)/arm/mode/str(seed)/case
for idx,(arm,mode,case) in enumerate(keys):
 small=set()
 for seed in [525,526,527]:
  with gzip.open(dest(212,arm,mode,seed,case)/'candidates.jsonl.gz','rt') as f:
   for line in f:
    r=json.loads(line)
    if r.get('effective_structure_id'):small.add(r['effective_structure_id'])
 for seed in [525,526,527]:
  p=dest(636,arm,mode,seed,case);final=set(json.loads((p/'history.json').read_text())['history'][-1]['candidate_ids']);scored=[]
  with gzip.open(p/'candidates.jsonl.gz','rt') as f:
   for line in f:
    r=json.loads(line)
    if r['status']=='evaluated':scored.append((r['effective_structure_id'],r['reward'],r.get('effective_classification')=='correct',r['candidate_id'] in final))
  truth=[x[1] for x in scored if x[2]];benchmark=max(truth) if truth else None;nov=[x for x in scored if x[0] not in small and not x[2]]
  row=dict(arm=arm,sampling=mode,case=case,seed=seed,scored=len(scored),correct_reference_available=benchmark is not None,best_correct_reward=benchmark,new_wrong_occurrences=len(nov),new_wrong_unique=len({x[0] for x in nov}),new_wrong_final_slots=sum(x[3] for x in nov),new_wrong_final_unique=len({x[0] for x in nov if x[3]}))
  for name,pred in [('ge085',lambda x:x[1]>=.85),('ge090',lambda x:x[1]>=.9),('above_correct',lambda x:benchmark is not None and x[1]>benchmark+1e-9)]:
   sel=[x for x in nov if pred(x)];row[name+'_occurrences']=len(sel);row[name+'_unique']=len({x[0] for x in sel});row[name+'_final_slots']=sum(x[3] for x in sel);row[name+'_final_unique']=len({x[0] for x in sel if x[3]});row[name+'_any']=bool(sel)
  out.append(row)
 if idx%8==7:print('GROUP',idx+1,flush=True)
with (O/'high_score_new_non_target.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
print('HIGH_SCORE_COMPLETE',len(out),flush=True)
