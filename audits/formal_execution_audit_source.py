import json,gzip,hashlib,math,collections,time,functools
from pathlib import Path
ROOT=Path('/data1/users/liugang/src/eq-bpe/project');B=ROOT/'restarts/stage01_20261113';S=B/'stage1';O=B/'closing/task1';O.mkdir(parents=True,exist_ok=True)
errors=[];hashes={};n=0;attempts=0;warnings=collections.Counter()
def check(ok,msg):
 if not ok:errors.append(msg)
@functools.lru_cache(None)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
jobs=json.loads((S/'downstream_jobs.json').read_text());check(len(jobs)==840,'job count');check(len({json.dumps(j,sort_keys=True) for j in jobs})==840,'duplicate jobs')
for j in jobs:
 p=S/'search/results'/str(j['scale'])/j['arm']/j['sampling']/str(j['seed'])/j['case'];tag=str(p.relative_to(S))
 try:
  c=json.loads((p/'complete.json').read_text());check(c['status']=='PASS',tag+' status')
  for name,h in c['files'].items():
   actual=digest(str(p/name));check(actual==h,tag+' hash '+name);hashes[str((p/name).relative_to(ROOT))]=actual
  r=json.loads((p/'RUN.json').read_text());check(all(r[k]==j[k] for k in ['scale','arm','seed','case']),tag+' identity');check(r['random_fraction']==j['random_fraction'],tag+' sampling')
  pre=S/str(j['scale']);check(r['pretrained_sha256']==digest(str(pre/j['arm']/str(j['seed'])/'best.pt')),tag+' pretrained');check(r['split_sha256']==digest(str(pre/'split.json')),tag+' split')
  check(r['surrogate_sha256']==digest(str(ROOT/'data/mlp_cache'/j['case']/'surrogate_cache.pt')),tag+' surrogate')
  h=json.loads((p/'history.json').read_text())['history'];check(len(h)==5,tag+' cycles')
  for x in h:
   lr=x['continuation_learning_rates'];check(len(lr)==5 and all(a>b for a,b in zip(lr,lr[1:])) and math.isclose(lr[0],1e-5) and math.isclose(x['continuation_final_lr'],1e-7),tag+' SFT schedule')
  count=collections.Counter();total=0
  with gzip.open(p/'candidates.jsonl.gz','rt') as f:
   for line in f:
    row=json.loads(line);total+=1
    if row['status']=='evaluated':count[row['cycle']]+=1
  check(dict(count)=={i:400 for i in range(1,6)},tag+' candidate budget');attempts+=total
  m=json.loads((p/'mechanism.json').read_text());check(m['attempted']==total and m['evaluated']==2000,tag+' mechanism counts')
  rr=json.loads((p/'result.json').read_text());check(rr['top1']==m['top1_correct'] and rr['top10']==m['top10_correct'],tag+' outcome')
  n+=1
 except Exception as e:errors.append(tag+' '+repr(e))
 if n%100==0:print('AUDITED',n,flush=True)
pretrains=0;epoch_range=[];splits=[]
from math_tokenizer.eqgpt.latex_codec import LatexParser,key,structure_to_latex
for scale in [212,318,424,530,636]:
 rows=[json.loads(x) for x in (S/f'corpora/{scale}/sources.jsonl').read_text().splitlines()];sp=json.loads((S/str(scale)/'split.json').read_text());tr=set(sp['train']);va=set(sp['validation']);check(not tr&va and tr|va=={r['source_id'] for r in rows},f'{scale} split IDs')
 groups=lambda ids:{repr(key(LatexParser(structure_to_latex(r['gre_ids'])).parse())) for r in rows if r['source_id'] in ids}
 check(not groups(tr)&groups(va),f'{scale} canonical AST overlap');splits.append(dict(scale=scale,train=len(tr),validation=len(va)))
 for arm in ['eqgpt_reference','ast_bpe_200','ast_bpe_300','ast_bpe_450']:
  for seed in [525,526,527]:
   p=S/str(scale)/arm/str(seed);t=json.loads((p/'training.json').read_text());h=t['history'];c=json.loads((p/'COMPLETE.json').read_text());early=float('inf');bad=0
   for x in h:
    loss=x['validation']['loss']
    if loss<early-1e-5:early=loss;bad=0
    else:bad+=1
   check(bad==15 and c['stopped_by']=='early_stopping',str(p)+' early stopping');check(c['best_epoch']==min(h,key=lambda x:x['validation']['loss'])['epoch'],str(p)+' best epoch');check(all(b['learning_rate']<=a['learning_rate'] for a,b in zip(h,h[1:])),str(p)+' cosine');pretrains+=1;epoch_range.append(len(h))
q=json.loads((S/'SEARCH_QUEUE.json').read_text());check(q['finished']==840 and not q['failed'] and not q['active'] and q['pending']==0,'final queue')
for p in (S/'search_logs').glob('*.log'):
 txt=p.read_text(errors='replace')
 for token in ['out of memory','Traceback','RuntimeWarning']:
  if token.lower() in txt.lower():warnings[token]+=1
report=dict(status='PASS' if not errors else 'FINDINGS',audited_cases=n,pretrains=pretrains,attempts=attempts,successful_scored_candidates=n*2000,epoch_range=[min(epoch_range),max(epoch_range)],splits=splits,errors=errors,log_files_with_markers=dict(warnings),recovered_failures=json.loads((S/'RECOVERED_FAILURES.json').read_text()),scope='file integrity, identities, budgets, schedules, within-scale ID/AST isolation; no claim of all mathematical equivalence or manual source review',time=time.time())
(O/'audit.json').write_text(json.dumps(report,indent=2));(O/'frozen_result_hashes.json').write_text(json.dumps(hashes,indent=2));print(json.dumps(report),flush=True)
