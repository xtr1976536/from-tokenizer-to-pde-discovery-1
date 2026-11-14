import json,gzip,math
from pathlib import Path
import numpy as np,torch
from math_tokenizer.eqgpt.existing_surrogate import load_existing
from math_tokenizer.eqgpt.author import numerical_namespace
ROOT=Path('/data1/users/liugang/src/eq-bpe/project');B=ROOT/'restarts/stage01_20261113';S=B/'stage1';O=B/'closing/task3/questions';O.mkdir(exist_ok=True)
p=S/'search/results/636/ast_bpe_300/model/525/Chaffee_Infante_equation';correct=None;wrong=None
with gzip.open(p/'candidates.jsonl.gz','rt') as f:
 for line in f:
  r=json.loads(line)
  if r['status']!='evaluated':continue
  if r.get('effective_classification')=='correct' and (correct is None or r['reward']>correct['reward']):correct=r
  if r['candidate_id']=='2:449':wrong=r
torch.set_num_threads(4);_,net,db,_,_=load_existing(ROOT/'data/mlp_cache/Chaffee_Infante_equation','Chaffee_Infante_equation','cpu',525,10000);outputs=[]
for label,r in [('non_target',wrong),('correct',correct)]:
 env=numerical_namespace('cpu','Chaffee_Infante_equation');cap={};original=np.linalg.lstsq
 def wrapped(X,y,*args,**kwargs):
  out=original(X,y,*args,**kwargs);cap.update(X=X.copy(),y=y.copy(),coef=out[0].copy());return out
 np.linalg.lstsq=wrapped
 try:reward,_,_=env['calculate_reward'](r['gre'][1:].copy(),net,db,{},env['get_mask_invalid'](['x','t']),['x','t'],.2)
 finally:np.linalg.lstsq=original
 X=cap['X'];y=cap['y'];coef=cap['coef'];fit=X.dot(coef);sse=float(((y-fit)**2).sum());sst=float(((y-y.mean())**2).sum());r2=1-sse/sst;M=X.shape[1]+1;penalty=1-.2*np.log10(M)
 outputs.append(dict(label=label,candidate_id=r['candidate_id'],gre=r['gre'],saved_reward=r['reward'],cpu_replayed_reward=float(reward),cpu_R2=r2,saved_implied_R2=r['reward']/penalty,columns=M,penalty=penalty,N=len(y),SSE=sse,SST=sst,cpu_coefficients=[1.]+coef.tolist(),cpu_reward_error=float(reward)-r['reward']))
(O/'reward_two_equations.json').write_text(json.dumps(outputs,indent=2));print(json.dumps(outputs),flush=True)
