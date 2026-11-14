"""Eight pretraining workers; adopt live workers without restarting training."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'restarts/stage01_20261113/stage1';lock=(out/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
gpu='GPU-e857f4d0-daa6-f33d-3dc8-927f15ab7d59';arms=['eqgpt_reference','ast_bpe_200','ast_bpe_300','ast_bpe_450'];jobs=[(s,a,seed) for seed in [525,526,527] for s in [212,318,424,530,636] for a in arms]
previous=json.loads((out/'QUEUE.json').read_text());active={};failed=[];finished=[];pending=[]
class Adopted:
 def __init__(self,pid,job):self.pid=pid;self.job=job
 def poll(self):
  p=Path('/proc')/str(self.pid)
  try:
   if p.exists() and p.joinpath('stat').read_text().split()[2]!='Z':
    cmd=p.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode();s,a,seed=self.job
    if 'scripts/pretrain_stage1_20261113.py' in cmd and f'--scale {s} ' in cmd and f'--arm {a} ' in cmd and f'--seed {seed}' in cmd:return None
  except OSError:pass
  return 0 if (out/str(self.job[0])/self.job[1]/str(self.job[2])/'COMPLETE.json').exists() else 1
for r in previous['active']:
 j=tuple(r['job']);active[j]=(Adopted(r['pid'],j),None,r.get('cap_GiB',20))
for j in jobs:
 if j in active:continue
 if (out/str(j[0])/j[1]/str(j[2])/'COMPLETE.json').exists():finished.append(j)
 else:pending.append(j)
env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONPATH='.:src',STAGE1_WORKER_CAP_GIB='8')
def save():
 d=dict(total=60,gpu=gpu,max_workers=8,new_worker_hard_cap_GiB=8,max_reserved_GiB=84,max_combined_GiB=92,active=[dict(job=j,pid=p.pid,cap_GiB=cap) for j,(p,f,cap) in active.items()],pending=pending,finished=finished,failed=failed,updated=time.time())
 p=out/'QUEUE.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)
while pending or active:
 for j,(p,f,cap) in list(active.items()):
  rc=p.poll()
  if rc is not None:
   if f:f.close()
   del active[j]
   if rc==0 and (out/str(j[0])/j[1]/str(j[2])/'COMPLETE.json').exists():finished.append(j)
   else:failed.append(dict(job=j,exit=rc))
 used=int(subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
 reserved=sum(cap for p,f,cap in active.values())
 if pending and not failed and len(active)<8 and used+8192<87000 and reserved+8<=84:
  j=pending.pop(0);s,a,seed=j;log=(out/f'pretrain_{s}_{a}_{seed}.log').open('a')
  cmd=[sys.executable,'-u','scripts/pretrain_stage1_20261113.py','--scale',str(s),'--arm',a,'--seed',str(seed)]
  active[j]=(subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=log),log,8)
 save()
 if failed and not active:break
 time.sleep(5)
save()
