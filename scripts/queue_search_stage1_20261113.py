"""One locked queue, two GPUs, six distinct cases per GPU."""
import fcntl,hashlib,json,os,shlex,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];base=ROOT/'restarts/stage01_20261113/stage1';lock=(base/'search_queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
GPUS=['GPU-e857f4d0-daa6-f33d-3dc8-927f15ab7d59','GPU-ba72db90-47e9-38fd-0e9e-905c3b1798b5'];PER_GPU=6;CAP=14;LIMIT=92*1024
pre=json.loads((base/'QUEUE.json').read_text());assert len(pre['finished'])==60 and not pre['failed'];assert json.loads((base/'SEARCH_SMOKE_PASS.json').read_text())['status']=='PASS'
jobs=json.loads((base/'downstream_jobs.json').read_text())
def key(j):return '_'.join(str(j[k]) for k in ['scale','arm','sampling','seed','case'])
def destination(j):return base/'search/results'/str(j['scale'])/j['arm']/j['sampling']/str(j['seed'])/j['case']
def complete(j):
 p=destination(j)/'complete.json'
 if not p.exists():return False
 m=json.loads(p.read_text());assert m['status']=='PASS'
 for name,digest in m['files'].items():assert hashlib.sha256((p.parent/name).read_bytes()).hexdigest()==digest
 return True
assert len(jobs)==840 and len({key(j) for j in jobs})==840
previous=json.loads((base/'SEARCH_QUEUE.json').read_text());assert not previous['failed'];active={};finished=[];pending=[];failed=[]
class Adopted:
 def __init__(self,pid,j):self.pid=pid;self.job=j
 def poll(self):
  p=Path('/proc')/str(self.pid)
  try:
   if p.exists() and p.joinpath('stat').read_text().split()[2]!='Z':
    argv=p.joinpath('cmdline').read_bytes().decode().split(chr(0));j=self.job
    if any('scripts/search_stage1_20261113.py' in x for x in argv) and all(argv[argv.index(flag)+1]==str(j[field]) for flag,field in [('--scale','scale'),('--arm','arm'),('--seed','seed'),('--case','case')]):return None
  except (OSError,ValueError,IndexError):pass
  return 0 if complete(self.job) else 1
for r in previous['active']:
 j=r['job'];k=key(j);assert k not in active;gpu=r.get('gpu',previous.get('gpu',GPUS[0]));assert gpu in GPUS
 active[k]=dict(job=j,process=Adopted(r['pid'],j),log=None,gpu=gpu,cap=r.get('cap_GiB',CAP))
for j in jobs:
 if key(j) in active:continue
 if complete(j):finished.append(j)
 else:pending.append(j)
logs=base/'search_logs';logs.mkdir(exist_ok=True)
def save():
 assert len(active)==len({key(x['job']) for x in active.values()})
 assert not ({key(j) for j in finished}&set(active))
 assert len(finished)+len(pending)+len(active)+len(failed)==840
 d=dict(total=840,finished=len(finished),pending=len(pending),active=[dict(job=x['job'],pid=x['process'].pid,gpu=x['gpu'],cap_GiB=x['cap']) for x in active.values()],failed=failed,updated=time.time(),gpu=GPUS[0],gpus=GPUS,max_workers=12,max_workers_per_gpu=6,new_worker_cap_GiB=CAP,max_reserved_GiB_per_gpu=84,memory_limit_GiB_per_gpu=92,sft='cosine 1e-5 to 1e-7;5 epochs per cycle')
 p=base/'SEARCH_QUEUE.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)
while pending or active:
 for k,x in list(active.items()):
  rc=x['process'].poll()
  if rc is not None:
   if x['log']:x['log'].close()
   del active[k]
   if rc==0 and complete(x['job']):finished.append(x['job'])
   else:failed.append(dict(job=x['job'],exit_code=rc,gpu=x['gpu']))
 for gpu in GPUS:
  workers=[x for x in active.values() if x['gpu']==gpu]
  used=int(subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
  if pending and not failed and len(workers)<PER_GPU and used+15*1024<=LIMIT and sum(x['cap'] for x in workers)+CAP<=84:
   j=pending.pop(0);k=key(j)
   if complete(j):finished.append(j);continue
   assert k not in active
   f=(logs/(k+'.log')).open('a');env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONPATH='.:src',STAGE1_SEARCH_CAP_GIB=str(CAP))
   cmd=[sys.executable,'-u','scripts/search_stage1_20261113.py','--scale',str(j['scale']),'--arm',j['arm'],'--seed',str(j['seed']),'--random-fraction',str(j['random_fraction']),'--case',j['case']]
   active[k]=dict(job=j,process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=f),log=f,gpu=gpu,cap=CAP)
 save()
 if failed and not active:break
 time.sleep(5)
save()
if not failed:subprocess.run([sys.executable,'scripts/summarize_stage1_20261113.py'],cwd=ROOT,check=True)
