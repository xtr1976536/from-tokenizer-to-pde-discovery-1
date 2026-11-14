"""Persist compact progress for pretraining and downstream; no notifications."""
import json,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'restarts/stage01_20261113';stage=out/'stage1'
while True:
 try:
  q=json.loads((stage/'QUEUE.json').read_text());search_path=stage/'SEARCH_QUEUE.json';search=json.loads(search_path.read_text()) if search_path.exists() else None
  source=search if search else q;active=[]
  for r in source['active']:
   p=Path('/proc')/str(r['pid']);live=p.exists() and p.joinpath('stat').read_text().split()[2]!='Z';active.append(dict(job=r['job'],pid=r['pid'],live=live))
  gpu=subprocess.check_output(['nvidia-smi','--id='+','.join(search.get('gpus',[q['gpu']]) if search else [q['gpu']]),'--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
  report=dict(updated=time.time(),phase='downstream' if search else 'pretraining',pretraining_finished=len(q['finished']),pretraining_total=60,pretraining_failed=q['failed'],downstream_finished=search['finished'] if search else 0,downstream_total=840,downstream_failed=search['failed'] if search else [],active=active,gpu_memory_MiB_and_utilization=gpu,queue_stale_seconds=time.time()-source['updated'])
  p=out/'STATUS.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(p)
  if q['failed'] or (search and (search['failed'] or search['finished']==840)):break
 except Exception as e:(out/'monitor_error.txt').write_text(type(e).__name__+': '+str(e))
 time.sleep(30)
