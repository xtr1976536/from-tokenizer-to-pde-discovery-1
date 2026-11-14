"""Summarize only fully complete, hash-verified 840-case matrix."""
import csv,hashlib,json,statistics
from collections import defaultdict
from pathlib import Path
root=Path(__file__).resolve().parents[1];base=root/'restarts/stage01_20261113/stage1';jobs=json.loads((base/'downstream_jobs.json').read_text());groups=defaultdict(list)
for j in jobs:
 p=base/'search/results'/str(j['scale'])/j['arm']/j['sampling']/str(j['seed'])/j['case'];complete=json.loads((p/'complete.json').read_text())
 assert complete['status']=='PASS'
 for name,digest in complete['files'].items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==digest
 result=json.loads((p/'result.json').read_text());mechanism=json.loads((p/'mechanism.json').read_text());groups[(j['scale'],j['arm'],j['sampling'],j['case'])].append((j['seed'],result,mechanism))
rows=[]
for key,items in sorted(groups.items()):
 assert sorted(x[0] for x in items)==[525,526,527]
 r=dict(zip(['scale','arm','sampling','case'],key));r['seeds']=3
 for metric in ['top1','top10']:
  values=[int(x[1][metric]) for x in items];r[metric+'_successes']=sum(values);r[metric+'_mean']=statistics.mean(values);r[metric+'_std']=statistics.stdev(values)
 times=[x[2]['search_seconds'] for x in items];r['seconds_mean']=statistics.mean(times);r['seconds_std']=statistics.stdev(times);rows.append(r)
assert len(rows)==280
out=base/'summary';out.mkdir(exist_ok=True)
with (out/'per_case.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(out/'COMPLETE.json').write_text(json.dumps({'status':'PASS','case_runs':len(jobs),'groups':len(rows),'seeds':[525,526,527]},indent=2))
print('SUMMARY_COMPLETE',len(jobs),flush=True)
