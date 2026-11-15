"""Count content-token occurrences in the 212/636 formula corpora."""
from pathlib import Path
from collections import Counter
import argparse,csv,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from math_tokenizer.eqgpt.compact_alphabet import CompactAlphabet
from math_tokenizer.eqgpt.latex_codec import structure_to_latex

def run(output):
 output.mkdir(parents=True,exist_ok=True);summary=[]
 for scale in [212,636]:
  source=ROOT/f'restarts/stage01_20261113/stage1/corpora/{scale}/sources.jsonl'
  records=[json.loads(x) for x in source.read_text().splitlines()]
  for size in [300,450]:
   modelpath=ROOT/f'restarts/stage01_20261113/stage1/{scale}/tokenizers/{size}.model.json';m=CompactAlphabet.load(modelpath);count=Counter()
   for r in records:
    text=structure_to_latex(r['gre_ids']);ids=m.encode(text);assert m.decode(ids)==text;count.update(ids)
   values=np.array([count[i] for i in range(len(m.vocab))]);total=int(values.sum());assert len(values)==size-3
   with (output/f'{scale}_{size}.csv').open('w',newline='') as f:
    w=csv.writer(f,lineterminator="\n");w.writerow(['display_id','model_id','content_id','token_hex','token','count','frequency_pct'])
    for i,payload in enumerate(m.vocab):w.writerow([i+1,i+3,i,payload.hex(),payload.decode(),int(values[i]),100*values[i]/total])
   summary.append(dict(corpus=scale,vocab_size=size,formulas=len(records),content_tokens=len(values),total_occurrences=total,used=int((values>0).sum()),unused=int((values==0).sum()),top10_share_pct=float(100*np.sort(values)[-10:].sum()/total),vocabulary_sha256=hashlib.sha256(json.dumps([v.hex() for v in m.vocab]).encode()).hexdigest(),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
 (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'analysis/token_frequency');a=p.parse_args();print(json.dumps(run(a.output),indent=2))
