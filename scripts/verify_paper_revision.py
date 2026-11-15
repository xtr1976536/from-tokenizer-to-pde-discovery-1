"""Check the paper-revision statistics and bilingual result invariants."""
from pathlib import Path
import json,re
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parents[1]
def verify():
 records=[]
 for scale in [212,636]:
  for size in [300,450]:
   t=pd.read_csv(ROOT/f'analysis/token_frequency/{scale}_{size}.csv',keep_default_na=False)
   assert len(t)==size-3 and t.content_id.to_list()==list(range(size-3))
   assert abs(t.frequency_pct.sum()-100)<1e-9
   assert np.allclose(t.frequency_pct,100*t['count']/t['count'].sum(),rtol=1e-12)
   records.append(dict(corpus=scale,vocabulary=size,used=int((t['count']>0).sum()),total=len(t)))
 summary=pd.read_csv(ROOT/'analysis/paper_revision/primary_candidate_summary.csv')
 assert summary.total_candidates.tolist()==[16000,16000]
 assert summary.new_structure_occurrences.tolist()==[11517,12561]
 assert summary.special_candidate_occurrences.tolist()==[7,18]
 assert summary.cases_with_special_in_top10.tolist()==[4,4]
 en=ROOT/'paper/latex-en';zh=ROOT/'paper/latex'
 for p in (zh/'tables').glob('*.tex'):
  t=en/'tables'/p.name
  assert t.exists()
  assert re.findall(r'\d+(?:\.\d+)?',p.read_text())==re.findall(r'\d+(?:\.\d+)?',t.read_text()),p.name
  assert re.findall(r'\\ding\{\d+\}',p.read_text())==re.findall(r'\\ding\{\d+\}',t.read_text()),p.name
 for lang in [en,zh]:
  s=(lang/'main.tex').read_text();assert 'https://github.com/xtr1976536/from-tokenizer-to-pde-discovery-1' in s
  assert len(re.findall(r'\\section\{',s))==10
 assert not re.search(r'[\u4e00-\u9fff]',(en/'main.tex').read_text())
 print(json.dumps(dict(status='PASS',token_distributions=records,candidate_counts='PASS',bilingual_tables='PASS'),indent=2))
if __name__=='__main__':verify()
