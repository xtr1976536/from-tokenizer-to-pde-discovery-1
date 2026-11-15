"""Pool the primary vocabulary-size 300/450 candidate statistics over eight cases."""
from pathlib import Path
import argparse,json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def run(output):
 output.mkdir(parents=True,exist_ok=True);df=pd.read_csv(ROOT/'paper/data/seed525_novelty.csv');df=df[df.arm.isin(['ast_bpe_300','ast_bpe_450']) & (df.sampling=='mix20')];rows=[]
 for size in [300,450]:
  q=df[df.arm==f'ast_bpe_{size}'];assert len(q)==8
  rows.append(dict(vocab_size=size,cases=len(q),total_candidates=int(q.scored.sum()),new_structure_occurrences=int(q.new_scored.sum()),new_structure_pct=100*q.new_scored.sum()/q.scored.sum(),new_structure_top10_slots=int(q.new_final.sum()),special_candidate_occurrences=int(q.high_wrong_count.sum()),special_candidate_unique_per_run_sum=int(q.high_wrong_unique.sum()),special_top10_slots=int(q.high_wrong_final.sum()),cases_with_special_in_top10=int((q.high_wrong_final>0).sum())))
 pd.DataFrame(rows).to_csv(output/'primary_candidate_summary.csv',index=False);return rows
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'analysis/paper_revision');a=p.parse_args();print(json.dumps(run(a.output),indent=2))
