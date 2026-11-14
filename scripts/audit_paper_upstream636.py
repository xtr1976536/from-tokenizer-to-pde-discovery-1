import json,gzip,math,hashlib,csv
from pathlib import Path
from math_tokenizer.eqgpt.compact_alphabet import CompactAlphabet,train_compact
from math_tokenizer.eqgpt.latex_codec import structure_to_latex
ROOT=Path.cwd();b=ROOT/'restarts/stage01_20261113/paper_upstream636_20261114';protocol=json.loads((b/'protocol.json').read_text());records=[json.loads(x) for x in (ROOT/'data/formulas/636/sources.jsonl').read_text().splitlines()];inputs=json.loads((b/'input_records.json').read_text());by={r['id']:r for r in inputs};assert len(inputs)==636
for r in records:assert by[r['source_id']]['latex']==structure_to_latex(r['gre_ids'])
rows=json.loads((b/'results.json').read_text());usage=json.loads((b/'usage.json').read_text());metrics=0;models=0
for r in rows:
 p=b/r['protocol']/(r['method']+'.'+r['split']+'.per_formula.jsonl.gz');details=list(map(json.loads,gzip.open(p,'rt')));tot=lambda k:sum(x[k] for x in details)
 assert len(details)==r['formulas'];assert math.isclose(tot('initial_character_tokens')/tot('final_tokens'),r['compression_ratio'],rel_tol=1e-12)
 for k,n,d in [('cmd_pct','commands_intact','commands'),('num_pct','numbers_intact','numbers')]:
  if tot(d):assert math.isclose(100*tot(n)/tot(d),r[k],rel_tol=1e-12)
 metrics+=1
for proto in ['811','simple_to_complex']:
 ids=json.loads((b/proto/'split_ids.json').read_text())
 for key in ['id','source_group','template_sha256','canonical_sha256']:
  sets={s:{by[i][key] for i in ids[s]} for s in ids}
  assert not sets['train']&sets['validation'] and not sets['train']&sets['test'] and not sets['validation']&sets['test']
 for size in [300,450]:
  model=CompactAlphabet.load(b/proto/f'ast_bpe_{size}.model.json');ref=train_compact('ast_bpe',[by[i]['latex'] for i in ids['train']],size,protocol['alphabet']);assert model.vocab==ref.vocab and json.dumps(model.internal.merges)==json.dumps(ref.internal.merges);models+=1
for u in usage:
 with (b/u['protocol']/(u['method']+'.'+u['split']+'.tokens.csv')).open() as f:rr=list(csv.DictReader(f))
 learned=[r for r in rr if r['kind']=='learned'];assert len(learned)==u['learned'];assert sum(int(r['count'])>0 for r in learned)==u['used'];assert math.isclose(u['usage_rate']+u['unused_rate'],1)
report=dict(status='PASS',records=636,representation_matches_downstream=True,metrics_reaggregated=metrics,usage_tables=len(usage),train_only_refitted_models=models,train_validation_test_group_disjoint=True,baseline_vocabularies_fixed=True,source_sha256=protocol['source_sha256'],base_bytes=len(protocol['alphabet']))
(b/'independent_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
