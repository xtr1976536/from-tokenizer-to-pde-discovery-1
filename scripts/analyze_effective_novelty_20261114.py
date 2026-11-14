# -*- coding: utf-8 -*-
import json,gzip,hashlib,collections,functools,csv,time
import numpy as np
from pathlib import Path
from math_tokenizer.eqgpt.stage1_search_20261113 import structure_id
from math_tokenizer.eqgpt.author import DICTIONARY
from math_tokenizer.eqgpt.latex_codec import structure_to_latex
from math_tokenizer.eqgpt.data import canonical
from math_tokenizer.eqgpt.evaluation import expanded_terms
ROOT=Path('/data1/users/liugang/src/eq-bpe/project');B=ROOT/'restarts/stage01_20261113';S=B/'stage1';O=B/'closing/task3/effective';O.mkdir(exist_ok=True)
rowsout=[];dist=[];examples=[];pooled=[];symmetry=[];plot=collections.defaultdict(list)
@functools.lru_cache(maxsize=200000)
def termset(gre):return frozenset(repr(t) for term in canonical([DICTIONARY['id2word'][int(i)] for i in gre]) for t in expanded_terms(term))
training={};trainterms={};poolids={}
for scale in [212,636]:
 records=[json.loads(x) for x in (S/f'corpora/{scale}/sources.jsonl').read_text().splitlines()];sp=json.loads((S/str(scale)/'split.json').read_text());ids=set(sp['train']);training[scale]={structure_id(r['gre_ids']) for r in records if r['source_id'] in ids};poolids[scale]={structure_id(r['gre_ids']) for r in records};trainterms[scale]=set().union(*(termset(tuple(r['gre_ids'])) for r in records if r['source_id'] in ids))
newterms=trainterms[636]-trainterms[212];newtrain=training[636]-poolids[212]
jobs=json.loads((S/'downstream_jobs.json').read_text());keys=sorted({(j['arm'],j['sampling'],j['case']) for j in jobs});start=time.time()
def quant(v):
 a=np.array(v,dtype=float)
 return dict(n=len(a),mean=float(a.mean()) if len(a) else None,**{f'q{int(q*100):02}':float(np.quantile(a,q)) if len(a) else None for q in [.05,.25,.5,.75,.95]})
for gi,(arm,mode,case) in enumerate(keys):
 runs={}
 for scale in [212,636]:
  for seed in [525,526,527]:
   p=S/'search/results'/str(scale)/arm/mode/str(seed)/case;h=json.loads((p/'history.json').read_text())['history'];final=set(h[-1]['candidate_ids']);rr=[];attempts=0
   with gzip.open(p/'candidates.jsonl.gz','rt') as f:
    for line in f:
     r=json.loads(line);attempts+=1;sid=r.get('effective_structure_id')
     if not sid:continue
     rr.append(dict(sid=sid,eff=r.get('effective_structure_id'),scored=r['status']=='evaluated',reward=r.get('reward'),cycle=r['cycle'],final=r['candidate_id'] in final,selected=r.get('selected_this_cycle',False),gre=r.get('gre'),text=r.get('text'),correct=r.get('effective_classification')=='correct'))
   runs[(scale,seed)]=(rr,attempts)
 for scale in [212,636]:
  other_scale=636 if scale==212 else 212
  cross_pool={r['sid'] for seed in [525,526,527] for r in runs[(other_scale,seed)][0]}
  for seed in [525,526,527]:
   allr,att=runs[(scale,seed)];sc=[r for r in allr if r['scored']];other_seed_pool={r['sid'] for t in [525,526,527] if t!=seed for r in runs[(scale,t)][0]}
   motif=sum(bool(termset(tuple(r['gre']))&newterms) for r in sc if r['gre']);unique={r['sid'] for r in allr}
   symmetry.append(dict(arm=arm,sampling=mode,case=case,scale=scale,seed=seed,scored=len(sc),unique_parseable=len(unique),cross_scale_unseen_scored=sum(r['sid'] not in cross_pool for r in sc),within_scale_other_seeds_unseen_scored=sum(r['sid'] not in other_seed_pool for r in sc),new_term_motif_scored=motif))
 smallpool={r['sid'] for seed in [525,526,527] for r in runs[(212,seed)][0]};smallfirst={r['sid'] for seed in [525,526,527] for r in runs[(212,seed)][0] if r['cycle']==1};bigall={r['sid'] for seed in [525,526,527] for r in runs[(636,seed)][0]}
 for seed in [525,526,527]:
  small={r['sid'] for r in runs[(212,seed)][0]};large,attempts=runs[(636,seed)];base=dict(arm=arm,sampling=mode,case=case,seed=seed)
  for r in large:
   r['novel_pooled']=r['sid'] not in smallpool;r['novel_paired']=r['sid'] not in small;r['novel_train212']=r['sid'] not in training[212];r['matches_added_training']=r['sid'] in newtrain;r['new_term_motif']=bool(termset(tuple(r['gre']))&newterms) if r['gre'] else False
  for flag in ['novel_pooled','novel_paired','novel_train212','matches_added_training','new_term_motif']:
   scored=[r for r in large if r['scored']];novel=[r for r in large if r[flag]];ns=[r for r in scored if r[flag]];final=[r for r in large if r['final']];nf=[r for r in final if r[flag]];uniq={r['sid'] for r in large};un={r['sid'] for r in novel};us={r['sid'] for r in scored};uns={r['sid'] for r in ns};unf={r['sid'] for r in nf}
   rowsout.append(dict(base,definition=flag,attempts=attempts,parseable=len(large),scored=len(scored),unique_parseable=len(uniq),unique_scored=len(us),novel_parseable=len(novel),novel_scored=len(ns),unique_novel_parseable=len(un),unique_novel_scored=len(uns),novel_selected_current=sum(r['selected'] for r in ns),final_slots=len(final),novel_final_slots=len(nf),unique_novel_final=len(unf),novel_first_cycle_scored=sum(r['cycle']==1 for r in ns),first_cycle_scored=sum(r['cycle']==1 for r in scored),novel_correct=sum(r['correct'] for r in ns),novel_final_correct=sum(r['correct'] for r in nf),final_correct=sum(r['correct'] for r in final)))
  scored=[r for r in large if r['scored']];vals=np.array([r['reward'] for r in scored]);order=np.argsort(vals);ranks=np.empty(len(vals));i=0
  while i<len(order):
   j=i+1
   while j<len(order) and vals[order[j]]==vals[order[i]]:j+=1
   ranks[order[i:j]]=((i+j-1)/2+.5)/len(order);i=j
  for novel in [False,True]:
   selected=[r for r in scored if r['novel_pooled']==novel];scores=[r['reward'] for r in selected];pct=[float(ranks[i]) for i,r in enumerate(scored) if r['novel_pooled']==novel];dist.append(dict(base,novel=novel,measure='raw_reward',**quant(scores)));dist.append(dict(base,novel=novel,measure='within_run_percentile',**quant(pct)));plot[case+('_new' if novel else '_seen')].extend(scores);plot['rank_'+arm+'_'+mode+('_new' if novel else '_seen')].extend(pct)
   by=collections.defaultdict(list)
   for r in selected:by[r['sid']].append(r['reward'])
   dist.append(dict(base,novel=novel,measure='unique_structure_mean_reward',**quant([np.mean(v) for v in by.values()])))
  for r in sorted((r for r in scored if r['novel_pooled']),key=lambda x:(not x['final'],-x['reward']))[:2]:examples.append(dict(base,structure=r['sid'],formula=structure_to_latex(r['gre']),reward=r['reward'],final_top10=r['final'],correct=r['correct'],seen_in_train212=not r['novel_train212'],matches_added_training=r['matches_added_training'],new_term_motif=r['new_term_motif']))
 pooled.append(dict(arm=arm,sampling=mode,case=case,small_unique=len(smallpool),large_unique=len(bigall),large_unique_new=len(bigall-smallpool),shared_unique=len(bigall&smallpool),small_first_cycle_unique=len(smallfirst)))
 if gi%8==7:print('GROUPS',gi+1,'/',len(keys),flush=True)
def csvout(name,rows):
 with (O/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
csvout('symmetry_and_seed_control.csv',symmetry);csvout('novelty_per_run.csv',rowsout);csvout('reward_distributions.csv',dist);csvout('candidate_examples.csv',examples);csvout('pooled_unique_structures.csv',pooled);np.savez_compressed(O/'plot_arrays.npz',**{k:np.asarray(v,dtype=np.float32) for k,v in plot.items()})
(O/'novelty_analysis_complete.json').write_text(json.dumps(dict(status='PASS',paired_groups=len(keys),large_runs=168,small_runs=168,run_definition_rows=len(rowsout),training212_structures=len(training[212]),training636_structures=len(training[636]),added_train_structures_vs_full212=len(newtrain),small_training_terms=len(trainterms[212]),large_training_terms=len(trainterms[636]),added_term_motifs=len(newterms),seconds=time.time()-start),indent=2));print('NOVELTY_DONE',flush=True)
