# -*- coding: utf-8 -*-
from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parent;F=B/'latex/figures';T=B/'latex/tables';D=json.loads((B/'data/downstream.json').read_text());R=pd.DataFrame([dict(x['job'],**x['mechanism']) for x in D['cases']]);P=R[R.seed==525]
MINT='#A9DFCF';PURPLE='#C5B9E2';GREEN='#358572';VIOLET='#7A64A3';GRAY='#777777'
plt.rcParams.update({'font.family':'STIXGeneral','font.size':9,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'axes.spines.top':True,'axes.spines.right':True,'axes.linewidth':.55,'pdf.fonttype':42,'lines.linewidth':1.4})
def save(fig,n):
 fig.savefig(F/(n+'.pdf'),bbox_inches='tight',pad_inches=.03);fig.savefig(F/(n+'.png'),dpi=260,bbox_inches='tight',pad_inches=.03);plt.close(fig)
U=json.loads((B/'data/upstream636/results.json').read_text());US=json.loads((B/'data/upstream636/usage.json').read_text());v=sorted([r for r in U if r['protocol']=='811' and r['split']=='validation' and r['method'].startswith('ast_bpe')],key=lambda x:x['vocab_size']);x=[r['vocab_size'] for r in v];fig,axs=plt.subplots(1,2,figsize=(5.5,2.1),layout='constrained')
for ax,ys,c,label in [(axs[0],[r['compression_ratio'] for r in v],GREEN,'Compression (bytes/token)'),(axs[1],[100*next(u['usage_rate'] for u in US if u['protocol']=='811' and u['split']=='validation' and u['method']==r['method']) for r in v],VIOLET,'Subword utilization (%)')]:
 ax.plot(x,ys,color=c,marker='o',ms=2.8)
 for cap in [300,450]:
  j=x.index(cap);ax.scatter(cap,ys[j],s=28,fc='white',ec=c,zorder=5);ax.annotate(str(cap),(cap,ys[j]),xytext=(0,8 if ax==axs[0] else -13),textcoords='offset points',ha='center',fontsize=7.5)
 ax.set_xlabel('Vocabulary size');ax.grid(alpha=.15)
 ax.set_ylabel(label)
save(fig,'capacity_tradeoff')
fig,axs=plt.subplots(1,2,figsize=(5.5,2.25),layout='constrained',sharey=True)
for ax,s in zip(axs,[212,636]):
 for off,mode,color,edge,label in [(-.18,'model',MINT,GREEN,'Model sampling'),(.18,'mix20',PURPLE,VIOLET,'20% uniform')]:
  yy=[100*P[(P.scale==s)&(P.arm==f'ast_bpe_{n}')&(P.sampling==mode)].top1_correct.mean() for n in [200,300,450]];ax.bar(np.arange(3)+off,yy,.34,color=color,edgecolor=edge,lw=.65,label=label)
 ax.set_xticks(range(3));ax.set_xticklabels(['200','300','450']);ax.set_ylim(0,105);ax.set_xlabel('Vocabulary size');ax.set_title(f'{s} formulas',fontsize=9);ax.grid(axis='y',alpha=.12)
axs[0].set_ylabel('Top-1 recovery (%)');axs[0].legend(frameon=False,fontsize=7,loc='upper left');save(fig,'vocab_sampling')
N=pd.read_csv(B/'data/seed525_novelty.csv');A=N[N.arm!='eqgpt_reference'];g=A.groupby(['arm','sampling']).sum(numeric_only=True)
fig,axs=plt.subplots(1,2,figsize=(5.5,2.2),layout='constrained');x=np.arange(2)
for off,col,color,edge,label in [(-.18,'new_scored',MINT,GREEN,'Scored candidates'),(.18,'new_final',PURPLE,VIOLET,'Final Top-10')]:
 vals=[100*g.loc[(f'ast_bpe_{n}','mix20'),col]/(g.loc[(f'ast_bpe_{n}','mix20'),'scored'] if col=='new_scored' else 80) for n in [300,450]];axs[0].bar(x+off,vals,.34,color=color,edgecolor=edge,lw=.65,label=label)
axs[0].set_xticks(x);axs[0].set_xticklabels(['300','450']);axs[0].set_xlabel('Vocabulary size');axs[0].set_ylabel('Previously unobserved (%)');axs[0].set_ylim(0,105);axs[0].legend(frameon=False,fontsize=7)
vals=[]
for s in [212,636]:
 a=P[(P.scale==s)&(P.arm!='eqgpt_reference')];vals.append(100*a.effective_correct_scored_occurrences.sum()/a.evaluated.sum())
axs[1].bar(x,vals,color=[MINT,PURPLE],edgecolor=[GREEN,VIOLET],lw=.7,width=.55)
for xx,yy in zip(x,vals):axs[1].text(xx,yy+.05,f'{yy:.2f}',ha='center',fontsize=8)
axs[1].set_xticks(x);axs[1].set_xticklabels(['212','636']);axs[1].set_xlabel('Corpus size');axs[1].set_ylabel('Correct candidates (%)');axs[1].set_ylim(0,max(vals)*1.25)
for ax in axs:ax.grid(axis='y',alpha=.12)
save(fig,'structure_analysis')
a=np.load(B/'data/seed525_rank_scores.npz');fig,ax=plt.subplots(figsize=(5.5,2.3),layout='constrained')
for key,c,label in [('seen',GREEN,'Previously observed'),('new',VIOLET,'Previously unobserved')]:
 q=np.linspace(0,1,501);ax.plot(np.quantile(a[key],q),q,color=c,label=label)
ax.plot([0,1],[0,1],c=GRAY,ls='--',lw=.6);ax.set_xlabel('Within-run reward percentile');ax.set_ylabel('Cumulative fraction');ax.legend(frameon=False);save(fig,'reward_distribution')
# Wide main table: cases, not corpus-averaged scores.
cases=['Wave_equation','Convection_diffusion_equation','KG_equation','Burgers_equation','Eq_6_2_12','PDE_divide','KdV_equation','Chaffee_Infante_equation'];short=['Wave','CD','KG','Burgers','E6','PD','KdV','CI'];keys=[('eqgpt_reference','legacy','EqGPT'),('ast_bpe_300','mix20','AST-BPE 300'),('ast_bpe_450','mix20','AST-BPE 450')]
lines=[r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llccccccccr@{}}',r'\toprule',r'语料 & 模型 & '+' & '.join(short)+r' & 总计\\',r'\midrule']
for scale in [212,636]:
 group=[P[(P.scale==scale)&(P.arm==a)&(P.sampling==m)] for a,m,_ in keys];best1=max(x.top1_correct.sum() for x in group);best10=max(x.top10_correct.sum() for x in group)
 for j,((arm,mode,label),rr) in enumerate(zip(keys,group)):
  cells=[]
  for case in cases:
   z=rr[rr.case==case].iloc[0];cells.append(('\\ding{51}' if z.top1_correct else '--')+'/'+('\\ding{51}' if z.top10_correct else '--'))
  n1=int(rr.top1_correct.sum());n10=int(rr.top10_correct.sum());total=(r'\textbf{'+str(n1)+'}' if n1==best1 else str(n1))+'/'+(r'\textbf{'+str(n10)+'}' if n10==best10 else str(n10));lines.append(' & '.join([str(scale) if j==0 else '',label]+cells+[total])+r'\\')
 if scale==212:lines.append(r'\midrule')
lines += [r'\bottomrule',r'\end{tabular*}'];(T/'main_results.tex').write_text('\n'.join(lines))
# Full-width upstream tables; highlight best values, including ties.
for proto in ['811','simple_to_complex']:
 rr=[r for r in U if r['protocol']==proto and r['split']=='test'];best=[max(r[k] for r in rr) for k in ['compression_ratio','cmd_pct','num_pct']];uses={r['method']:next((u['usage_rate'] for u in US if u['protocol']==proto and u['split']=='test' and u['method']==r['method']),None) for r in rr};bu=max(v for v in uses.values() if v is not None)
 labels={'ast_bpe_300':'AST-BPE 300','ast_bpe_450':'AST-BPE 450','qwen2.5':'Qwen2.5','llama3':'Llama 3','o200k_base':r'\texttt{o200k\_base}'};ls=[r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrr@{}}',r'\toprule',r'分词器 & 压缩率 & 命令完整率(\%) & 数字完整率(\%) & 使用率(\%)\\',r'\midrule']
 for r in rr:
  cells=[]
  for k,vbest,dec in zip(['compression_ratio','cmd_pct','num_pct'],best,[3,1,1]):
   val=f'{r[k]:.{dec}f}';cells.append(r'\textbf{'+val+'}' if r[k]==vbest else val)
  u=uses[r['method']];cells.append('--' if u is None else ((r'\textbf{'+f'{100*u:.1f}'+'}') if u==bu else f'{100*u:.1f}'));ls.append(' & '.join([labels[r['method']]]+cells)+r'\\')
 ls += [r'\bottomrule',r'\end{tabular*}'];(T/f'upstream_{proto}.tex').write_text('\n'.join(ls))
print('V2_ASSETS_READY','density',vals)
