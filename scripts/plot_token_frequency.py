"""Render the token-usage histograms used in Appendix E."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
ROOT=Path(__file__).resolve().parents[1]
def run(source,output):
 output.parent.mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'font.family':'STIXGeneral','font.size':10,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65})
 fig,axs=plt.subplots(2,2,figsize=(6.55,4.8),layout='constrained',sharex=True,sharey=True);bins=np.linspace(0,2,21);audit=[]
 for j,(scale,size) in enumerate([(212,300),(212,450),(636,300),(636,450)]):
  data=pd.read_csv(source/f'{scale}_{size}.csv');v=data.frequency_pct.to_numpy();counts,edges=np.histogram(v[v<=2],bins=bins);above=int((v>2).sum());assert int(counts.sum())+above==len(v)
  a=axs.flat[j];color='#358572' if scale==212 else '#7A64A3';a.bar(edges[:-1],counts,width=.1,align='edge',color=color,alpha=.68,ec='white',lw=.35);a.bar(2.2,above,width=.16,color=color,alpha=.68,ec='#555555',lw=.6,hatch='///');a.axvline(2.075,c='#bbbbbb',ls=':',lw=.7)
  a.set_xlim(-.02,2.34);a.set_ylim(0,250);a.set_xticks([0,.5,1,1.5,2,2.2]);a.set_xticklabels(['0','0.5','1','1.5','2','>2']);a.tick_params(labelbottom=True);a.yaxis.set_major_locator(MaxNLocator(integer=True,nbins=5));a.grid(axis='y',alpha=.15);a.set_axisbelow(True);a.set_xlabel('Token usage frequency (%)');a.set_title(f'({chr(97+j)}) {scale} formulas, vocabulary size {size}',fontsize=10,pad=8)
  if j%2==0:a.set_ylabel('Number of tokens')
  a.text(.97,.95,f'Zero frequency: {int((v==0).sum())}\nAbove 2%: {above}',transform=a.transAxes,ha='right',va='top',fontsize=9,color='#555555')
  for i,c in enumerate(counts):
   if c>=10:a.text((edges[i]+edges[i+1])/2,c+3,str(c),ha='center',va='bottom',fontsize=7.5,color='#555555')
  audit.append(dict(corpus=scale,vocab_size=size,counts=counts.tolist(),bin_edges_pct=edges.tolist(),above_2_pct=above,total_token_types=len(v)))
 fig.savefig(output.with_suffix('.pdf'),bbox_inches='tight',pad_inches=.04);fig.savefig(output.with_suffix('.png'),dpi=240,bbox_inches='tight',pad_inches=.04);output.with_suffix('.json').write_text(json.dumps(audit,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=ROOT/'analysis/token_frequency');p.add_argument('--output',type=Path,default=ROOT/'analysis/token_frequency/token_usage_histograms');a=p.parse_args();run(a.source,a.output)
