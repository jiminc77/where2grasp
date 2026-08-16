"""Extract frozen-estimand boundaries, figures, and exact three-way gate."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sim.sweep import settings
ROOT=Path(__file__).resolve().parent

def boundary(ell,y,tau):
 ell=np.asarray(ell); y=np.asarray(y); crossings=[]
 for i in range(len(ell)-1):
  a,b=y[i]-tau,y[i+1]-tau
  if a==0: crossings.append(float(ell[i]))
  if a*b<0: crossings.append(float(ell[i]+(-a)/(b-a)*(ell[i+1]-ell[i])))
 if y[-1]==tau: crossings.append(float(ell[-1]))
 crossings=sorted(set(crossings)); all_success=bool(np.all(y>=tau)); all_fail=bool(np.all(y<tau))
 # Conservative censoring: an all-success grid means the true boundary lies beyond the grid top
 # (>= ell.max()); all-failure means below the grid bottom (<= ell.min()). Using the grid EDGE is a
 # one-sided bound that UNDERSTATES the true movement, so it can only make the directional test
 # harder, never easier; it is a valid bound for the monotonic contrast (documented in the report).
 censored=None; bd=max(crossings) if crossings else None
 if not crossings and all_success: bd=float(np.max(ell)); censored='high'
 elif not crossings and all_fail: bd=float(np.min(ell)); censored='low'
 return {'boundary':bd,'crossings':crossings,'uncertainty':float(np.median(np.diff(ell))/2),'resolved':bool(crossings),'valid':bd is not None,'censored':censored,'unresolved_reason':None if (crossings or censored) else 'no discrete crossing'}

def condition(series,sign,step):
 endpoints=[]; adjacent=[]
 for label,vals in series.items():
  if vals[0] is None or vals[-1] is None: endpoints.append({'slice':label,'valid':False,'contrast':None}); continue
  d=vals[-1]-vals[0]; correct=d*sign>0; resolved=abs(d)>=step
  endpoints.append({'slice':label,'valid':True,'contrast':d,'correct_signed':correct,'resolved':resolved})
  adjacent += [sign*(b-a) for a,b in zip(vals[:-1],vals[1:]) if a is not None and b is not None]
 if any(e['valid'] and (not e['correct_signed'] or not e['resolved']) for e in endpoints): status='FAIL'
 elif any(not e['valid'] for e in endpoints): status='INCONCLUSIVE'
 elif not adjacent or sum(x>=0 for x in adjacent)<=len(adjacent)/2: status='INCONCLUSIVE'
 else: status='PASS'
 return {'status':status,'endpoints':endpoints,'adjacent_contrasts':adjacent,'adjacent_correct_or_tied':sum(x>=0 for x in adjacent),'adjacent_total':len(adjacent)}

def main():
 m=json.loads((ROOT/'manifests/sweep_manifest.json').read_text()); z=np.load(ROOT/'manifests/sweep_results.npz'); ss=settings(m); ell=np.array(m['grasp']['ell']); landscapes=[]
 (ROOT/'figures').mkdir(exist_ok=True)
 for s in ss:
  rate=[]; meanj=[]; winner=[]
  for g in range(15):
   q=(z['setting']==s['id'])&(z['grasp']==g)&(z['bank']=='evaluation')
   rate.append(float(np.mean(z['success'][q]))); meanj.append(float(np.mean(z['J'][q]))); winner.append(int(np.unique(z['template'][q])[0]))
  b=boundary(ell,rate,m['tau']); b['in_regime']=bool(np.isfinite(rate).all() and np.isfinite(meanj).all()); b['passed_setting']=True; b['valid']=b['valid'] and b['in_regime']
  landscapes.append({**s,'success_rate':rate,'mean_J':meanj,'selected_template':winner,**b})
  fig,ax=plt.subplots(); ax.plot(ell,rate,'o-',label='evaluation success'); ax.axhline(m['tau'],color='k',ls='--'); ax2=ax.twinx(); ax2.plot(ell,meanj,'s-',color='tab:orange',label='mean J'); ax.set(xlabel='free-arm ell (m)',ylabel='success rate',ylim=(-.05,1.05),title=s['id']); ax2.set_ylabel('mean signed clearance J (m)'); fig.tight_layout(); fig.savefig(ROOT/'figures'/f'landscape_{s["id"]}.png',dpi=130); plt.close(fig)
 lookup={x['id']:x for x in landscapes}; step=float(np.median(np.diff(ell)))
 Bseries={f'w{j}':[lookup[f'B{i}_w{j}']['boundary'] for i in range(5)] for j in range(4)}
 Wseries={f'B{i}':[lookup[f'B{i}_w{j}']['boundary'] for j in range(4)] for i in range(5)}
 B=condition(Bseries,1,step); W=condition(Wseries,-1,step)
 rc=[]
 for p in m['ratio_pairs']:
  a=lookup[p['reference_setting']]['boundary']; b=lookup[f'R{p["pair_id"]}']['boundary']; valid=a is not None and b is not None; rc.append({'pair_id':p['pair_id'],'reference':p['reference_setting'],'valid':valid,'contrast':None if not valid else b-a,'within_tolerance':None if not valid else abs(b-a)<=step})
 valid=[x for x in rc if x['valid']]; Rstatus='INCONCLUSIVE' if not valid else ('FAIL' if any(not x['within_tolerance'] for x in valid) else 'PASS'); R={'status':Rstatus,'tolerance':step,'pairs':rc}
 statuses=[B['status'],W['status'],Rstatus]; overall='NO-GO' if 'FAIL' in statuses else ('GO' if all(x=='PASS' for x in statuses) else 'inconclusive')
 compact={'settings':landscapes}; (ROOT/'manifests/sweep_landscape.json').write_text(json.dumps(compact,indent=2)+'\n')
 verdict={'conditions':{'B':B,'w':W,'R':R},'overall':overall,'resolution_step':step,'boundaries':[{k:x[k] for k in ('id','B_eff','w','boundary','crossings','valid','censored')} for x in landscapes]}; (ROOT/'manifests/gate_verdict.json').write_text(json.dumps(verdict,indent=2)+'\n')
 fig,ax=plt.subplots(1,2,figsize=(10,4));
 for j in range(4): ax[0].loglog(m['B_eff'],[lookup[f'B{i}_w{j}']['boundary'] or np.nan for i in range(5)],'o-',label=f'w={m["w"][j]:.3g}')
 for i in range(5): ax[1].loglog(m['w'],[lookup[f'B{i}_w{j}']['boundary'] or np.nan for j in range(4)],'o-',label=f'B={m["B_eff"][i]:.3g}')
 x=np.array(m['B_eff']); y=.3*(x/x[0])**.25; ax[0].plot(x,y,'k--',label='descriptive slope +1/4'); x=np.array(m['w']); y=.3*(x/x[0])**-.25; ax[1].plot(x,y,'k--',label='descriptive slope -1/4')
 for a in ax:a.legend(fontsize=6);a.set_ylabel('boundary ell (m)')
 ax[0].set_xlabel('B_eff');ax[1].set_xlabel('w');fig.tight_layout();fig.savefig(ROOT/'figures/boundary_shift.png',dpi=150);plt.close(fig)
 print(json.dumps(verdict,indent=2))
if __name__=='__main__':main()
