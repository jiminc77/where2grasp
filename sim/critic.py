"""Frozen Step-4 miniature privileged teacher/history student grasp critic.

The evaluation-bank landscape is opened only after all models have been fitted.
The student and explicit-system-ID rows consume the identical wrench-free H_shape
vector: proprio(8), settled shape(8), and known-geometry ell(1).
"""
from __future__ import annotations
import argparse, hashlib, json, logging, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from sim.sweep import settings
from sim.identify import fit_ridge, cv_ridge

ROOT=Path(__file__).resolve().parent
MP=ROOT/'manifests/s34_manifest.json'
FROZEN_DIGEST='ce0c99494ca042e3d781786d0b84a8418f162c853d2a755567aecd267a1f7278'
TRAIN=['B0_w0','B0_w1','B1_w0','B1_w1','B2_w0','B2_w3','B3_w0','B3_w2','B3_w3','B4_w1','B4_w3','R0','R1']
VAL=['B1_w2','B4_w2']; TEST=['B2_w1','B2_w2','B3_w1']
SEED=3403

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def seed_all(seed=SEED): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
class Phi(nn.Module):
 def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(2,32),nn.ReLU(),nn.Linear(32,4))
 def forward(self,x): return self.net(x)
class Critic(nn.Module):
 def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(6,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU(),nn.Linear(32,1))
 def forward(self,g,z): return self.net(torch.cat((g,z),1)).squeeze(1)
class Student(nn.Module):
 def __init__(self,d): super().__init__(); self.net=nn.Sequential(nn.Linear(d,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,4))
 def forward(self,x): return self.net(x)

def train_critic(rows, props, blind=False, train_ids=TRAIN, epochs=300):
 seed_all(); phi=Phi().double(); q=Critic().double()
 pars=list(q.parameters()) if blind else list(q.parameters())+list(phi.parameters()); opt=torch.optim.Adam(pars,lr=1e-2)
 def batch(ids):
  r=[x for x in rows if x[0] in ids]; g=torch.tensor([x[1] for x in r],dtype=torch.double); y=torch.tensor([x[2] for x in r],dtype=torch.double)
  p=torch.tensor([props[x[0]] for x in r],dtype=torch.double); return g,p,y
 tg,tp,ty=batch(train_ids); vg,vp,vy=batch(VAL); best=(float('inf'),None,0); curve=[]
 for ep in range(1,epochs+1):
  opt.zero_grad(); z=torch.zeros((len(tg),4),dtype=torch.double) if blind else phi(tp); loss=nn.functional.binary_cross_entropy_with_logits(q(tg,z),ty); loss.backward(); opt.step()
  with torch.no_grad():
   vz=torch.zeros((len(vg),4),dtype=torch.double) if blind else phi(vp); vl=float(nn.functional.binary_cross_entropy_with_logits(q(vg,vz),vy))
  curve.append([ep,float(loss),vl])
  if vl<best[0]: best=(vl,{k:v.detach().clone() for k,v in q.state_dict().items()},ep) if blind else (vl,({k:v.detach().clone() for k,v in phi.state_dict().items()},{k:v.detach().clone() for k,v in q.state_dict().items()}),ep)
 if blind: q.load_state_dict(best[1])
 else: phi.load_state_dict(best[1][0]); q.load_state_dict(best[1][1])
 return phi,q,{'best_epoch':best[2],'best_val_bce':best[0],'final_train_bce':curve[-1][1],'epochs_budget':epochs},curve

def run(expected_digest):
 live=digest(MP)
 if live!=FROZEN_DIGEST or live!=expected_digest: raise RuntimeError(f'manifest digest mismatch: {live} != frozen/expected')
 cfg=json.loads(MP.read_text()); seed_all(); (ROOT/'logs').mkdir(exist_ok=True); (ROOT/'figures').mkdir(exist_ok=True)
 logging.basicConfig(filename=ROOT/'logs/critic_train.log',filemode='w',level=logging.INFO,format='%(asctime)s %(message)s')
 sm=json.loads((ROOT/'manifests/sweep_manifest.json').read_text()); ss={x['id']:x for x in settings(sm)}
 # Train-only property scaling. This scaler is frozen before validation/evaluation.
 raw={k:np.log10([v['B_eff'],v['w']]) for k,v in ss.items()}; a=np.array([raw[k] for k in TRAIN]); pm,ps=a.mean(0),a.std(0); props={k:(v-pm)/ps for k,v in raw.items()}
 sw=np.load(ROOT/'manifests/sweep_results.npz'); keep=(sw['bank']=='selection')&np.isin(sw['seed'],[2000,2001])
 rows=[]
 for sid in TRAIN+VAL:
  for gi in range(15):
   ix=keep&(sw['setting']==sid)&(sw['grasp']==gi); assert ix.any()
   candidates=[]
   for template in sorted(set(sw['template'][ix].tolist())):
    tx=ix&(sw['template']==template)
    candidates.append((float(sw['J'][tx].mean()),int(template),tx))
   _,selected_template,tx=max(candidates,key=lambda x:(x[0],-x[1]))
   ell=float(sw['ell'][tx][0]); rows.append((sid,np.array([gi/14.,ell]),float(sw['success'][tx].mean())))
 phi,q,ts,tc=train_critic(rows,props); _,qb,bs,bc=train_critic(rows,props,blind=True)
 logging.info('teacher %s',ts); logging.info('blind %s',bs)
 # Wrench-free H_shape. Fit student only on train/val bank, evaluate seed 2002 only.
 hz=np.load(ROOT/'manifests/histories_s3.npz'); assert str(hz['manifest_digest'].item())==live
 def hf(i): return np.r_[hz['proprio'][i],hz['shape'][i],sm['grasp']['ell'][int(hz['grasp'][i])]]
 hX=np.array([hf(i) for i in range(len(hz['setting']))]); hS=hz['setting'].astype(str); hseed=hz['seed']; hm=hX[np.isin(hS,TRAIN)&np.isin(hseed,[2000,2001])].mean(0); hs=hX[np.isin(hS,TRAIN)&np.isin(hseed,[2000,2001])].std(0); hs[hs<1e-12]=1
 ti=np.isin(hS,TRAIN)&np.isin(hseed,[2000,2001]); vi=np.isin(hS,VAL)&np.isin(hseed,[2000,2001])
 X=torch.tensor((hX[ti]-hm)/hs,dtype=torch.double); V=torch.tensor((hX[vi]-hm)/hs,dtype=torch.double)
 with torch.no_grad(): Z=phi(torch.tensor([props[s] for s in hS[ti]],dtype=torch.double)); VZ=phi(torch.tensor([props[s] for s in hS[vi]],dtype=torch.double))
 student=Student(hX.shape[1]).double(); opt=torch.optim.Adam(student.parameters(),lr=1e-2); best=(float('inf'),None,0)
 for ep in range(1,301):
  opt.zero_grad(); loss=((student(X)-Z)**2).mean(); loss.backward(); opt.step()
  with torch.no_grad(): vl=float(((student(V)-VZ)**2).mean())
  if vl<best[0]: best=(vl,{k:v.detach().clone() for k,v in student.state_dict().items()},ep)
 student.load_state_dict(best[1]); ds={'best_epoch':best[2],'best_val_latent_mse':best[0],'final_train_latent_mse':float(loss),'epochs_budget':300}; logging.info('student %s',ds)
 # Step-3 ridge: same H_shape schema and deterministic grouped CV, with no wrench.
 s3=set(cfg['splits']['step3_train']); ri=np.isin(hS,list(s3))&np.isin(hseed,[2000,2001]); rX=hX[ri]; rS=hS[ri]; rY=np.array([np.log10([ss[s]['B_eff'],ss[s]['w'],ss[s]['B_eff']/ss[s]['w']]) for s in rS]); ridge,alpha,cvs,cvt=cv_ridge(rX,rY,rS,cfg)
 logging.info('sysid ridge alpha=%s cv=%s',alpha,min(cvs))
 # Oracle is deliberately loaded only now, after fitting is complete.
 landscape=json.loads((ROOT/'manifests/sweep_landscape.json').read_text())['settings']; land={x['id']:x for x in landscape}; all23=list(land); inreg=[s for s in all23 if s not in ['B0_w2','B0_w3','B1_w3','B4_w0']]
 def zhist(s,L,mode):
  ix=np.where((hS==s)&(hseed==2002))[0]; ix=ix[np.lexsort((hz['template'][ix],hz['grasp'][ix]))]
  if not len(ix): return None
  take=ix[:L]
  if mode=='student':
   with torch.no_grad(): return student(torch.tensor((hX[take]-hm)/hs,dtype=torch.double)).mean(0).numpy()
  pred=ridge.predict(hX[take]).mean(0); p=(pred[:2]-pm)/ps
  with torch.no_grad(): return phi(torch.tensor(p[None],dtype=torch.double)).numpy()[0]
 def predict_curve(s,row,L=3):
  if row=='blind': z=np.zeros(4); qq=qb
  elif row=='teacher':
   with torch.no_grad(): z=phi(torch.tensor(props[s][None],dtype=torch.double)).numpy()[0]
   qq=q
  else: z=zhist(s,L,'student' if row=='student' else 'sysid'); qq=q
  # Missing out-of-regime histories use the explicitly labeled cold-start prior.
  if z is None: z=np.zeros(4)
  gf=np.c_[np.arange(15)/14.,np.asarray(sm['grasp']['ell'])]
  with torch.no_grad():
   logits=qq(torch.tensor(gf,dtype=torch.double),torch.tensor(np.tile(z,(15,1)),dtype=torch.double))
   return torch.sigmoid(logits).numpy()
 def choose(s,row,L=3): return int(np.argmax(predict_curve(s,row,L)))
 def metric(row,domain,L=3):
  out=[]
  for s in domain:
   g=choose(s,row,L); d=land[s]; out.append({'setting':s,'chosen_grasp':g,'mean_J_regret':float(max(d['mean_J'])-d['mean_J'][g]),'success_rate_regret':float(max(d['success_rate'])-d['success_rate'][g]),'history_available':bool(np.any((hS==s)&(hseed==2002)))})
  return {'mean_J_regret':float(np.mean([x['mean_J_regret'] for x in out])),'success_rate_regret':float(np.mean([x['success_rate_regret'] for x in out])),'per_setting':out}
 rows_out={}
 for r in ['teacher','blind','student','explicit_sysid']:
  key='sysid' if r=='explicit_sysid' else r; rows_out[r]={'test':metric(key,TEST),'in_regime_19':metric(key,inreg),'all_23_secondary':metric(key,all23)}
 probe=metric('sysid',inreg,L=1)
 # Additional diagnostic: recover the material-dependent success map and its
 # tau=0.5 boundary. Linear interpolation makes the crossing error continuous
 # in grasp-index units while preserving the frozen threshold and ordering.
 def crossing(curve,tau=.5):
  y=np.asarray(curve,dtype=float)
  for i in range(len(y)-1):
   if (y[i]-tau)*(y[i+1]-tau)<=0 and y[i]!=y[i+1]:
    return float(i+(tau-y[i])/(y[i+1]-y[i]))
  return 0.0 if np.all(y<tau) else float(len(y)-1)
 def map_recovery(row):
  per=[]
  for sid in TEST:
   pred=predict_curve(sid,row); measured=np.asarray(land[sid]['success_rate'],dtype=float)
   corr=float(np.corrcoef(pred,measured)[0,1]) if np.std(pred)>0 and np.std(measured)>0 else None
   pb,mb=crossing(pred),crossing(measured)
   per.append({'setting':sid,'map_rmse':float(np.sqrt(np.mean((pred-measured)**2))),'map_correlation':corr,'predicted_boundary_index':pb,'measured_boundary_index':mb,'boundary_error_index':abs(pb-mb),'predicted_success_curve':pred.tolist(),'measured_success_curve':measured.tolist()})
  return {'domain':'held-out TEST settings','curve_target':'measured evaluation-bank success_rate','boundary_rule':'first tau=0.5 crossing, linearly interpolated in grasp-index units','map_rmse':float(np.mean([x['map_rmse'] for x in per])),'map_correlation':float(np.mean([x['map_correlation'] for x in per if x['map_correlation'] is not None])),'boundary_error_index':float(np.mean([x['boundary_error_index'] for x in per])),'per_setting':per}
 secondary={r:map_recovery('sysid' if r=='explicit_sysid' else r) for r in rows_out}
 adaptation=[]
 for L in [0,1,3]:
  m=metric('blind' if L==0 else 'student',TEST,L=max(L,1)); adaptation.append({'interactions':L,'mean_J_regret':m['mean_J_regret'],'success_rate_regret':m['success_rate_regret'],'domain':'held-out TEST settings','kind':'offline deterministic history-prefix ablation'})
 # Property-level sanity: fixed architecture/budget, nested properties; N is settings, not histories.
 lc=[]
 for n in [4,8,13]:
  _,_,s,_=train_critic(rows,props,train_ids=TRAIN[:n]); lc.append({'n_train_properties':n,'best_validation_bce':s['best_val_bce'],'best_epoch':s['best_epoch']})
 result={'schema_version':1,'manifest_digest':live,'assertions':{'critic_input_dim':6,'grasp_feature_dim':2,'latent_dim':4,'train_val_test_disjoint':not(set(TRAIN)&set(VAL) or set(TRAIN)&set(TEST) or set(VAL)&set(TEST)),'train_label_seeds':[2000,2001],'held_out_history_seed':2002,'oracle_seeds':[3000,3001,3002,3003,3004],'sysid_and_student_same_history_schema':True,'sysid_primary_wrench_free':True,'R2_and_B2_w1_excluded_from_fitting':('R2' not in TRAIN+VAL and 'B2_w1' not in TRAIN+VAL)},'splits':{'train':TRAIN,'validation':VAL,'test':TEST},'training':{'seed':SEED,'teacher':ts,'blind':bs,'student_distillation':ds,'sysid_ridge':{'alpha':alpha,'mean_grouped_cv_rmse':min(cvs)},'property_learning_curve':lc,'critic_label':'selection-bank seeds 2000-2001 per-setting/grasp success rate'},'rows':rows_out,'secondary_map_recovery':secondary,'prescribed_probe_one_shot':{'row':'explicit_sysid','mode':'single deterministic held-out diagnostic history then act','in_regime_19':probe},'adaptation_curve':adaptation,'domain_notes':{'primary':'19 in-regime settings; mean-J regret','all_23_secondary':'Four settings lack real replay histories; history rows use labeled z=0 cold-start fallback there.','oracle':'evaluation-bank measured landscape, never used in fitting','adaptation':'offline history-length ablation, not on-policy','student_probe':'Not run: a valid small-deflection Pi_g<=0.5 probe requires new real Genesis replay data and mechanics, so it is recorded as follow-up rather than fabricated or allowed to delay the cheap map-recovery diagnostic.'}}
 (ROOT/'manifests/critic_results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 labels=list(rows_out); vals=[rows_out[x]['in_regime_19']['mean_J_regret'] for x in labels]; fig,ax=plt.subplots(figsize=(7,4)); ax.bar(labels,vals); ax.set_ylabel('mean-J selection regret (19 in-regime)'); ax.tick_params(axis='x',rotation=15); fig.tight_layout(); fig.savefig(ROOT/'figures/regret_by_row.png',dpi=180); plt.close(fig)
 fig,ax=plt.subplots(figsize=(5,3.5)); ax.plot([x['interactions'] for x in adaptation],[x['mean_J_regret'] for x in adaptation],'o-'); ax.set(xlabel='offline history interactions',ylabel='mean-J regret (held-out TEST)',xticks=[0,1,3]); fig.tight_layout(); fig.savefig(ROOT/'figures/adaptation_curve.png',dpi=180); plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(9,3.8)); labels=list(secondary)
 axes[0].bar(labels,[secondary[x]['map_rmse'] for x in labels]); axes[0].set_ylabel('success-map RMSE')
 axes[1].bar(labels,[secondary[x]['boundary_error_index'] for x in labels]); axes[1].set_ylabel('tau=0.5 boundary error (index)')
 for ax in axes: ax.tick_params(axis='x',rotation=20)
 fig.suptitle('Secondary map recovery on held-out TEST settings'); fig.tight_layout(); fig.savefig(ROOT/'figures/map_recovery_by_row.png',dpi=180); plt.close(fig)
 logging.info('complete rows=%s adaptation=%s map_recovery=%s', {r:rows_out[r]['in_regime_19']['mean_J_regret'] for r in rows_out},adaptation,{r:{'rmse':v['map_rmse'],'correlation':v['map_correlation'],'boundary_error':v['boundary_error_index']} for r,v in secondary.items()})
 print(json.dumps({'manifest_digest':live,'mean_J_regret_in_regime':{r:rows_out[r]['in_regime_19']['mean_J_regret'] for r in rows_out},'secondary_map_recovery':{r:{'map_rmse':v['map_rmse'],'map_correlation':v['map_correlation'],'boundary_error_index':v['boundary_error_index']} for r,v in secondary.items()},'adaptation':adaptation})); return result
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--expected-digest',required=True); a=p.parse_args(); run(a.expected_digest)
