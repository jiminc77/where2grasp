"""Frozen Step-3 channel-wise identifiability estimand."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sim.sweep import settings
from sim.history import manifest_digest
ROOT=Path(__file__).resolve().parent; MP=ROOT/'manifests/s34_manifest.json'
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rmse(a,b): return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))
class LinearModel:
 def __init__(self,mu,sd,W): self.mu,self.sd,self.W=mu,sd,W
 def predict(self,X): return np.c_[np.ones(len(X)),(X-self.mu)/self.sd]@self.W
def fit_ridge(X,Y,alpha):
 mu=X.mean(0); sd=X.std(0); sd[sd<1e-12]=1; A=np.c_[np.ones(len(X)),(X-mu)/sd]; reg=np.eye(A.shape[1])*alpha; reg[0,0]=0
 return LinearModel(mu,sd,np.linalg.solve(A.T@A+reg,A.T@Y))
def fit_mlp(X,Y,epochs=200):
 mu=X.mean(0); sd=X.std(0); sd[sd<1e-12]=1; net=torch.nn.Sequential(torch.nn.Linear(X.shape[1],64),torch.nn.ReLU(),torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,3)).double()
 torch.manual_seed(3402); opt=torch.optim.Adam(net.parameters(),lr=1e-2); xx=torch.tensor((X-mu)/sd); yy=torch.tensor(Y)
 for _ in range(epochs): opt.zero_grad(); loss=((net(xx)-yy)**2).mean(); loss.backward(); opt.step()
 class M:
  def predict(self,Z): return net(torch.tensor((Z-mu)/sd)).detach().numpy()
 return M()
def rows():
 m=json.loads((ROOT/'manifests/sweep_manifest.json').read_text()); cfg=json.loads(MP.read_text()); ss={s['id']:s for s in settings(m)}
 z=np.load(ROOT/'manifests/histories_s3.npz')
 if str(z['manifest_digest'].item())!=manifest_digest(): raise RuntimeError("history/manifest digest mismatch")
 out=[]
 for i,sid in enumerate(z['setting']):
  s=ss[str(sid)]; y=np.log10([s['B_eff'],s['w'],s['B_eff']/s['w']])
  ell=m['grasp']['ell'][int(z['grasp'][i])]
  out.append((str(sid),z['proprio'][i],np.r_[z['shape'][i],ell],np.array([z['wrench'][i]]),y))
 return out,cfg,m
def cv_ridge(X,Y,S,cfg):
 folds=cfg['grouped_cv']['folds']; alphas=cfg['ridge_alpha_grid']; scores=[]; by_target=[]
 for a in alphas:
  ft=[]
  for ids in folds.values():
   va=np.isin(S,ids); tr=~va; pred=fit_ridge(X[tr],Y[tr],a).predict(X[va]); ft.append([rmse(Y[va,j],pred[:,j]) for j in range(3)])
  by_target.append(np.mean(ft,axis=0).tolist()); scores.append(float(np.mean(ft)))
 k=int(np.argmin(scores)); a=alphas[k]; model=fit_ridge(X,Y,a)
 return model,a,scores,by_target[k]

def run(expected_digest=None):
 live=digest(MP)
 if expected_digest and live!=expected_digest: raise RuntimeError(f'manifest digest mismatch: {live} != {expected_digest}')
 data,cfg,m=rows(); train=set(cfg['splits']['step3_train']); tr=[r for r in data if r[0] in train]; te=[r for r in data if r[0] not in train]
 channels={'proprioception-only':lambda r:r[1],'+shape':lambda r:np.r_[r[1],r[2],r[3]*0+0][:24],'+wrench':lambda r:np.r_[r[1],r[2],r[3]]}
 # +shape expression excludes wrench while retaining fixed dimensionality.
 result={}
 for name,fn in channels.items():
  X=np.array([fn(r) for r in tr]); Y=np.array([r[4] for r in tr]); S=np.array([r[0] for r in tr]); Xt=np.array([fn(r) for r in te]); Yt=np.array([r[4] for r in te])
  ridge,a,cvs,cvt=cv_ridge(X,Y,S,cfg); pr=ridge.predict(Xt)
  mlp=fit_mlp(X,Y); pm=mlp.predict(Xt)
  result[name]={'ridge':{'alpha':a,'train_cv_rmse_mean':min(cvs),'train_cv_rmse':dict(zip(cfg['targets'],cvt)),'test_rmse':dict(zip(cfg['targets'],[rmse(Yt[:,j],pr[:,j]) for j in range(3)]))},'mlp':{'epochs':200,'test_rmse':dict(zip(cfg['targets'],[rmse(Yt[:,j],pm[:,j]) for j in range(3)]))}}
 # paired shape guard, normalized RMS
 pair_shape=[]
 for a,b in cfg['splits']['final_test_pairs']:
  A=np.mean([r[2] for r in te if r[0]==a],0); B=np.mean([r[2] for r in te if r[0]==b],0); pair_shape.append(float(np.sqrt(np.mean((A-B)**2))/(max(np.sqrt(np.mean(A*A)),1e-12))))
 sh=result['+shape']['ridge']['test_rmse']; wr=result['+wrench']['ridge']['test_rmse']; T=cfg['targets']; mar=cfg['margins']
 positive=sh[T[2]]<=mar['tol_ratio']; guard=max(pair_shape)<=mar['tol_shape']; null=sh[T[0]]>=mar['K']*sh[T[2]] and sh[T[1]]>=mar['K']*sh[T[2]]; repair=wr[T[0]]<=mar['tol_indiv'] and wr[T[1]]<=mar['tol_indiv']
 verdict='INCONCLUSIVE' if not(positive and guard) else ('PASS' if null and repair else 'FAIL')
 # L prefixes: aggregate deterministic groups of L tuples, refit/evaluate ratio target.
 hist={}
 for L in [1,3]:
  def ag(src,fn):
   o=[]
   for sid in sorted(set(r[0] for r in src)):
    q=[r for r in src if r[0]==sid]
    for i in range(0,len(q)-L+1,L): o.append((sid,np.mean([fn(x) for x in q[i:i+L]],0),q[i][4]))
   return o
  aa=ag(tr,channels['+shape']); bb=ag(te,channels['+shape']); X=np.array([x[1] for x in aa]); y=np.array([x[2][2] for x in aa]); S=np.array([x[0] for x in aa]); mod,al,_,_=cv_ridge(X,np.c_[y,y,y],S,cfg); pred=mod.predict(np.array([x[1] for x in bb]))[:,0]; hist[str(L)]={'shape_ratio_rmse':rmse(np.array([x[2][2] for x in bb]),pred),'alpha':al,'n_test':len(bb)}
 hz=np.load(ROOT/'manifests/histories_s3.npz')
 out={'schema_version':1,'manifest_digest':live,'history_source':{'kind':str(hz['source'].item()),'rollout_count':int(hz['rollout_count']),'wall_clock_s':float(hz['wall_clock_s']),'log':'sim/logs/history_s3.log'},'folds':cfg['grouped_cv']['folds'],'models':result,'positive_control':{'passed':positive,'shape_ratio_rmse':sh[T[2]],'tol_ratio':mar['tol_ratio']},'confound_guard':{'passed':guard,'paired_relative_shape_rms':pair_shape,'tol_shape':mar['tol_shape']},'null_side':{'passed':null,'K':mar['K'],'shape_individual_to_ratio_error':[sh[T[0]]/max(sh[T[2]],1e-15),sh[T[1]]/max(sh[T[2]],1e-15)]},'repair_side':{'passed':repair,'absolute_bound':mar['tol_indiv'],'secondary_reduction':[sh[T[0]]/max(wr[T[0]],1e-15),sh[T[1]]/max(wr[T[1]],1e-15)]},'verdict':verdict,'proprioception_baseline':result['proprioception-only']['ridge']['test_rmse'],'history_length':hist,'friction_absence_note':'Genesis c5026a9 provides set_mu_s and set_mu_k, but the lift-and-clear clamp is a rigid welded attachment with no relevant frictional-slip contact. A meaningful slip probe requires new contact geometry and is therefore deferred per the frozen directive.'}
 (ROOT/'manifests/identifiability.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
 labels=cfg['targets']; x=np.arange(3); fig,ax=plt.subplots(figsize=(8,4)); width=.25
 for i,(n,v) in enumerate(result.items()): ax.bar(x+(i-1)*width,[v['ridge']['test_rmse'][t] for t in labels],width,label=n)
 ax.set_xticks(x,labels,rotation=15); ax.set_ylabel('test RMSE (dex)'); ax.legend(); fig.tight_layout(); fig.savefig(ROOT/'figures/identifiability_table.png',dpi=180); plt.close(fig)
 fig,ax=plt.subplots(figsize=(5,3.5)); ax.plot([1,3],[hist[str(i)]['shape_ratio_rmse'] for i in [1,3]],'o-'); ax.set(xlabel='history length L',ylabel='shape ratio RMSE (dex)',xticks=[1,3]); fig.tight_layout(); fig.savefig(ROOT/'figures/history_length.png',dpi=180); plt.close(fig)
 print(json.dumps({'manifest_digest':live,'verdict':verdict,'positive_control':positive})); return out
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--expected-digest',required=True); a=p.parse_args(); run(a.expected_digest)
