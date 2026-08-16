"""Frozen batched Genesis grasp-landscape sweep."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
from sim.scene import build_scene,add_straight_rod,add_moving_clamp,attach_moving_clamp,vertices
from sim.material import apply_properties
ROOT=Path(__file__).resolve().parent; MAN=ROOT/'manifests/sweep_manifest.json'

def settings(m):
 out=[]
 for i,(E,B) in enumerate(zip(m['raw_E'],m['B_eff'])):
  for j,(mass,w) in enumerate(zip(m['segment_masses'],m['w'])): out.append(dict(id=f'B{i}_w{j}',kind='independent',bi=i,wi=j,raw_E=E,B_eff=B,mass=mass,w=w))
 for p in m['ratio_pairs']:
  q=p['scaled']; E=q['raw_E']; mass=q['segment_mass']; B=np.interp(E,m['raw_E'],m['B_eff']); out.append(dict(id=f'R{p["pair_id"]}',kind='ratio',pair_id=p['pair_id'],raw_E=E,B_eff=float(B),mass=mass,w=mass*m['gravity']/m['interval']))
 return out

def draws(seed):
 r=np.random.default_rng(seed); return dict(dx=r.uniform(-.0035,.0035),dy=r.uniform(-.0035,.0035),dur=r.uniform(.92,1.08),arc=r.uniform(.9,1.1))

def run_batch(m, envspec):
 n=len(envspec); scene=build_scene(m['integrator']['dt'],m['integrator']['substeps'],m['integrator']['damping'],m['integrator']['angular_damping'])
 rods=[]; boxes=[]
 for k,nv in enumerate(m['grasp']['n_vertices']):
  y=.12*k; rods.append(add_straight_rod(scene,nv,m['interval'],1e7,.001,pos=(0,y,.5))); boxes.append(add_moving_clamp(scene,(0,y,.5)))
 scene.build(n_envs=n,env_spacing=(2,2))
 Es=np.array([x['setting']['raw_E'] for x in envspec]); masses=np.array([x['setting']['mass'] for x in envspec])
 for rod,box in zip(rods,boxes): apply_properties(rod,Es,masses); attach_moving_clamp(rod,box)
 rand=[draws(x['seed']) for x in envspec]
 for step in range(360):
  pos=[]
  for x,q in zip(envspec,rand):
   tmpl=m['templates'][x['template']]; u=min(1.,(step+1)/(360*q['dur'])); s=u*u*(3-2*u) if tmpl['kind']=='ease' else u
   xx=q['dx']*(1-s)+(tmpl['arc']*q['arc']*np.sin(np.pi*s) if tmpl['kind']=='arc' else 0); yy=q['dy']*(1-s)
   pos.append((xx,yy,.5+.2*s))
  P=np.asarray(pos)
  for k,box in enumerate(boxes): Q=P.copy(); Q[:,1]+=.12*k; box.set_pos(Q)
  scene.step()
 # convergence based on tip drift over 200 steps
 prev=[vertices(r)[:,-1,2] for r in rods]; stable=False
 for chunk in range(100):
  for _ in range(200): scene.step()
  cur=[vertices(r)[:,-1,2] for r in rods]; drift=max(float(np.max(np.abs(a-b))) for a,b in zip(cur,prev)); prev=cur
  if drift<1e-3: stable=True; break
 rows=[]
 for gi,(rod,ell) in enumerate(zip(rods,m['grasp']['ell'])):
  v=vertices(rod); clamp=v[:,:2,2].mean(1); delta=clamp-v[:,-1,2]
  for e,x in enumerate(envspec): rows.append(dict(setting=x['setting']['id'],grasp=gi,ell=ell,template=x['template'],bank=x['bank'],seed=x['seed'],J=float(m['h']-delta[e]),success=bool(delta[e]<=m['h']),converged=stable,last_chunk_tip_drift=float(drift)))
 return rows

def main(batch_size=64):
 m=json.loads(MAN.read_text()); ss=settings(m); start=time.time(); rows=[]
 selection=[dict(setting=s,template=t,seed=seed,bank='selection') for s in ss for t in range(4) for seed in m['seed_banks']['selection']]
 for i in range(0,len(selection),batch_size): rows+=run_batch(m,selection[i:i+batch_size]); print('selection',min(i+batch_size,len(selection)),len(selection),flush=True)
 winners={}
 for s in ss:
  for g in range(15):
   rates=[np.mean([r['success'] for r in rows if r['setting']==s['id'] and r['grasp']==g and r['template']==t]) for t in range(4)]
   winners[s['id'],g]=int(np.argmax(rates))
 # Evaluation batches must share one template across all grasp rods; duplicate env by (setting,template,seed), then retain only grasps selecting it.
 evaluation=[dict(setting=s,template=t,seed=seed,bank='evaluation') for s in ss for t in range(4) for seed in m['seed_banks']['evaluation'] if any(winners[s['id'],g]==t for g in range(15))]
 evalrows=[]
 for i in range(0,len(evaluation),batch_size): evalrows+=run_batch(m,evaluation[i:i+batch_size]); print('evaluation',min(i+batch_size,len(evaluation)),len(evaluation),flush=True)
 evalrows=[r for r in evalrows if winners[r['setting'],r['grasp']]==r['template']]; rows+=evalrows
 for r in rows: r['selected_template']=bool(winners[r['setting'],r['grasp']]==r['template'])
 out=ROOT/'manifests/sweep_results.npz'; keys=['setting','grasp','ell','template','bank','seed','J','success','selected_template','converged','last_chunk_tip_drift']
 np.savez_compressed(out,**{k:np.array([r[k] for r in rows]) for k in keys},wall_clock_s=time.time()-start)
 print(json.dumps({'physical_rollouts':len(rows),'required_rollouts':5865,'wall_clock_s':time.time()-start,'nonconverged_fraction':float(np.mean([not r['converged'] for r in rows])),'output':str(out)}))
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--batch-size',type=int,default=64); a=p.parse_args(); main(a.batch_size)
