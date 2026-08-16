"""Step-3 history assembly and channel extraction.

The feature extractor is deliberately label-blind: it accepts only settled vertices,
template/drive data, and supported load.  Setting identifiers are provenance only.
"""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent

def sha256(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def manifest_digest(path=ROOT/'manifests/s34_manifest.json'):
 return sha256(path)

def supported_wrench(segment_mass,n_vertices,interval=0.01,g=9.81):
 """Production ideal wrench: supported weight divided by free length."""
 n_free=n_vertices-2; ell=n_free*interval
 return segment_mass*n_free*g/ell

def shape_summary(vertices,m=8):
 """Settled clamp-frame vertical droop at fixed free-arm index fractions."""
 v=np.asarray(vertices,float)
 if v.ndim!=2 or v.shape[1]!=3 or len(v)<3 or not np.isfinite(v).all(): raise ValueError('invalid vertices')
 q=v[2:,2]-v[:2,2].mean()
 return np.interp(np.linspace(0,len(q)-1,m),np.arange(len(q)),q)

def drive_summary(template,seed):
 """Terminal pose and frozen drive parameters; no property metadata."""
 r=np.random.default_rng(seed); dx=r.uniform(-.0035,.0035); dy=r.uniform(-.0035,.0035); dur=r.uniform(.92,1.08); arc=r.uniform(.9,1.1)
 return np.array([0.,0.,.7,template['arc'],dx,dy,dur,arc],float)

def extract_real(expected_digest,batch_size=64):
 """Replay the frozen tuples in Genesis and persist actual settled configurations."""
 if manifest_digest()!=expected_digest: raise RuntimeError("manifest digest mismatch")
 from sim.scene import build_scene,add_straight_rod,add_moving_clamp,attach_moving_clamp,vertices
 from sim.material import apply_properties
 from sim.sweep import settings,draws
 man=json.loads((ROOT/'manifests/sweep_manifest.json').read_text()); cfg=json.loads((ROOT/'manifests/s34_manifest.json').read_text())
 ss={s['id']:s for s in settings(man)}
 train=set(cfg['splits']['step3_train']); use=train|{x for p in cfg['splits']['final_test_pairs'] for x in p}; specs=[]
 for sid in sorted(use):
  seeds=[2000,2001] if sid in train else [2002]; grasps=range(4) if sid in train else range(8)
  for grasp in grasps:
   for template in range(4):
    for seed in seeds: specs.append(dict(setting=sid,grasp=grasp,template=template,seed=seed))
  if sum(x['setting']==sid for x in specs)!=32: raise RuntimeError(f"{sid}: history policy did not yield 32")
 start=time.time(); records=[]
 for grasp in sorted({x['grasp'] for x in specs}):
  group=[x for x in specs if x['grasp']==grasp]; nv=man['grasp']['n_vertices'][grasp]
  for off in range(0,len(group),batch_size):
   q=group[off:off+batch_size]; scene=build_scene(man['integrator']['dt'],man['integrator']['substeps'],man['integrator']['damping'],man['integrator']['angular_damping'])
   rod=add_straight_rod(scene,nv,man['interval'],1e7,.001,pos=(0,0,.5)); box=add_moving_clamp(scene,(0,0,.5)); scene.build(n_envs=len(q),env_spacing=(2,2))
   apply_properties(rod,np.array([ss[x['setting']]['raw_E'] for x in q]),np.array([ss[x['setting']]['mass'] for x in q])); attach_moving_clamp(rod,box); rand=[draws(x['seed']) for x in q]
   for step in range(360):
    pos=[]
    for x,r in zip(q,rand):
     t=man['templates'][x['template']]; u=min(1.,(step+1)/(360*r['dur'])); s=u*u*(3-2*u) if t['kind']=='ease' else u
     pos.append((r['dx']*(1-s)+(t['arc']*r['arc']*np.sin(np.pi*s) if t['kind']=='arc' else 0),r['dy']*(1-s),.5+.2*s))
    box.set_pos(np.asarray(pos)); scene.step()
   prev=vertices(rod)[:,-1,2]; stable=False
   for chunk in range(100):
    for _ in range(200): scene.step()
    cur=vertices(rod)[:,-1,2]; drift=float(np.max(np.abs(cur-prev))); prev=cur
    if drift<1e-3: stable=True; break
   if not stable: raise RuntimeError(f"batch failed convergence grasp={grasp} offset={off}")
   vv=vertices(rod)
   for e,x in enumerate(q):
    s=ss[x['setting']]; records.append((x['setting'],x['grasp'],x['template'],x['seed'],shape_summary(vv[e]),drive_summary(man['templates'][x['template']],x['seed']),supported_wrench(s['mass'],nv,man['interval'],man['gravity']),chunk+1,drift))
   print(json.dumps({"complete":len(records),"total":len(specs),"grasp":grasp,"batch_offset":off}),flush=True)
 out=ROOT/'manifests/histories_s3.npz'; fields=list(zip(*records))
 np.savez_compressed(out,setting=np.array(fields[0]),grasp=np.array(fields[1]),template=np.array(fields[2]),seed=np.array(fields[3]),shape=np.array(fields[4]),proprio=np.array(fields[5]),wrench=np.array(fields[6]),settle_chunks=np.array(fields[7]),last_tip_drift=np.array(fields[8]),manifest_digest=expected_digest,wall_clock_s=time.time()-start,rollout_count=len(records),source="actual Genesis rod.get_vertices_pos()")
 print(json.dumps({"output":str(out),"rollout_count":len(records),"wall_clock_s":time.time()-start,"manifest_digest":expected_digest}),flush=True)

def provenance(paths): return {str(p):sha256(p) for p in map(Path,paths)}

def write_provenance(path,inputs,extra=None):
 out={'schema_version':1,'inputs':provenance(inputs)}; out.update(extra or {})
 Path(path).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); return out

if __name__=="__main__":
 p=argparse.ArgumentParser(); p.add_argument("--expected-digest",required=True); p.add_argument("--batch-size",type=int,default=64); a=p.parse_args(); extract_real(a.expected_digest,a.batch_size)
