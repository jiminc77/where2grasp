"""End-to-end no-render throughput benchmark (Genesis has no reliable rod state reset API)."""
from __future__ import annotations
import json,time
from pathlib import Path
from sim.tasks.lift_and_clear import run_rollout
OUT=Path(__file__).resolve().parent/'manifests/throughput.json'
def main():
 rows=[]
 for n in (1,4,8,16,32,64):
  t=time.perf_counter(); ok=True; err=None
  try: run_rollout(1e7,.001,n_vertices=20,h=.006,n_envs=n)
  except Exception as e: ok=False; err=repr(e)
  dt=time.perf_counter()-t; rows.append({'n_envs':n,'seconds':dt,'rollouts_per_s':n/dt if ok else 0,'stable':ok,'error':err})
 good=[r for r in rows if r['stable']]; chosen=max(good,key=lambda r:r['rollouts_per_s']); projected=5865/chosen['rollouts_per_s']
 d={'method':'cold rebuild per measured batch; warm reset infeasible because Genesis ROD exposes no complete state reset/re-attachment contract','rows':rows,'chosen_n_envs':chosen['n_envs'],'projected_seconds_5865':projected,'under_one_day':projected<86400}
 OUT.write_text(json.dumps(d,indent=2)+'\n'); print(json.dumps(d,indent=2))
if __name__=='__main__': main()
