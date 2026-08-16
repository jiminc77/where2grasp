"""Bounded Step-2 pilot and immutable sweep pre-registration."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
CAL=ROOT/'manifests/calibration.json'; OUT=ROOT/'manifests/sweep_manifest.json'
G=9.81; INTERVAL=.01

def main():
    cal=json.loads(CAL.read_text())
    mats=cal['materials']; raw=[float(x['raw_E']) for x in mats]; be=[float(x['fitted_B_eff']) for x in mats]
    # Bounded analytical pilot (four candidate h values); its only purpose is experimental design.
    masses=np.geomspace(.0002,.002,4); hs=[.003,.0045,.006,.009]
    ell=np.linspace(.12,.54,15)
    scores=[]
    for h in hs:
        predicted=np.array([(8*b*h/(m*G/INTERVAL))**.25 for b in (be[0],be[-1]) for m in (masses[0],masses[-1])])
        scores.append(int(np.sum((predicted>=ell[0])&(predicted<=ell[-1]))))
    h=float(hs[int(np.argmax(scores))])
    templates=[
      {'index':0,'kind':'linear','drive_steps':360,'arc':0.0},
      {'index':1,'kind':'ease','drive_steps':360,'arc':0.0},
      {'index':2,'kind':'arc','drive_steps':360,'arc':0.025},
      {'index':3,'kind':'arc','drive_steps':360,'arc':-0.025}]
    # Three extra settings on common-scale lines whose bases are independent-grid settings.
    ratio=[]
    for i,(bi,wi,c) in enumerate([(1,1,2.0),(3,2,.5),(2,1,1.5)]):
        E,m=raw[bi],float(masses[wi])
        ratio.append({'pair_id':i,'reference_setting':f'B{bi}_w{wi}',
                      'base':{'raw_E':E,'segment_mass':m},
                      'scaled':{'raw_E':E*c,'segment_mass':m*c},'c':c})
    manifest={
      'schema_version':1,'frozen':True,'pilot':{'method':'Euler-Bernoulli design-only 2x2 screen','iterations':len(hs),'max_iterations':4,'candidate_h':hs,'on_grid_scores':scores},
      'integrator':{'dt':2.5e-4,'substeps':20,'damping':40.0,'angular_damping':20.0},
      'interval':INTERVAL,'gravity':G,'raw_E':raw,'B_eff':be,'segment_masses':masses.tolist(),'w':(masses*G/INTERVAL).tolist(),
      'h':h,'tau':.5,'grasp':{'orientation':'larger index means longer free arm','ell':ell.tolist(),'n_vertices':(np.rint(ell/INTERVAL).astype(int)+2).tolist(),'minimum_segments':12},
      'terminal_geometry':{'clamp_position':[0,0,.70],'clamp_orientation':[1,0,0,0],'start_z':.50},'templates':templates,
      'stochastic_distribution':{'initial_rod_pose_translation_xy_m':[-.002,.002],'clamp_start_translation_xy_m':[-.0015,.0015],'motion_duration_multiplier':[.92,1.08],'arc_multiplier':[.9,1.1],'distribution':'independent bounded uniform; terminal pose exact'},
      'seed_banks':{'pilot':list(range(1000,1012)),'selection':[2000,2001,2002],'evaluation':[3000,3001,3002,3003,3004]},
      'selection_rule':'CRN; maximize 3-draw mean success; ties lowest template index','evaluation_rule':'selected template only; disjoint 5-draw mean success and mean J; no reselection',
      'boundary_rule':'all tau crossings by linear interpolation; boundary=max crossing; uncertainty=half grid step; all-success/all-failure unresolved',
      'decision_rule':{'B_sign':'increase','w_sign':'decrease','resolution':'one grid step','adjacent':'majority correct-signed-or-tied','ratio_tolerance':'one grid step','precedence':'NO-GO if any FAIL; GO iff all PASS; otherwise inconclusive'},
      'ratio_pairs':ratio,'rollout_budget':5865}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'manifest':str(OUT),'h':h,'masses':masses.tolist(),'ell':[ell[0],ell[-1]],'pilot_scores':scores},indent=2))
if __name__=='__main__': main()
