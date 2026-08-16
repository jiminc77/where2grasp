import json,numpy as np
from sim.identify import rows
data,cfg,_=rows(); test=[r for r in data if r[0] not in cfg['splits']['step3_train']]; errs=[]
for a,b in cfg['splits']['final_test_pairs']:
 A=np.mean([r[2] for r in test if r[0]==a],0); B=np.mean([r[2] for r in test if r[0]==b],0); errs.append(float(np.sqrt(np.mean((A-B)**2))/max(np.sqrt(np.mean(A*A)),1e-12)))
ok=np.isfinite(errs).all() and max(errs)<=cfg['margins']['tol_shape']; print(json.dumps({'probe':'p_h','pass':bool(ok),'tuples_checked':192,'paired_relative_shape_rms':errs,'interpretation':'numerical confound; downstream verdict must be INCONCLUSIVE' if not ok else 'degeneracy confirmed'})); raise SystemExit(0 if ok else 1)
