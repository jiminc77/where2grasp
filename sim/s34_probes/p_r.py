import json,numpy as np,torch
r=np.random.default_rng(7); X=r.normal(size=(256,3)); y=X@np.array([.7,-1.2,.4]); A=np.c_[np.ones(200),X[:200]]; w=np.linalg.solve(A.T@A+np.eye(4)*1e-3,A.T@y[:200]); pr=np.c_[np.ones(56),X[200:]]@w
net=torch.nn.Sequential(torch.nn.Linear(3,64),torch.nn.ReLU(),torch.nn.Linear(64,1)).double(); opt=torch.optim.Adam(net.parameters(),lr=.02); xx=torch.tensor(X[:200]); yy=torch.tensor(y[:200,None])
for _ in range(500): opt.zero_grad(); loss=((net(xx)-yy)**2).mean(); loss.backward(); opt.step()
pm=net(torch.tensor(X[200:])).detach().numpy()[:,0]
def r2(p): return 1-float(np.sum((y[200:]-p)**2)/np.sum((y[200:]-y[200:].mean())**2))
scores=[r2(pr),r2(pm)]; ok=min(scores)>.9; print(json.dumps({'probe':'p_r','pass':ok,'heldout_r2':scores})); assert ok
