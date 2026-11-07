import pandas as pd, numpy as np, scipy.sparse as sp, json
from sklearn.linear_model import LogisticRegression
h=pd.read_pickle('/home/claude/work/hi2.pkl')
h=h[h.hidx>=0].copy()
h['inn_b']=h.inning.clip(upper=10)
# --- context model: P(scoreless) from opponent offense-season, park, inning, home, season  (pitcher-agnostic)
def onehot(s):
    c=s.astype('category'); return sp.csr_matrix((np.ones(len(s)),(np.arange(len(s)),c.cat.codes)),shape=(len(s),len(c.cat.categories)))
mats=[onehot(h.season.astype(str)+'_'+h.bat_team), onehot(h.home_team.astype(str)+'_'+(h.season//5).astype(str)), onehot(h.inn_b.astype(str)), onehot(h.home_pitching.astype(str)), onehot(h.ghost_runner.astype(str))]
X=sp.hstack(mats).tocsr()
base=h.hidx>0
lr=LogisticRegression(C=50,max_iter=300,solver='lbfgs')
lr.fit(X[base.values],h.loc[base,'scoreless'])
h['x_scoreless']=lr.predict_proba(X)[:,1]
h.to_pickle('/home/claude/work/hi3.pkl')
print('context model fitted; mean xscoreless opp:',h.loc[h.opportunity==1,'x_scoreless'].mean())
