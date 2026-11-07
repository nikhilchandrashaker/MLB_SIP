import pandas as pd, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
h=pd.read_pickle('/home/claude/work/halfinnings.pkl')
# final results
tot=h.groupby(['game_id','bat_team']).runs.sum().unstack()
gt=h.groupby('game_id').apply(lambda d: d.groupby('bat_team').runs.sum().to_dict())
res={}
for gid,d in gt.items():
    (t1,r1),(t2,r2)=list(d.items())[:2]
    res[gid]={t1:int(r1>r2) if r1!=r2 else np.nan,t2:int(r2>r1) if r1!=r2 else np.nan}
h['bat_win']=[res[g][b] for g,b in zip(h.game_id,h.bat_team)]
h['fld_win']=1-h['bat_win']
h['bat_lead0']=-h['lead_before']
h['inn_c']=h['inning'].clip(upper=12)
m=h.dropna(subset=['bat_win'])
X=m[['bat_lead0','inn_c','bat_home']].values
clf=HistGradientBoostingClassifier(max_iter=150,learning_rate=0.1,random_state=0).fit(X,m['bat_win'])
h['wp_bat']=clf.predict_proba(h[['bat_lead0','inn_c','bat_home']].values)[:,1]
h['wp_pit_before']=1-h['wp_bat']
nxt=h.groupby('game_id')['wp_bat'].shift(-1)
h['wp_pit_after']=np.where(nxt.isna(),h['fld_win'],nxt)
h['wpa']=h['wp_pit_after']-h['wp_pit_before']
h.to_pickle('/home/claude/work/hi2.pkl')
o=h[h.opportunity==1]
print(o.groupby('scoreless').wpa.mean(), o.groupby(['inn_c','scoreless']).wpa.mean().unstack().round(3).head(10))
