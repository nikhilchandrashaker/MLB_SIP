import pandas as pd, numpy as np, json
pc=pd.read_pickle('/home/claude/work/pc.pkl').sort_values('opps',ascending=False).reset_index(drop=True); ps=pd.read_pickle('/home/claude/work/ps.pkl')
S=json.load(open('/home/claude/work/stats.json')); h=pd.read_pickle('/home/claude/work/hi3.pkl'); o=h[h.opportunity==1]
idx={p:i for i,p in enumerate(pc.pitcher_id)}
P=[[r.pitcher_id,r.pitcher,int(r.first_season),int(r.last_season),int(r.opps),int(r.shutdowns),round(r.xSIP,2),round(r.SIP_regressed,2),int(r.base_n),None if np.isnan(r.base_pct) else round(r.base_pct,2)] for r in pc.itertuples()]
Srows=[[idx[r.pitcher_id],int(r.season),r.team,int(r.opps),int(r.shutdowns),None if np.isnan(r.base_pct) else round(r.base_pct,1),round(r.xSIP,2)] for r in ps.itertuples()]
# reorder: s row = [pidx,season,team,opps,shut,base,x]
L=pd.read_csv('/home/claude/work/league_season.csv'); Ls=[[int(r.season),round(r.SIP,2),round(r.base,2)] for r in L.itertuples()]
pr=o.groupby(o.prev_runs.clip(upper=5).astype(int)).success.mean()*100; PRr=[[('5+' if k==5 else str(k)),round(v,2)] for k,v in pr.items()]
data=dict(p=P,s=Srows,l=Ls,pr=PRr,st=0,k=dict(sip=100*S['sip_league'],base=100*S['base_league'],opps=S['n_opps'],games=S['n_games'],big=S['bigleads_within_pitcher']['diff_pts']))
t=open('/home/claude/proj/dashboard/template.html').read().replace('/*DATA*/',json.dumps(data,separators=(',',':'),ensure_ascii=False))
open('/home/claude/proj/SIP_dashboard.html','w',encoding='utf-8').write(t); print(len(t)/1e6,'MB')
