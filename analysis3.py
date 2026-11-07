import pandas as pd, numpy as np, json, statsmodels.api as sm, warnings
warnings.filterwarnings('ignore')
h=pd.read_pickle('/home/claude/work/hi3.pkl')
h['fail']=1-h['scoreless']
S={}
o=h[h.opportunity==1].copy(); nb=h[(h.opportunity==0)&(h.hidx>0)].copy()
S['n_halfinnings']=int(len(h)); S['n_games']=int(h.game_id.nunique()); S['n_opps']=int(len(o)); S['n_base']=int(len(nb))
S['sip_league']=float(o.success.mean()); S['base_league']=float(nb.scoreless.mean()); S['xsip_league']=float(o.x_scoreless.mean())
# ----- Models
def design(d,extra):
    X=pd.concat([d[extra].astype(float),
        pd.get_dummies(d.inn_b,prefix='inn',drop_first=True,dtype=float),
        pd.get_dummies(d.season,prefix='s',drop_first=True,dtype=float),
        pd.get_dummies(d.fld_team,prefix='f',drop_first=True,dtype=float),
        pd.get_dummies(d.bat_team,prefix='b',drop_first=True,dtype=float)],axis=1)
    return sm.add_constant(X).astype(np.float32)
import os
rows=[]
def run(name,d,extra,keep):
    X=design(d,extra); y=d.fail.values
    g=pd.factorize(d.game_id)[0]
    r=sm.Logit(y,X.values.astype(np.float64)).fit(disp=0,maxiter=50,cov_type='cluster',cov_kwds={'groups':g})
    cols=list(X.columns)
    for k in keep:
        i=cols.index(k); b=r.params[i]; se=r.bse[i]
        rows.append(dict(model=name,term=k,odds_ratio=np.exp(b),lo=np.exp(b-1.96*se),hi=np.exp(b+1.96*se),p=r.pvalues[i],n=len(d)))
if not os.path.exists('/home/claude/work/models.csv'):
    # Model 1: opportunity vs not (all half-innings except first), 400K sample
    a=h[h.hidx>0].copy()
    a['starter']=a.pitcher_is_starter; a['opp']=a.opportunity; a['home_p']=a.home_pitching
    m1=a.sample(400000,random_state=1)
    run('M1: opportunity vs other half-innings',m1,['opp','home_p','starter'],['opp','home_p','starter'])
    # Model 2: within opportunities, prior runs
    o['pr2']=(o.prev_runs==2).astype(float); o['pr3']=(o.prev_runs==3).astype(float); o['pr4']=(o.prev_runs==4).astype(float); o['pr5']=(o.prev_runs>=5).astype(float)
    o['starter']=o.pitcher_is_starter; o['home_p']=o.home_pitching; o['long_wait']=(o.prev_pa>=8).astype(float)
    run('M2: runs scored in prior half-inning (ref = 1 run)',o,['pr2','pr3','pr4','pr5','home_p','starter','long_wait'],['pr2','pr3','pr4','pr5','long_wait','starter','home_p'])
    # Model 3: lead entering the inning
    o['lead_le0']=(o.lead_before<=0).astype(float); o['lead34']=o.lead_before.between(3,4).astype(float); o['lead56']=o.lead_before.between(5,6).astype(float); o['lead7p']=(o.lead_before>=7).astype(float)
    run('M3: lead entering the inning (ref = 1-2 runs)',o,['lead_le0','lead34','lead56','lead7p','home_p','starter'],['lead_le0','lead34','lead56','lead7p'])
    pd.DataFrame(rows).to_csv('/home/claude/work/models.csv',index=False)
    print(pd.DataFrame(rows).round(3).to_string())
# ----- Pitcher tables
def agg(keys):
    ob=o.groupby(keys).agg(opps=('success','size'),shutdowns=('success','sum'),xsl=('x_scoreless','mean'))
    nbg=nb.groupby(keys).agg(base_n=('scoreless','size'),base_sl=('scoreless','sum'),xbl=('x_scoreless','mean'))
    t=ob.join(nbg,how='left').fillna({'base_n':0,'base_sl':0}).reset_index()
    t['SIP']=100*t.shutdowns/t.opps; t['xSIP']=100*t.xsl
    t['base_pct']=np.where(t.base_n>0,100*t.base_sl/t.base_n.replace(0,np.nan),np.nan); t['xbase']=100*t.xbl
    t['SIP_plus']=t.SIP-t.xSIP
    t['gap_vs_own_baseline']=t.SIP-t.base_pct
    return t
names=h.drop_duplicates('pitcher_id').set_index('pitcher_id').pitcher
team=h.groupby(['pitcher_id','season']).fld_team.agg(lambda s:s.value_counts().index[0]).rename('team').reset_index()
ps=agg(['pitcher_id','season']).merge(team,on=['pitcher_id','season']); ps['pitcher']=ps.pitcher_id.map(names)
pc=agg(['pitcher_id']); pc['pitcher']=pc.pitcher_id.map(names)
first=h.groupby('pitcher_id').season.min(); last=h.groupby('pitcher_id').season.max()
pc['first_season']=pc.pitcher_id.map(first); pc['last_season']=pc.pitcher_id.map(last)
# empirical-Bayes regression of SIP_plus (career)
q=pc[pc.opps>=50]
p=q.SIP/100; samp=(p*(1-p)/q.opps).mean()
tau2=max(((q.SIP_plus/100).var()-samp),0)
S['sip_plus_var_obs']=float((q.SIP_plus/100).var()); S['sip_plus_var_sampling']=float(samp); S['tau2_sip_plus']=float(tau2); S['tau_sip_plus_pts']=float(100*np.sqrt(tau2)); S['n_pitchers_50']=int(len(q))
def reg(t):
    pp=(t.SIP/100).clip(0.05,0.95); v=pp*(1-pp)/t.opps
    w=tau2/(tau2+v) if tau2>0 else 0*v
    return t.xSIP+w*(t.SIP-t.xSIP), w
pc['SIP_regressed'],pc['reliability_weight']=reg(pc); ps['SIP_regressed'],ps['reliability_weight']=reg(ps)
# ----- Reliability (split-half) across pitchers with >=60 opps and >=60 base
oo=o.sort_values(['pitcher_id','game_id','hidx']); oo['k']=oo.groupby('pitcher_id').cumcount()%2
nn=nb.sort_values(['pitcher_id','game_id','hidx']); nn['k']=nn.groupby('pitcher_id').cumcount()%2
A=oo.groupby(['pitcher_id','k']).agg(s=('success','mean'),n=('success','size'),xsl=('x_scoreless','mean')).unstack()
B=nn.groupby(['pitcher_id','k']).agg(b=('scoreless','mean'),m=('scoreless','size'),xbl=('x_scoreless','mean')).unstack()
Z=A.join(B,how='inner').dropna()
Z=Z[(Z[('n',0)]>=30)&(Z[('n',1)]>=30)&(Z[('m',0)]>=30)&(Z[('m',1)]>=30)]
def sb(r): return 2*r/(1+r)
res={}
r=np.corrcoef(Z[('s',0)],Z[('s',1)])[0,1]; res['SIP']=(r,sb(r))
r=np.corrcoef(Z[('b',0)],Z[('b',1)])[0,1]; res['Baseline scoreless%']=(r,sb(r))
d0=Z[('s',0)]-Z[('b',0)]; d1=Z[('s',1)]-Z[('b',1)]
r=np.corrcoef(d0,d1)[0,1]; res['Gap = SIP - own baseline']=(r,sb(r))
a0=(Z[('s',0)]-Z[('xsl',0)]); a1=(Z[('s',1)]-Z[('xsl',1)])
r=np.corrcoef(a0,a1)[0,1]; res['SIP+ (context-adjusted)']=(r,sb(r))
S['reliability']={k:{'split_half_r':float(v[0]),'spearman_brown':float(v[1])} for k,v in res.items()}
S['reliability_n']=int(len(Z))
S['r_sip_vs_base_pitcher']=float(np.corrcoef(pc[(pc.opps>=60)&(pc.base_n>=60)].SIP,pc[(pc.opps>=60)&(pc.base_n>=60)].base_pct)[0,1])
S['n_pitchers_60']=int(((pc.opps>=60)&(pc.base_n>=60)).sum())
# season-to-season
a=ps[(ps.opps>=20)&(ps.base_n>=20)].copy(); b=a.copy(); b['season']-=1
mrg=a.merge(b,on=['pitcher_id','season'],suffixes=('_t','_t1'))
S['yoy_n']=int(len(mrg)); S['yoy_sip_r']=float(np.corrcoef(mrg.SIP_t,mrg.SIP_t1)[0,1]); S['yoy_base_r']=float(np.corrcoef(mrg.base_pct_t,mrg.base_pct_t1)[0,1])
S['yoy_gap_r']=float(np.corrcoef(mrg.gap_vs_own_baseline_t,mrg.gap_vs_own_baseline_t1)[0,1])
# pitcher variance vs binomial: observed SD of SIP among >=100 opps vs expected
q=pc[pc.opps>=100]; pp=q.SIP/100
S['n_100']=int(len(q)); S['sd_sip_obs_100']=float(100*pp.std()); S['sd_sip_binom_100']=float(100*np.sqrt((0.72*0.28/q.opps).mean()))
# team-season
tm=o.groupby(['fld_team','season']).agg(opps=('success','size'),SIP=('success','mean')).reset_index()
tb=nb.groupby(['fld_team','season']).scoreless.mean().rename('base').reset_index()
tm=tm.merge(tb,on=['fld_team','season']); tm['SIP']*=100; tm['base']*=100
S['team_season_r']=float(np.corrcoef(tm.SIP,tm.base)[0,1])
tm.rename(columns={'fld_team':'team','base':'base_pct'}).to_csv('/home/claude/work/team_season_sip.csv',index=False)
# season-level league
ls=pd.DataFrame({'SIP':100*o.groupby('season').success.mean(),'base':100*nb.groupby('season').scoreless.mean(),'opps':o.groupby('season').size()}).reset_index()
ls.to_csv('/home/claude/work/league_season.csv',index=False)
# extra: mean xSIP check, top-bottom
pc.to_pickle('/home/claude/work/pc.pkl'); ps.to_pickle('/home/claude/work/ps.pkl')
json.dump(S,open('/home/claude/work/stats.json','w'),indent=1)
print(json.dumps(S,indent=1))
