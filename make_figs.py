import pandas as pd, numpy as np, json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
F='/home/claude/proj/figures/'
h=pd.read_pickle('/home/claude/work/hi3.pkl'); pc=pd.read_pickle('/home/claude/work/pc.pkl'); ps=pd.read_pickle('/home/claude/work/ps.pkl')
S=json.load(open('/home/claude/work/stats.json')); M=pd.read_csv('/home/claude/work/models.csv')
o=h[h.opportunity==1].copy(); nb=h[(h.opportunity==0)&(h.hidx>0)].copy()
NAVY='#1f3a5f'; RED='#c8452c'; GRAY='#8a8f98'; GOLD='#d9a441'; TEAL='#2a8c82'
plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.25,'axes.titleweight':'bold','axes.titlesize':13,'figure.dpi':100,'savefig.dpi':170,'savefig.bbox':'tight'})
def wil(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; w=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c-w,c+w
def save(n): plt.savefig(F+n); plt.close()
# 1 diagram
fig,ax=plt.subplots(figsize=(11,4.6)); ax.axis('off'); ax.set_xlim(0,11); ax.set_ylim(0,4.6)
def box(x,y,w,hh,t,fc,tc='white',fs=10.5):
    ax.add_patch(FancyBboxPatch((x,y),w,hh,boxstyle='round,pad=0.05,rounding_size=0.12',fc=fc,ec='none')); ax.text(x+w/2,y+hh/2,t,ha='center',va='center',color=tc,fontsize=fs,fontweight='bold',wrap=True)
def arr(x1,y1,x2,y2,c='#444'): ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=16,color=c,lw=1.6))
box(0.2,1.8,2.5,1.2,'Half-inning N\nTeam A bats\nscores 1+ runs',NAVY)
box(3.6,1.8,2.7,1.2,'Half-inning N+1\nTeam A pitcher takes\nthe mound',GRAY)
ax.text(4.95,3.35,'SHUTDOWN OPPORTUNITY',ha='center',color=GOLD,fontweight='bold',fontsize=11)
box(7.4,2.9,3.2,1.1,'Success: scoreless inning\n(shutdown)',TEAL)
box(7.4,0.6,3.2,1.1,'Failure: 1+ runs allowed\n(gives the cushion back)',RED)
arr(2.8,2.4,3.5,2.4); arr(6.4,2.5,7.3,3.4); arr(6.4,2.3,7.3,1.2)
ax.text(3.4,0.95,r'SIP = shutdowns / opportunities $\times$ 100',ha='center',fontsize=13,fontweight='bold',color=NAVY)
ax.text(3.4,0.45,'Credit goes to the pitcher who starts the half-inning',ha='center',fontsize=9.5,color=GRAY)
save('fig01_sip_definition_diagram.png')
# 2 league by season
ls=pd.read_csv('/home/claude/work/league_season.csv')
fig,ax=plt.subplots(figsize=(10,4.8)); ax.plot(ls.season,ls.SIP,'-o',color=NAVY,lw=2.2,label='SIP (after own team scores)'); ax.plot(ls.season,ls.base,'-s',color=RED,lw=2.2,label='Scoreless % in all other half-innings')
ax.set_title('League SIP tracks the ordinary scoreless-inning rate'); ax.set_ylabel('Percent scoreless'); ax.set_xlabel('Season'); ax.legend(frameon=False); ax.set_ylim(67,76); save('fig02_league_sip_vs_baseline_by_season.png')
# 3 prior runs
o['pr']=o.prev_runs.clip(upper=5).astype(int); g=o.groupby('pr').success.agg(['sum','count']); lo,hi=zip(*[wil(a,b) for a,b in zip(g['sum'],g['count'])]); p=g['sum']/g['count']
fig,ax=plt.subplots(figsize=(8.5,5)); ax.errorbar(g.index,100*p,yerr=[100*(p-np.array(lo)),100*(np.array(hi)-p)],fmt='o-',color=NAVY,capsize=5,lw=2,ms=8)
ax.axhline(100*S['base_league'],color=RED,ls='--',label='Baseline (all other half-innings)'); ax.set_xticks([1,2,3,4,5]); ax.set_xticklabels(['1','2','3','4','5+'])
ax.set_xlabel('Runs scored by own team in the previous half-inning'); ax.set_ylabel('SIP (%)'); ax.set_ylim(68,76); ax.set_title('A bigger inning does not buy a better (or worse) shutdown'); ax.legend(frameon=False); save('fig03_sip_by_runs_scored_prior_half.png')
# 4 by inning
a=o.groupby(o.inning.clip(upper=10)).success.mean()*100; b=nb.groupby(nb.inning.clip(upper=10)).scoreless.mean()*100
fig,ax=plt.subplots(figsize=(9,4.8)); w=.38; x=np.arange(1,11)
ax.bar(x-w/2,a.values,w,color=NAVY,label='SIP'); ax.bar(x+w/2,b.values,w,color=RED,alpha=.85,label='Other half-innings'); ax.set_ylim(60,78); ax.set_xticks(x); ax.set_xticklabels([str(i) for i in range(1,10)]+['10+'])
ax.set_xlabel('Inning'); ax.set_ylabel('Scoreless %'); ax.set_title('SIP by inning vs the baseline'); ax.legend(frameon=False,ncol=2); save('fig04_sip_by_inning.png')
# 5 forest
lab={'opp':'Shutdown opportunity (vs other innings)','home_p':'Home pitcher','starter':'Pitcher is game starter','pr2':'Prior half: 2 runs (vs 1)','pr3':'Prior half: 3 runs','pr4':'Prior half: 4 runs','pr5':'Prior half: 5+ runs','long_wait':'Long wait (8+ batters faced by own team)','lead_le0':'Lead entering: tied/trailing (vs 1-2)','lead34':'Lead entering: 3-4','lead56':'Lead entering: 5-6','lead7p':'Lead entering: 7+'}
sel=M[M.term.isin(lab)&~((M.model.str.startswith('M1'))&(M.term!='opp'))].copy(); sel=sel[~((sel.model.str.startswith('M2'))&sel.term.isin(['home_p']))&~((sel.model.str.startswith('M3'))&sel.term.isin(['home_p']))]
order=['opp','pr2','pr3','pr4','pr5','long_wait','starter','lead_le0','lead34','lead56','lead7p']; sel=sel.drop_duplicates('term'); sel['k']=sel.term.map({t:i for i,t in enumerate(order)}); sel=sel.sort_values('k',ascending=False)
fig,ax=plt.subplots(figsize=(9.5,5.8)); y=np.arange(len(sel)); cols=[RED if (r.lo>1 or r.hi<1) else NAVY for r in sel.itertuples()]
ax.hlines(y,sel.lo,sel.hi,color=cols,lw=2.5); ax.scatter(sel.odds_ratio,y,color=cols,zorder=3,s=45); ax.axvline(1,color='k',lw=1); ax.set_yticks(y); ax.set_yticklabels([lab[t] for t in sel.term]); ax.set_xlabel('Odds ratio of allowing a run (95% CI, game-clustered)'); ax.set_title('Adjusted effects on giving up a run'); ax.grid(axis='y',alpha=0); save('fig05_adjusted_odds_ratios.png')
# 6 lead entering
o['ld']=o.lead_before.clip(-2,8); g=o.groupby('ld').success.agg(['sum','count']); p=g['sum']/g['count']; lo,hi=zip(*[wil(a,b) for a,b in zip(g['sum'],g['count'])])
fig,ax=plt.subplots(figsize=(9,4.8)); ax.errorbar(g.index,100*p,yerr=[100*(p-np.array(lo)),100*(np.array(hi)-p)],fmt='o-',color=NAVY,capsize=4,lw=2); ax.axhline(100*S['base_league'],color=RED,ls='--',label='Baseline')
ax.set_xticks(range(-2,9)); ax.set_xticklabels(['-2 or worse']+[str(i) for i in range(-1,8)]+['8+']); ax.set_xlabel("Pitching team's lead entering the half-inning"); ax.set_ylabel('SIP (%)'); ax.set_title('Raw SIP by lead size (bigger leads reflect better teams too)'); ax.legend(frameon=False); save('fig06_sip_by_lead_entering.png')
# 7 pitcher variance vs binomial
q=pc[pc.opps>=100].copy(); rng=np.random.default_rng(7); sims=[]
for _ in range(200): sims.append(100*rng.binomial(q.opps.values,(q.xSIP/100).values)/q.opps.values-q.xSIP.values)
sims=np.concatenate(sims); obs=(q.SIP-q.xSIP).values
fig,ax=plt.subplots(figsize=(9,4.8)); bins=np.linspace(-14,14,48); ax.hist(sims,bins=bins,density=True,color=GRAY,alpha=.55,label=f'Pure luck (binomial), SD {sims.std():.2f}'); ax.hist(obs,bins=bins,density=True,histtype='step',color=NAVY,lw=2.4,label=f'Observed, SD {obs.std():.2f}')
ax.set_xlabel('Career SIP minus context-expected SIP (points), pitchers with 100+ opportunities'); ax.set_ylabel('Density'); ax.set_title('Pitchers spread out only a little more than luck predicts'); ax.legend(frameon=False); save('fig07_pitcher_spread_vs_luck.png')
# 8 split half
oo=o.sort_values(['pitcher_id','game_id','hidx']); oo['k']=oo.groupby('pitcher_id').cumcount()%2; nn=nb.sort_values(['pitcher_id','game_id','hidx']); nn['k']=nn.groupby('pitcher_id').cumcount()%2
A=oo.groupby(['pitcher_id','k']).agg(s=('success','mean'),n=('success','size'),x=('x_scoreless','mean')).unstack(); B=nn.groupby(['pitcher_id','k']).agg(b=('scoreless','mean'),m=('scoreless','size')).unstack(); Z=A.join(B,how='inner').dropna(); Z=Z[(Z[('n',0)]>=30)&(Z[('n',1)]>=30)&(Z[('m',0)]>=30)&(Z[('m',1)]>=30)]
panels=[('SIP',100*Z[('s',0)],100*Z[('s',1)]),('Baseline scoreless %',100*Z[('b',0)],100*Z[('b',1)]),('Gap: SIP minus own baseline',100*(Z[('s',0)]-Z[('b',0)]),100*(Z[('s',1)]-Z[('b',1)])),('SIP+ (SIP minus context-expected)',100*(Z[('s',0)]-Z[('x',0)]),100*(Z[('s',1)]-Z[('x',1)]))]
fig,axs=plt.subplots(2,2,figsize=(10,9))
for ax,(t,u,v) in zip(axs.ravel(),panels):
    r=np.corrcoef(u,v)[0,1]; ax.scatter(u,v,s=9,alpha=.35,color=NAVY); lim=[min(u.min(),v.min()),max(u.max(),v.max())]; ax.plot(lim,lim,color=GRAY,lw=1,ls=':'); m,c=np.polyfit(u,v,1); xs=np.linspace(*lim,50); ax.plot(xs,m*xs+c,color=RED,lw=2); ax.set_title(f'{t}\nr = {r:.2f}',fontsize=11); ax.set_xlabel('Odd-numbered innings'); ax.set_ylabel('Even-numbered innings')
fig.suptitle(f'Split-half reliability ({len(Z):,} pitchers, 30+ innings in each half)',fontweight='bold',y=1.0); plt.tight_layout(); save('fig08_split_half_reliability.png')
# 9 pitcher SIP vs baseline
q=pc[(pc.opps>=60)&(pc.base_n>=60)]; fig,ax=plt.subplots(figsize=(7.5,6)); ax.scatter(q.base_pct,q.SIP,s=q.opps/6+6,alpha=.4,color=NAVY); m,c=np.polyfit(q.base_pct,q.SIP,1); xs=np.linspace(q.base_pct.min(),q.base_pct.max(),50); ax.plot(xs,m*xs+c,color=RED,lw=2.2)
ax.set_xlabel('Scoreless % in all other half-innings'); ax.set_ylabel('SIP (%)'); ax.set_title(f'Pitcher SIP mostly mirrors general run prevention (r = {np.corrcoef(q.base_pct,q.SIP)[0,1]:.2f})',fontsize=11.5); save('fig09_sip_vs_baseline_scatter.png')
# 10 WPA
g=o.groupby([o.inning.clip(upper=9),'scoreless']).wpa.mean().unstack(); 
fig,ax=plt.subplots(figsize=(9,4.8)); x=np.arange(1,10); ax.bar(x-.2,100*g[1],.4,color=TEAL,label='Shutdown (scoreless)'); ax.bar(x+.2,100*g[0],.4,color=RED,label='Failure (1+ runs)'); ax.axhline(0,color='k',lw=1)
ax.set_xticks(x); ax.set_xticklabels([str(i) for i in range(1,9)]+['9+']); ax.set_xlabel('Inning'); ax.set_ylabel('Average win-probability change for pitching team (points)'); ax.set_title('What a shutdown is worth: win probability added'); ax.legend(frameon=False); save('fig10_win_probability_swing.png')
# 11 leaderboard
q=pc[pc.opps>=200].copy(); q['d']=q.SIP_regressed-q.xSIP; t=pd.concat([q.nlargest(10,'SIP_regressed'),q.nsmallest(10,'SIP_regressed')]).sort_values('SIP_regressed')
fig,ax=plt.subplots(figsize=(9,7)); cols=[RED]*10+[TEAL]*10; ax.barh(t.pitcher+' ('+t.first_season.astype(str)+'-'+t.last_season.astype(str).str[-2:]+')',t.SIP_regressed,color=cols); ax.set_xlim(66,76); ax.axvline(100*S['sip_league'],color='k',ls='--',lw=1); ax.set_xlabel('Regressed SIP (%)'); ax.set_title('Career regressed SIP extremes (200+ opportunities)'); ax.grid(axis='y',alpha=0); save('fig11_regressed_sip_extremes.png')
# 12 team season
tm=pd.read_csv('/home/claude/work/team_season_sip.csv'); fig,ax=plt.subplots(figsize=(7.5,6)); ax.scatter(tm.base_pct,tm.SIP,s=18,alpha=.6,color=NAVY); m,c=np.polyfit(tm.base_pct,tm.SIP,1); xs=np.linspace(tm.base_pct.min(),tm.base_pct.max(),50); ax.plot(xs,m*xs+c,color=RED,lw=2.2)
ax.set_xlabel('Team scoreless % in other half-innings'); ax.set_ylabel('Team SIP (%)'); ax.set_title(f'Team-seasons: SIP vs baseline (r = {np.corrcoef(tm.base_pct,tm.SIP)[0,1]:.2f})'); save('fig12_team_season_sip.png')
print('done')
