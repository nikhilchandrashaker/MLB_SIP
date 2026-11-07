import pandas as pd, numpy as np, os, shutil
D='/home/claude/proj/data/'; os.makedirs(D,exist_ok=True)
h=pd.read_pickle('/home/claude/work/hi3.pkl'); pc=pd.read_pickle('/home/claude/work/pc.pkl'); ps=pd.read_pickle('/home/claude/work/ps.pkl')
h['half']=np.where(h.bat_home==1,'B','T')
cols=dict(game_id='game_id',date='date',season='season',inning='inning',half='half',fld_team='pitching_team',bat_team='batting_team',pitcher_id='pitcher_id',pitcher='pitcher',
 home_pitching='pitcher_is_home',pitcher_is_starter='pitcher_is_game_starter',n_pitchers='pitchers_used_in_half',prev_runs='own_runs_prior_half',prev_pa='own_batters_prior_half',lead_before='own_lead_entering',
 runs='runs_allowed',scoreless='scoreless',x_scoreless='expected_scoreless_prob',wpa='win_prob_added',ghost_runner='extra_inning_runner_rule',opportunity='shutdown_opportunity')
a=h[h.hidx>0][list(cols)].rename(columns=cols)
for c in ['expected_scoreless_prob','win_prob_added']: a[c]=a[c].round(4)
a['own_runs_prior_half']=a.own_runs_prior_half.astype('Int64'); a['own_batters_prior_half']=a.own_batters_prior_half.astype('Int64')
a.to_csv(D+'half_innings_all_2000_2025.csv.gz',index=False,compression='gzip')
op=a[a.shutdown_opportunity==1].drop(columns='shutdown_opportunity').rename(columns={'scoreless':'shutdown'}); op.to_csv(D+'shutdown_opportunities_2000_2025.csv',index=False)
r=lambda t,c,n=2: t.assign(**{k:t[k].round(n) for k in c})
cc=['SIP','xSIP','base_pct','xbase','SIP_plus','gap_vs_own_baseline','SIP_regressed']
for t,name,ex in [(pc,'pitcher_career_sip.csv',['first_season','last_season']),(ps,'pitcher_season_sip.csv',['season','team'])]:
    t=t.rename(columns={'opps':'opportunities','base_n':'baseline_half_innings','base_pct':'baseline_scoreless_pct','xbase':'expected_baseline_pct','xSIP':'expected_SIP','shutdowns':'shutdowns'})
    t['reliability_weight']=t.reliability_weight.round(3)
    keep=['pitcher_id','pitcher']+ex+['opportunities','shutdowns','SIP','expected_SIP','SIP_plus','SIP_regressed','reliability_weight','baseline_half_innings','baseline_scoreless_pct','expected_baseline_pct','gap_vs_own_baseline']
    t=t[keep].copy()
    for c in ['SIP','expected_SIP','SIP_plus','SIP_regressed','baseline_scoreless_pct','expected_baseline_pct','gap_vs_own_baseline']: t[c]=t[c].round(2)
    t.sort_values('opportunities',ascending=False).to_csv(D+name,index=False)
shutil.copy('/home/claude/work/team_season_sip.csv',D+'team_season_sip.csv'); shutil.copy('/home/claude/work/league_season.csv',D+'league_season_sip.csv'); shutil.copy('/home/claude/work/models.csv',D+'regression_results.csv')
dd=[('shutdown_opportunities_2000_2025.csv / half_innings_all_2000_2025.csv.gz','game_id','Retrosheet game id (home team + yyyymmdd + game number)'),
('','pitching_team / batting_team','Retrosheet team codes for the fielding and batting side of the half-inning'),
('','half','T = top, B = bottom'),
('','pitcher_id / pitcher','Retrosheet id and name of the pitcher who started the half-inning (gets the credit)'),
('','pitcher_is_home / pitcher_is_game_starter','1 if pitching team is home / pitcher is his team\'s starter that game'),
('','pitchers_used_in_half','Number of pitchers who faced a batter in the half-inning'),
('','own_runs_prior_half / own_batters_prior_half','Runs scored / plate appearances by the pitching team in the preceding half-inning'),
('','own_lead_entering','Pitching team lead (runs) entering the half-inning, including the runs it just scored'),
('','runs_allowed / scoreless / shutdown','Runs allowed in the half-inning; 1 if zero; shutdown = scoreless on an opportunity'),
('','expected_scoreless_prob','Pitcher-agnostic context model: opponent offense-season, park era, inning, home/away, extra-inning runner rule'),
('','win_prob_added','Change in pitching-team win probability from the start to the next half-inning (empirical model); for the final half-inning, to the game result'),
('','extra_inning_runner_rule','1 for extra innings in 2020-2025 (automatic runner on second)'),
('pitcher_*_sip.csv','SIP / expected_SIP / SIP_plus','SIP (%), context-expected SIP, and their difference in points'),
('','SIP_regressed / reliability_weight','Empirical-Bayes SIP pulled toward expectation; weight is the share of the observed deviation kept'),
('','baseline_scoreless_pct / gap_vs_own_baseline','Scoreless % in the pitcher\'s other half-innings; SIP minus that baseline')]
pd.DataFrame(dd,columns=['file','column','description']).to_csv(D+'data_dictionary.csv',index=False)
for f in sorted(os.listdir(D)): print(f, round(os.path.getsize(D+f)/1e6,2),'MB')
print(len(op), op.shutdown.mean().round(4))
