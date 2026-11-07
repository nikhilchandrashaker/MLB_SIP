import pandas as pd, numpy as np, glob, os, re
W='/home/claude/work'; R='/home/claude/rs/seasons'
frames=[]
for y in range(2000,2026):
    d=pd.read_csv(f'{W}/e{y}.csv'); d['season']=y; frames.append(d)
e=pd.concat(frames,ignore_index=True)
e.columns=[c.lower() for c in e.columns]
for c in ['bat_dest_id','run1_dest_id','run2_dest_id','run3_dest_id']:
    e[c]=e[c].fillna(0)
e['runs']=sum((e[c]>=4).astype(int) for c in ['bat_dest_id','run1_dest_id','run2_dest_id','run3_dest_id'])
bat_ev={2,3,14,15,16,17,18,19,20,21,22,23}
e['pa']=e['event_cd'].isin(bat_ev).astype(int)
e['hidx']=2*(e['inn_ct']-1)+e['bat_home_id']
g=e.groupby(['game_id','hidx'],sort=False)
h=g.agg(season=('season','first'),inning=('inn_ct','first'),bat_home=('bat_home_id','first'),
        home_team=('home_team_id','first'),bat_team=('bat_team_id','first'),fld_team=('fld_team_id','first'),
        pitcher_id=('pit_id','first'),last_pitcher_id=('pit_id','last'),n_pitchers=('pit_id','nunique'),
        runs=('runs','sum'),pa=('pa','sum'),away0=('away_score_ct','first'),home0=('home_score_ct','first'),
        first_outs=('outs_ct','first')).reset_index()
h=h.sort_values(['game_id','hidx']).reset_index(drop=True)
# previous half-inning (opponent-batting half belongs to the other team; the pitcher's team batted in hidx-1)
h['prev_runs']=h.groupby('game_id')['runs'].shift(1)
h['prev_pa']=h.groupby('game_id')['pa'].shift(1)
h['prev_hidx']=h.groupby('game_id')['hidx'].shift(1)
h['fld_score0']=np.where(h['bat_home']==1,h['away0'],h['home0'])
h['bat_score0']=np.where(h['bat_home']==1,h['home0'],h['away0'])
h['lead_before']=h['fld_score0']-h['bat_score0']   # pitching team's lead entering the half-inning (already includes the runs it just scored)
h['opportunity']=((h['prev_runs']>=1)&(h['prev_hidx']==h['hidx']-1)).astype(int)
h['scoreless']=(h['runs']==0).astype(int)
h['success']=np.where(h['opportunity']==1,h['scoreless'],np.nan)
h['date']=h['game_id'].str[3:11]
h['home_pitching']=1-h['bat_home']
# game starter flag: first pitcher for the fielding team in the game
first=h.groupby(['game_id','fld_team'])['pitcher_id'].transform('first')
h['pitcher_is_starter']=(h['pitcher_id']==first).astype(int)
h['extra_innings']=(h['inning']>9).astype(int)
h['ghost_runner']=((h['extra_innings']==1)&(h['season'].isin([2020,2021,2022,2023,2024,2025]))).astype(int)
# names
names={}
for f in glob.glob(f'{R}/*/*.ROS'):
    for l in open(f,encoding='latin-1'):
        p=l.strip().split(',')
        if len(p)>=3: names[p[0]]=f'{p[2]} {p[1]}'
h['pitcher']=h['pitcher_id'].map(names).fillna(h['pitcher_id'])
h.to_pickle('/home/claude/work/halfinnings.pkl')
print(len(h), h.opportunity.sum(), h.loc[h.opportunity==1,'success'].mean(), h.loc[h.opportunity==0,'scoreless'].mean())
# sanity: runs per game vs known (~4.5 R/G)
gm=h.groupby(['game_id','season']).runs.sum().reset_index()
print(gm.groupby('season').runs.mean().round(2).tail(8).to_dict())
