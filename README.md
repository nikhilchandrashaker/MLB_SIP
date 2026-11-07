# Shutdown Inning Percentage (SIP)

**SIP = scoreless half-innings / shutdown opportunities x 100**

A *shutdown opportunity* is a half-inning a pitcher starts immediately after his own team scored 1+ runs. A *shutdown* is a scoreless half-inning in that spot.

Data: Retrosheet play-by-play (via the Chadwick Baseball Bureau mirror), MLB regular seasons 2000-2025: **61,634 games, 1,101,190 half-innings, 291,338 shutdown opportunities.**

## Headline findings
| Question | Answer |
|---|---|
| League SIP vs all other half-innings | 72.08% vs 72.17% (adjusted OR 1.003, 95% CI 0.988-1.019) |
| Does a bigger previous inning change SIP? | No (1 run 72.1%, 5+ runs 71.9%; all ORs include 1) |
| Coasting with big leads? | Small: 5+ run lead is about 0.8 points lower within the same pitchers (SE 0.3) |
| Is SIP a pitcher skill? | Only as general run prevention. Split-half r: SIP 0.15, ordinary scoreless rate 0.41, shutdown-specific gap -0.04 |
| Why it matters | Shutdown +5.0 win-probability points; allowing a run -13.1 |

## What is in the package
```
SIP_research_paper.pdf        10-page paper with all figures embedded
SIP_dashboard.html            Single-file dashboard, open in any browser (no server)
figures/                      12 PNGs (same images as in the paper)
data/
  shutdown_opportunities_2000_2025.csv    291,338 rows, one per opportunity
  half_innings_all_2000_2025.csv.gz       every half-inning (except each game's first) with flags
  pitcher_career_sip.csv                  3,946 pitchers
  pitcher_season_sip.csv                  pitcher-seasons
  team_season_sip.csv, league_season_sip.csv
  regression_results.csv                  odds ratios and confidence intervals
  data_dictionary.csv
code/                         scripts in run order (see below)
```

## Dashboard
Open `SIP_dashboard.html`. Type in the search bar to find any pitcher (accents ignored; results sorted by opportunities). The profile shows SIP, expected SIP, SIP+, regressed SIP, his scoreless rate in other innings, and a season-by-season table and chart. The leaderboard can be filtered by season and minimum opportunities, and ranked by regressed SIP, raw SIP, SIP+ or volume. Click any row to open that pitcher.

## Key columns
- **SIP**: raw shutdown % (credit to the pitcher who starts the half-inning).
- **expected_SIP**: from a pitcher-independent model (opponent offense that season, park era, inning, home/away, extra-inning runner rule).
- **SIP_plus**: SIP minus expected SIP, in points.
- **SIP_regressed**: SIP shrunk toward expectation by an empirical-Bayes weight (see `reliability_weight`). Use this for rankings.
- **gap_vs_own_baseline**: SIP minus the pitcher's scoreless % in all his other half-innings.

## Conventions and caveats
- Runs in a half-inning count against the whole inning, not just the starting pitcher; the starter of the inning gets the opportunity.
- Walk-off bottom halves count as failures. Extra innings from 2020 use the automatic runner and are flagged (`extra_inning_runner_rule`).
- Regular season only. 2020 is a 60-game season.
- The first half-inning of each game cannot be an opportunity and is not in the half-inning file.
- Expected SIP does not adjust for a pitcher's own defense or pitch quality.

## Reproduce
1. Clone `chadwickbureau/retrosheet` (seasons 2000-2025) and build Chadwick's `cwevent`.
2. Run per season: `cwevent -Q -n -y YEAR -f 0,2,3,4,8,9,14,34,58,59,60,61,96 -x 0,1,2 YEAR???.EV[AN] > eYEAR.csv`
3. Run in order: `build_data.py` -> `analysis1.py` (win probability) -> `analysis2.py` (context model) -> `analysis3.py` (models, pitcher tables, reliability) -> `make_figs.py` -> `export.py` -> `build_dash.py` -> `make_paper.py`. Paths at the top of each script assume `/home/claude`; edit as needed.
Python: pandas, numpy, scikit-learn, statsmodels, matplotlib, reportlab.

## Credit
The information used here was obtained free of charge from and is copyrighted by Retrosheet (www.retrosheet.org). Event data mirrored by the Chadwick Baseball Bureau.
