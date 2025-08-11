import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
import numpy as np
from footy.odds.odds_utils import round_to_tick, implied_prob_from_odds, remove_overround, kelly_fraction, breakeven_prob
from footy.models.calibration import IsoCalibrator
from footy.predict.predictor import build_market_json
from footy.backtest import simulator
import pandas as pd

report = {"ok":True,"checks":[]}

def check(name, cond, detail=None):
    report["checks"].append({"name":name,"ok":bool(cond),"detail":detail})
    if not cond: report["ok"]=False

# odds utils
check("tick_2.18", round_to_tick(2.18)==2.18, round_to_tick(2.18))
check("tick_3.03_to_3.05", round_to_tick(3.03)==3.05, round_to_tick(3.03))
check("implied_2.5", abs(implied_prob_from_odds(2.5)-0.4)<1e-12, implied_prob_from_odds(2.5))
fair = remove_overround({"H":0.5,"D":0.3,"A":0.3})
check("overround_sum1", abs(sum(fair.values())-1.0)<1e-12, sum(fair.values()))
f = kelly_fraction(0.5, 2.0, commission=0.05)
check("kelly_commission_zero_at_even_with_commission", abs(f - 0.0) < 1e-12, f)
be = breakeven_prob(2.20, commission=0.02)
check("breakeven_between_0_1", 0<be<1, be)

# calibration
y = np.array([0,1,2,0,1,2])
proba = np.array([[0.7,0.2,0.1],
                  [0.2,0.6,0.2],
                  [0.1,0.2,0.7],
                  [0.8,0.1,0.1],
                  [0.1,0.8,0.1],
                  [0.2,0.1,0.7]])
cal = IsoCalibrator().fit(y, proba)
cal_p = cal.transform(proba)
check("cal_rows_sum1", np.allclose(cal_p.sum(axis=1), 1.0), cal_p.sum(axis=1).tolist())

# predictor JSON
match_row = {"match_id":"m1","datetime_utc":"2025-08-14T19:00:00Z","league":"Premier League","home":"Arsenal","away":"Spurs"}
model_probs = {"H":0.50,"D":0.27,"A":0.23}
odds_quotes = [
    {"market":"Match Winner","outcome":"Home","odds":2.20},
    {"market":"Match Winner","outcome":"Draw","odds":3.40},
    {"market":"Match Winner","outcome":"Away","odds":3.60},
]
js = build_market_json(match_row, model_probs, odds_quotes, commission=0.05, edge_threshold=0.02, kelly_frac=0.5, cap=0.02)
check("json_has_1x2", "1x2" in js["markets"], list(js["markets"].keys()))
check("json_best_home_odds", js["markets"]["1x2"]["betfair"]["best_back_odds"]["H"]==2.20, js["markets"]["1x2"]["betfair"]["best_back_odds"])

# backtest
bt = pd.DataFrame([
    {"match_id":"m1","outcome":"H","model_prob":0.55,"market_odds":2.10,"result":1},
    {"match_id":"m2","outcome":"A","model_prob":0.40,"market_odds":3.10,"result":0},
    {"match_id":"m3","outcome":"D","model_prob":0.32,"market_odds":3.50,"result":1},
])
res = simulator.simulate_bets(bt, commission=0.05, kelly_frac=0.5, cap=0.02)
check("bt_has_metrics", all(k in res for k in ["roi","sharpe","max_drawdown","equity_curve"]), res)

print(json.dumps(report, indent=2))
