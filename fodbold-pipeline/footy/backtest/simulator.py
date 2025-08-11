from __future__ import annotations
import pandas as pd
import numpy as np
from ..odds.odds_utils import kelly_fraction

def simulate_bets(bets: pd.DataFrame, commission: float = 0.05, kelly_frac: float = 0.5, cap: float = 0.02):
    bankroll = 1.0
    equity = [bankroll]
    pnl = []
    stakes = []
    rets = []
    for _, r in bets.iterrows():
        p = float(r['model_prob']); o = float(r['market_odds'])
        f = kelly_fraction(p, o, commission) * kelly_frac
        f = max(0.0, min(f, cap))
        stake = bankroll * f
        won = int(r['result'])
        profit = stake * ((o-1.0)*(1.0-commission)) if won else -stake
        bankroll += profit
        pnl.append(profit); stakes.append(stake); equity.append(bankroll)
        rets.append(profit/(stake if stake>0 else 1.0))
    roi = float(sum(pnl)/(sum(stakes) if sum(stakes)>0 else 1.0))
    sharpe = float((sum(rets)/len(rets))/ ( (np.std(rets)+1e-9) ) ) if len(rets)>1 else 0.0
    max_dd = 0.0; peak = equity[0]
    for v in equity:
        if v>peak: peak=v
        dd = (peak - v)/peak if peak>0 else 0.0
        if dd>max_dd: max_dd = dd
    return {"roi": roi, "sharpe": sharpe, "max_drawdown": float(max_dd), "equity_curve": equity}
