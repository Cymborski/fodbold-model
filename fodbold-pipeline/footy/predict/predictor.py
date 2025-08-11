from __future__ import annotations
from typing import Dict, Any, List
from ..odds.odds_utils import implied_prob_from_odds, remove_overround, edge, kelly_fraction

def build_market_json(match_row: dict, model_probs: Dict[str,float], odds_quotes: List[dict],
                      commission: float, edge_threshold: float, kelly_frac: float, cap: float) -> Dict[str, Any]:
    best = {"H":0.0,"D":0.0,"A":0.0}
    for q in odds_quotes:
        m = str(q.get("market","")).lower()
        out = str(q.get("outcome","")).lower()
        if "match winner" in m or "1x2" in m:
            if out in ("home","1"): best["H"] = max(best["H"], float(q["odds"]))
            elif out in ("draw","x"): best["D"] = max(best["D"], float(q["odds"]))
            elif out in ("away","2"): best["A"] = max(best["A"], float(q["odds"]))
    implied = {k: 1.0/best[k] for k in best if best[k]>0}
    implied_adj = remove_overround(implied) if implied else {}
    ed = {k: model_probs[k] - implied_adj.get(k,0.0) for k in ["H","D","A"]}
    kelly = {k: max(0.0, min(kelly_fraction(model_probs[k], best[k] if best[k]>0 else 1.0, commission)*kelly_frac, cap)) for k in ["H","D","A"]}
    return {
        "match_id": match_row["match_id"],
        "datetime_utc": match_row["datetime_utc"],
        "league": match_row["league"],
        "home": match_row["home"],
        "away": match_row["away"],
        "markets": {
            "1x2": {
                "probs": model_probs,
                "betfair": {
                    "marketId": None,
                    "best_back_odds": {"H": best["H"] or None, "D": best["D"] or None, "A": best["A"] or None},
                    "implied_adj": implied_adj
                },
                "edge": ed,
                "kelly_fraction": kelly
            }
        }
    }
