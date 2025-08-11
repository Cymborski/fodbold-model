from __future__ import annotations

TICK_LADDER = [
    (1.01, 2.0, 0.01),
    (2.0, 3.0, 0.02),
    (3.0, 4.0, 0.05),
    (4.0, 6.0, 0.10),
    (6.0, 10.0, 0.20),
    (10.0, 20.0, 0.50),
    (20.0, 30.0, 1.00),
    (30.0, 50.0, 2.00),
    (50.0, 100.0, 5.00),
    (100.0, 1000.0, 10.00),
]

def round_to_tick(odds: float) -> float:
    for lo, hi, step in TICK_LADDER:
        if lo <= odds <= hi:
            n = round((odds - lo)/step)
            val = lo + n*step
            return float(f"{val:.2f}")
    return float(f"{odds:.2f}")

def implied_prob_from_odds(odds: float) -> float:
    return 1.0/odds if odds and odds>0 else 0.0

def remove_overround(implied: dict[str,float]) -> dict[str,float]:
    s = sum(implied.values())
    if s <= 0:
        return implied
    return {k:v/s for k,v in implied.items()}

def edge(model_p: float, market_p: float) -> float:
    return model_p - market_p

def kelly_fraction(p: float, odds: float, commission: float = 0.0) -> float:
    b = (odds - 1.0)*(1.0 - commission)
    q = 1.0 - p
    if b <= 0:
        return 0.0
    f = (p*b - q) / b
    return max(0.0, f)

def breakeven_prob(odds: float, commission: float = 0.0) -> float:
    b = (odds - 1.0)*(1.0 - commission)
    return 1.0/(1.0 + b) if b>0 else 1.0
