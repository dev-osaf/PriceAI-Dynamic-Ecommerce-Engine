import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Membership functions  (triangular / trapezoidal shapes)
# ─────────────────────────────────────────────────────────────────────────────

def trimf(x: float, a: float, b: float, c: float) -> float:
    """Triangular membership function: peaks at b, zero at a and c."""
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a)
    else:
        return (c - x) / (c - b)


def trapmf(x: float, a: float, b: float, c: float, d: float) -> float:
    """Trapezoidal membership function: flat top between b and c."""
    if x <= a or x >= d:
        return 0.0
    elif a < x < b:
        return (x - a) / (b - a)
    elif b <= x <= c:
        return 1.0
    else:
        return (d - x) / (d - c)


# ── Demand membership functions ───────────────────────────────────────────────
def demand_low(d: float)    -> float: return trapmf(d, 0,  0,  25, 45)
def demand_medium(d: float) -> float: return trimf (d, 25, 50, 75)
def demand_high(d: float)   -> float: return trapmf(d, 55, 75, 100, 100)


# ── Competitor relative price membership functions ─────────────────────────
# comp_rel = (our_price - comp_price) / our_price  × 100
# positive  → competitor is cheaper than us
# negative  → competitor is more expensive than us

def comp_cheaper(c: float)   -> float: return trapmf(c, 5,  15,  100,  100)
def comp_same(c: float)      -> float: return trimf  (c, -10, 0,  10)
def comp_expensive(c: float) -> float: return trapmf(c, -100, -100, -15, -5)


# ── Discount output universe [0 – 40 %] ──────────────────────────────────────
DISCOUNT_UNIVERSE = np.linspace(0, 40, 500)

def disc_none(x: float)   -> float: return trapmf(x, 0,  0,  3,  6)
def disc_low(x: float)    -> float: return trimf  (x, 3,  10, 18)
def disc_medium(x: float) -> float: return trimf  (x, 12, 22, 32)
def disc_high(x: float)   -> float: return trapmf(x, 25, 33, 40, 40)


# ─────────────────────────────────────────────────────────────────────────────
#  Fuzzy Inference Engine
# ─────────────────────────────────────────────────────────────────────────────

def compute_discount(demand: float, our_price: float, comp_price: float) -> dict:
    """
    Run the full Mamdani fuzzy inference pipeline.

    Parameters
    ──────────
    demand      : float  [0 – 100]
    our_price   : float  (in dollars)
    comp_price  : float  (in dollars)

    Returns
    ───────
    dict with keys:
      'discount_pct'   – crisp recommended discount (0–40 %)
      'memberships'    – debug dict of all membership degrees
      'rule_activations' – dict of activated rule strengths
    """
    # ── 1. Fuzzification ────────────────────────────────────────────────────
    # competitor relative cheapness (positive = they are cheaper than us)
    comp_rel = (our_price - comp_price) / max(our_price, 0.01) * 100

    d_low  = demand_low(demand)
    d_med  = demand_medium(demand)
    d_high = demand_high(demand)

    c_cheap = comp_cheaper(comp_rel)
    c_same  = comp_same(comp_rel)
    c_exp   = comp_expensive(comp_rel)

    # ── 2. Rule Evaluation (Mamdani AND = min) ─────────────────────────────
    r1 = min(d_low,  c_cheap)   # → High
    r2 = min(d_low,  c_same)    # → Medium
    r3 = min(d_low,  c_exp)     # → Low
    r4 = min(d_med,  c_cheap)   # → Medium
    r5 = min(d_med,  c_same)    # → Low
    r6 = min(d_med,  c_exp)     # → None
    r7 = min(d_high, c_cheap)   # → Low
    r8 = min(d_high, c_same)    # → None
    r9 = min(d_high, c_exp)     # → None

    # aggregate consequents (OR = max)
    agg_none   = max(r6, r8, r9)
    agg_low    = max(r3, r5, r7)
    agg_medium = max(r2, r4)
    agg_high   = r1

    # ── 3. Defuzzification (Centroid / Centre-of-Gravity) ──────────────────
    u = DISCOUNT_UNIVERSE
    aggregated_mf = np.zeros_like(u)
    for i, x in enumerate(u):
        clipped = max(
            min(agg_none,   disc_none(x)),
            min(agg_low,    disc_low(x)),
            min(agg_medium, disc_medium(x)),
            min(agg_high,   disc_high(x)),
        )
        aggregated_mf[i] = clipped

    denom = aggregated_mf.sum()
    if denom < 1e-10:
        crisp_discount = 0.0
    else:
        crisp_discount = float((u * aggregated_mf).sum() / denom)

    return {
        "discount_pct": round(crisp_discount, 2),
        "memberships": {
            "demand_low": round(d_low, 3),
            "demand_medium": round(d_med, 3),
            "demand_high": round(d_high, 3),
            "comp_cheaper": round(c_cheap, 3),
            "comp_same": round(c_same, 3),
            "comp_expensive": round(c_exp, 3),
        },
        "rule_activations": {
            "R1_Low_Cheap→High":    round(r1, 3),
            "R2_Low_Same→Med":      round(r2, 3),
            "R3_Low_Exp→Low":       round(r3, 3),
            "R4_Med_Cheap→Med":     round(r4, 3),
            "R5_Med_Same→Low":      round(r5, 3),
            "R6_Med_Exp→None":      round(r6, 3),
            "R7_High_Cheap→Low":    round(r7, 3),
            "R8_High_Same→None":    round(r8, 3),
            "R9_High_Exp→None":     round(r9, 3),
        },
        "aggregated_mf": aggregated_mf,
        "comp_rel_pct":  round(comp_rel, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Quick self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        (20,  30.0, 25.0, "Low demand, comp cheaper → expect HIGH discount"),
        (50,  30.0, 30.0, "Med demand, comp same    → expect LOW discount"),
        (85,  30.0, 35.0, "High demand, comp pricier → expect NO discount"),
        (30,  29.0, 28.0, "Low demand, comp slightly cheaper → expect MED-HIGH"),
    ]

    for demand, our_price, comp_price, label in test_cases:
        result = compute_discount(demand, our_price, comp_price)
        print(f"\n{label}")
        print(f"  Inputs : demand={demand}, our=${our_price}, comp=${comp_price}")
        print(f"  Comp Rel: {result['comp_rel_pct']:.1f}% (positive=they are cheaper)")
        print(f"  → Recommended Discount: {result['discount_pct']:.2f}%")
