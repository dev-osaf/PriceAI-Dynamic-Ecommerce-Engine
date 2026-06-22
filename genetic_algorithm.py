import numpy as np
from dataclasses import dataclass
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
#  GA Configuration
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class GAConfig:
    pop_size:        int   = 200      # chromosomes per generation
    max_generations: int   = 80       # upper iteration limit
    mutation_rate:   float = 0.15     # probability a gene is mutated
    mutation_sigma:  float = 0.08     # std-dev of Gaussian noise (normalised)
    crossover_rate:  float = 0.80     # probability of crossover vs. cloning
    tournament_k:    int   = 5        # tournament selection group size
    patience:        int   = 20       # early-stop if no improvement
    elite_n:         int   = 4        # top chromosomes copied unchanged


# ─────────────────────────────────────────────────────────────────────────────
#  Problem bounds (normalised [0, 1] internally)
# ─────────────────────────────────────────────────────────────────────────────
PRICE_MIN,    PRICE_MAX    = 5.0,  80.0
DISCOUNT_MIN, DISCOUNT_MAX = 0.0,  40.0
N_GENES = 2   # [price, discount_pct]


def decode(chromosome: np.ndarray) -> tuple[float, float]:
    """Map normalised [0,1] genes → actual (price, discount_pct)."""
    price    = PRICE_MIN    + chromosome[0] * (PRICE_MAX    - PRICE_MIN)
    discount = DISCOUNT_MIN + chromosome[1] * (DISCOUNT_MAX - DISCOUNT_MIN)
    return round(price, 2), round(discount, 2)


def encode(price: float, discount: float) -> np.ndarray:
    """Map actual values → normalised chromosome."""
    g0 = (price    - PRICE_MIN)    / (PRICE_MAX    - PRICE_MIN)
    g1 = (discount - DISCOUNT_MIN) / (DISCOUNT_MAX - DISCOUNT_MIN)
    return np.clip([g0, g1], 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
#  Fitness function
# ─────────────────────────────────────────────────────────────────────────────
def fitness(chromosome: np.ndarray,
            cost:          float = 8.0,
            base_demand:   float = 50.0,
            base_price:    float = 29.0,
            elasticity:    float = -1.8) -> float:
    """
    Estimate total profit for a given (price, discount) chromosome.

    Demand model: price-elastic
      units = base_demand × (base_price / effective_price) ^ |elasticity|

    Profit = (effective_price - cost) × units

    Penalty applied if effective_price < cost (selling at a loss) or
    if the margin is below 5 % (unsustainable).
    """
    price, discount = decode(chromosome)
    effective_price = price * (1.0 - discount / 100.0)

    if effective_price <= cost:
        return -1e6   # hard penalty

    # price-elastic demand
    units   = base_demand * (base_price / effective_price) ** abs(elasticity)
    units   = max(units, 0.0)

    margin  = effective_price - cost
    profit  = margin * units

    # soft penalty for tiny margin (< 5 %)
    if (effective_price - cost) / effective_price < 0.05:
        profit *= 0.5

    return float(profit)


# ─────────────────────────────────────────────────────────────────────────────
#  Operators
# ─────────────────────────────────────────────────────────────────────────────

def tournament_select(population: np.ndarray,
                      fitnesses:  np.ndarray,
                      k:          int,
                      rng:        np.random.Generator) -> np.ndarray:
    """Pick k random chromosomes; return the fittest."""
    indices = rng.integers(0, len(population), size=k)
    best    = indices[np.argmax(fitnesses[indices])]
    return population[best].copy()


def uniform_crossover(p1: np.ndarray,
                      p2: np.ndarray,
                      rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Each gene randomly inherited from either parent."""
    mask = rng.random(N_GENES) < 0.5
    c1   = np.where(mask, p1, p2)
    c2   = np.where(mask, p2, p1)
    return c1, c2


def gaussian_mutate(chromosome: np.ndarray,
                    rate:  float,
                    sigma: float,
                    rng:   np.random.Generator) -> np.ndarray:
    """Add Gaussian noise to each gene with probability = rate."""
    child = chromosome.copy()
    for i in range(N_GENES):
        if rng.random() < rate:
            child[i] += rng.normal(0, sigma)
    return np.clip(child, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
#  Main GA runner
# ─────────────────────────────────────────────────────────────────────────────
def run_ga(cost:        float = 8.0,
           base_demand: float = 50.0,
           base_price:  float = 29.0,
           elasticity:  float = -1.8,
           config:      Optional[GAConfig] = None,
           seed:        int = 42) -> dict:
    """
    Execute the Genetic Algorithm and return the optimised pricing strategy.

    Returns
    ───────
    {
      'best_price'      : float,
      'best_discount'   : float,
      'best_profit'     : float,
      'best_units'      : float,
      'history'         : list[dict] per generation,
      'generations_run' : int,
    }
    """
    if config is None:
        config = GAConfig()

    rng = np.random.default_rng(seed)

    def _fit(c):
        return fitness(c, cost, base_demand, base_price, elasticity)

    # ── initialise random population ──────────────────────────────────────
    population = rng.random((config.pop_size, N_GENES))
    fitnesses  = np.array([_fit(c) for c in population])

    best_ever_fitness  = -np.inf
    best_ever_chrom    = population[np.argmax(fitnesses)].copy()
    stagnation_counter = 0
    history            = []

    for gen in range(config.max_generations):

        # ── elitism: carry top-N unchanged ────────────────────────────────
        elite_idx  = np.argsort(fitnesses)[-config.elite_n:]
        elites     = population[elite_idx].copy()

        # ── build next generation ──────────────────────────────────────────
        new_pop = []
        while len(new_pop) < config.pop_size - config.elite_n:
            p1 = tournament_select(population, fitnesses, config.tournament_k, rng)
            p2 = tournament_select(population, fitnesses, config.tournament_k, rng)

            if rng.random() < config.crossover_rate:
                c1, c2 = uniform_crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            c1 = gaussian_mutate(c1, config.mutation_rate, config.mutation_sigma, rng)
            c2 = gaussian_mutate(c2, config.mutation_rate, config.mutation_sigma, rng)
            new_pop.extend([c1, c2])

        population = np.vstack([elites, np.array(new_pop[:config.pop_size - config.elite_n])])
        fitnesses  = np.array([_fit(c) for c in population])

        # ── track best ────────────────────────────────────────────────────
        gen_best_idx = np.argmax(fitnesses)
        gen_best_fit = fitnesses[gen_best_idx]
        gen_best_p, gen_best_d = decode(population[gen_best_idx])

        # effective price for units calculation
        eff = gen_best_p * (1 - gen_best_d / 100)
        gen_units = base_demand * (base_price / max(eff, 0.01)) ** abs(elasticity)

        history.append({
            "generation":   gen + 1,
            "best_fitness": round(gen_best_fit, 4),
            "best_price":   gen_best_p,
            "best_discount": gen_best_d,
            "mean_fitness": round(float(fitnesses.mean()), 4),
            "units":        round(max(gen_units, 0), 2),
        })

        if gen_best_fit > best_ever_fitness + 1e-6:
            best_ever_fitness  = gen_best_fit
            best_ever_chrom    = population[gen_best_idx].copy()
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        if stagnation_counter >= config.patience:
            print(f"[GA] Early stop at generation {gen+1} (no improvement for {config.patience} gens)")
            break

    best_price, best_discount = decode(best_ever_chrom)
    eff_best = best_price * (1 - best_discount / 100)
    best_units = base_demand * (base_price / max(eff_best, 0.01)) ** abs(elasticity)

    return {
        "best_price":      best_price,
        "best_discount":   best_discount,
        "best_profit":     round(best_ever_fitness, 4),
        "best_units":      round(max(best_units, 0), 2),
        "effective_price": round(eff_best, 2),
        "history":         history,
        "generations_run": len(history),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = run_ga(cost=8.0, base_demand=50.0, base_price=29.0, elasticity=-1.8)
    print(f"\n🧬 GA Result after {result['generations_run']} generations:")
    print(f"  Optimal Price    : ${result['best_price']:.2f}")
    print(f"  Optimal Discount : {result['best_discount']:.1f}%")
    print(f"  Effective Price  : ${result['effective_price']:.2f}")
    print(f"  Est. Units Sold  : {result['best_units']:.1f}")
    print(f"  Est. Profit      : ${result['best_profit']:.2f}")
