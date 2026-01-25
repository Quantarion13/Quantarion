"""
Polyglot Validation + Simulation
-------------------------------
Purpose:
- Validate Kaprekar convergence (4-digit invariants)
- Provide deterministic trace for each seed
- Serve as reference for multi-layer AQARION tests (FFT, SNN, federation)
"""

# --- Kaprekar Validator ---
def kaprekar_step(n: int) -> int:
    s = f"{n:04d}"
    asc = int("".join(sorted(s)))
    desc = int("".join(sorted(s, reverse=True)))
    return desc - asc

def kaprekar_validate(seed: int, max_iter: int = 7):
    if not (0 <= seed <= 9999):
        return {"valid": False, "reason": "out_of_range"}

    s = f"{seed:04d}"
    if len(set(s)) < 2:
        return {"valid": False, "reason": "degenerate_digits"}

    n = seed
    trace = [n]

    for i in range(1, max_iter + 1):
        n = kaprekar_step(n)
        trace.append(n)
        if n == 6174:
            return {
                "valid": True,
                "converged": True,
                "iterations": i,
                "trace": trace
            }

    return {
        "valid": True,
        "converged": False,
        "iterations": max_iter,
        "trace": trace
    }

# --- Test Seeds / Simulation ---
test_seeds = [6174, 3524, 9831, 1000, 2111, 7641, 9998, 1111, 2222, 0]

for seed in test_seeds:
    result = kaprekar_validate(seed)
    print(f"Seed {seed:04d} -> {result}")

# --- Optional Exhaustive Run ---
def exhaustive_kaprekar_simulation():
    converged, degenerate, failed = 0, 0, []
    for seed in range(10000):
        r = kaprekar_validate(seed)
        if not r["valid"]:
            degenerate += 1
        elif r["converged"]:
            converged += 1
        else:
            failed.append(seed)
    return {"converged": converged, "degenerate": degenerate, "failed": failed}

if __name__ == "__main__":
    results = exhaustive_kaprekar_simulation()
    print("Exhaustive Validation Results:", results)
