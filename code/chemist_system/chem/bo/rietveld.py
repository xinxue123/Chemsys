"""True sequential closed-loop Bayesian optimization on the real Ce-Al-Ge
campaign using Rietveld-measured yields (pool-based active learning). 
The GP prior mean is the DFT-hull selectivity (first-principles, not fitted to yields)."""

import os, json, math
from .. import paths
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))

# Use the Rietveld dataset
from ..paths import ce_al_ge_dataset

DATASET = os.environ.get("ARCMELT_RIETVELD_DATASET", str(ce_al_ge_dataset("arcmelt_dataset_rietveld.json")))

def _load_pool():
    from ..decision.replay import load_cealge_observations
    rows = load_cealge_observations(DATASET)
    X = np.array([[r["params"]["atom_fraction_Ce"], r["params"]["atom_fraction_Al"]] for r in rows])
    Y = np.array([row["yield"] for row in rows])
    return X, Y

def _hull_prior(X):
    """DFT-hull selectivity (Al-flux fraction) x 100, as a yield-scale prior."""
    from ..energetics import hull as HA
    return np.array([100.0 * HA.selectivity(x[0], x[1]) for x in X])

def _phi(z):
    return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)

def _Phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))

class GP:
    """Physics-informed GP: fixed prior-mean fn, RBF kernel on the residual."""
    def __init__(self, prior_fn, bw=0.06, noise=4.0):
        self.prior_fn = prior_fn; self.bw = bw; self.noise = noise
    def fit(self, X, y):
        self.X = np.asarray(X, float); self.y = np.asarray(y, float)
        r = self.y - self.prior_fn(self.X)
        D2 = ((self.X[:, None, :] - self.X[None, :, :]) ** 2).sum(2)
        self.sf = max(float(r.std()), 1.0)
        K = self.sf**2 * np.exp(-0.5*D2/self.bw**2) + self.noise**2 * np.eye(len(X))
        self.Kinv = np.linalg.inv(K); self.alpha = self.Kinv @ r
        return self
    def predict(self, Xq):
        Xq = np.asarray(Xq, float)
        d2 = ((Xq[:, None, :] - self.X[None, :, :]) ** 2).sum(2)
        k = self.sf**2 * np.exp(-0.5*d2/self.bw**2)
        mean = self.prior_fn(Xq) + k @ self.alpha
        var = self.sf**2 - np.einsum("ij,jk,ik->i", k, self.Kinv, k)
        return mean, np.sqrt(np.clip(var, 1e-9, None))

def _ei(mean, std, fbest):
    out = np.zeros_like(mean)
    for i in range(len(mean)):
        s = std[i]
        if s < 1e-9:
            continue
        z = (mean[i] - fbest) / s
        out[i] = (mean[i] - fbest) * _Phi(z) + s * _phi(z)
    return out

def run_campaign(X, Y, prior_fn, rng, n_init=1, screen=None):
    n = len(Y); seen = list(rng.choice(n, n_init, replace=False))
    prior_all = prior_fn(X); best = [max(Y[i] for i in seen)]; refl = []
    while len(seen) < n:
        cand = [i for i in range(n) if i not in seen]
        gp = GP(prior_fn).fit(X[seen], Y[seen])
        mean, std = gp.predict(X[cand])
        acq = _ei(mean, std, max(Y[i] for i in seen))
        if screen is not None:
            ok = np.array([screen(X[i]) for i in cand])
            if ok.any(): acq = np.where(ok, acq, -1e9)
        j = cand[int(np.argmax(acq))]
        best.append(max(best[-1], Y[j]))
        _log_reflect(refl, X[j], prior_all[j], Y[j]); seen.append(j)
    return best, refl

def _log_reflect(refl, comp, prior, measured):
    resid = float(measured - prior)
    refl.append({"comp": [round(float(c), 3) for c in comp],
                 "prior": round(float(prior), 1), "measured": round(float(measured), 1),
                 "residual": round(resid, 1), "surprise": bool(abs(resid) > 20)})

def benchmark(reps=300, seed=0, thresh_frac=0.9):
    X, Y = _load_pool(); n = len(Y); ymax = Y.max(); thr = thresh_frac*ymax
    from ..energetics import hull as HA
    screen = lambda x: HA.selectivity(x[0],x[1]) > 0.6 and (1-x[0]-x[1]) <= 0.20
    zero = lambda XX: np.zeros(len(XX))
    res = {"agent": [], "data_gp": [], "random": []}
    ette = {"agent": [], "data_gp": [], "random": []}
    for r in range(reps):
        ba, _ = run_campaign(X, Y, _hull_prior, np.random.default_rng(seed+r), screen=screen)
        bd, _ = run_campaign(X, Y, zero, np.random.default_rng(seed+r))
        pm = np.random.default_rng(seed+r).permutation(n)
        br = [float(max(Y[pm[:k+1]])) for k in range(n)]
        for nm, b in (("agent", ba), ("data_gp", bd), ("random", br)):
            res[nm].append(b)
            ette[nm].append(next((i+1 for i, v in enumerate(b) if v >= thr), n))
    return _summarize(res, ette, ymax, thr, n, reps)

def _summarize(res, ette, ymax, thr, n, reps):
    out = {"ymax": float(ymax), "thr": float(thr), "n": n, "reps": reps,
           "curves": {}, "ette": {}}
    for k in res:
        A = np.array(res[k])
        out["curves"][k] = {"mean": A.mean(0).round(2).tolist(),
                            "std": A.std(0).round(2).tolist()}
        out["ette"][k] = {"mean": round(float(np.mean(ette[k])), 2),
                          "std": round(float(np.std(ette[k])), 2)}
    return out

if __name__ == "__main__":
    print("Running BO benchmark with Rietveld dataset...")
    print(f"Dataset: {DATASET}")
    r = benchmark(reps=300)
    output_path = str(paths.CE_AL_GE_BENCH / "bo_results_rietveld.json")
    json.dump(r, open(output_path, "w"), indent=1)
    print(f"Results saved to: {output_path}")
    print("\nExperiments to reach 90% of optimum (mean+/-std):")
    for k in ("agent", "data_gp", "random"):
        print(f"  {k:8s} {r['ette'][k]['mean']:.2f} +/- {r['ette'][k]['std']:.2f}")
