import numpy as np
import pymc as pm
import arviz as az
import pytensor
import pytensor.tensor as pt
import xarray as xr
import matplotlib.pyplot as plt

from scipy.special import logsumexp
from scipy.special import gammaincc  # regularized upper gamma
from math import sqrt
from pytensor.scan.basic import scan
from scipy.stats import gamma as gamma_dist

plt.style.use("ggplot")

def make_intervals(times_data):
    t = np.asarray(times_data, float)
    t = t[np.isfinite(t)]
    t = np.sort(t)
    if t.size == 0:
        raise ValueError("No events")

    # gaps: from 0→t0, then between events
    dt = np.empty_like(t)
    dt[0]  = t[0]                 # gap from 0 to first event
    di     = np.diff(t)
    # robust tiny epsilon to avoid zeros/negatives everywhere
    med    = np.median(di[di>0]) if np.any(di>0) else max(t[0], 600.0)
    eps    = max(1e-12, 1e-9*med)
    dt[1:] = np.where(di > 0, di, eps)
    # clip the first gap too (in case t[0]==0)
    dt[0]  = max(dt[0], eps)

    return t, dt, eps

def ecdf(y):
    """Return x (sorted), F(x) for ECDF."""
    x = np.sort(np.asarray(y, float))
    x = x[np.isfinite(x)]
    n = x.size
    F = np.arange(1, n + 1) / n
    return x, F

def plot_hist_and_ecdf_with_exceedance(dt, taus, tau_labels=None, bins=60):
    dt = np.asarray(dt, float)
    dt = dt[np.isfinite(dt)]
    if dt.size == 0:
        raise ValueError("dt is empty after removing non-finite values.")

    taus = list(taus)
    if tau_labels is None:
        tau_labels = [f"{t:g}" for t in taus]
    if len(tau_labels) != len(taus):
        raise ValueError("tau_labels must have same length as taus.")

    # Precompute ECDF + exceedance probabilities
    x, F = ecdf(dt)
    n = x.size

    def F_at(t):
        # ECDF at t: P(X <= t)
        k = np.searchsorted(x, t, side="right")
        return k / n

    exceed = [1.0 - F_at(t) for t in taus]  # P(X > t)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # Histogram
    ax1.hist(dt, bins=bins,color="steelblue")
    ax1.set_title("Inter-arrival Times (seconds)")
    ax1.set_xlabel(r"$\Delta t$ (s)")
    ax1.set_ylabel("Count")

    # ECDF
    ax2.plot(x, F, linewidth=1.5,color="black")
    ax2.set_title(r"ECDF of $\Delta t$")
    ax2.set_xlabel(r"$\Delta t$ (s)")
    ax2.set_ylabel(r"$F(\Delta t)$")

    # Add threshold lines + auto-annotations for exceedance
    y_positions = np.linspace(0.85, 0.65, num=len(taus))  # stagger text to avoid overlap

    for t, lbl, pexc, ytxt in zip(taus, tau_labels, exceed, y_positions):
        # vertical line on both panels
        ax1.axvline(t, linestyle="--", linewidth=1)
        ax2.axvline(t, linestyle="--", linewidth=1)

        # annotation on ECDF: show exceedance prob
        ax2.text(
            t, ytxt,
            rf"$P(\Delta t > {lbl}) = {pexc*100:.1f}\%$",
            rotation=90,
            va="center",
            ha="right"
        )

    # Make ECDF plot a bit nicer
    ax2.set_ylim(0, 1.02)

    plt.tight_layout()
    return fig, (ax1, ax2), exceed


def fit_hawkes_covariates(
    dt,
    X,
    feature_names=None,
    draws=1200,
    tune=1200,
    chains=4,
    target_accept=0.95,
    random_seed=42,
):
    dt = np.asarray(dt, dtype=float)
    X = np.asarray(X, dtype=float)

    if dt.ndim != 1:
        raise ValueError("dt must be a 1D array of inter-arrival times.")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (N, p).")
    if len(dt) != X.shape[0]:
        raise ValueError("dt and X must have the same number of rows.")
    if np.any(dt <= 0):
        raise ValueError("All inter-arrival times in dt must be positive.")

    N, p = X.shape

    if feature_names is None:
        feature_names = [f"x{i}" for i in range(p)]
    if len(feature_names) != p:
        raise ValueError("feature_names must have length equal to X.shape[1].")

    # Standardize covariates for sampler stability
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0, ddof=0)
    X_std = np.where(X_std == 0, 1.0, X_std)
    Xs = (X - X_mean) / X_std

    # Useful prior centers
    mean_rate = 1.0 / np.mean(dt)
    median_rate = 1.0 / np.median(dt)

    coords = {
        "interval": np.arange(N),
        "feature": feature_names,
    }

    with pm.Model(coords=coords) as hawkes_exog_model:
        # Data containers
        dt_data = pm.Data("dt", dt, dims="interval")
        X_data = pm.Data("X", Xs, dims=("interval", "feature"))

        # ---- Baseline intensity: IPP-style log-linear covariate model ----
        intercept = pm.Normal(
            "intercept",
            mu=np.log(mean_rate),
            sigma=1.5
        )

        beta_x = pm.Normal(
            "beta_x",
            mu=0.0,
            sigma=0.5,
            dims="feature"
        )

        eta = intercept + pt.dot(X_data, beta_x)
        mu = pm.Deterministic("mu", pt.exp(eta), dims="interval")

        # ---- Hawkes excitation parameters ----
        # Branching ratio constrained to (0,1) for stability
        rho = pm.Beta("rho", alpha=2.0, beta=5.0)

        # Exponential kernel decay
        decay = pm.LogNormal(
            "decay",
            mu=np.log(median_rate),
            sigma=1.0
        )

        # Optional derived Hawkes amplitude (for interpretability)
        alpha = pm.Deterministic("alpha", rho * decay)

        # ---- Recursive interval likelihood ----
        #
          # The +1 corresponds to the event occurring at the end of the interval.
        #
        def step(dt_i, mu_i, h_prev, rho, decay):
            e_i = pt.exp(-decay * dt_i)
            h_decay = h_prev * e_i

            lambda_i = mu_i + rho * decay * h_decay
            lambda_i = pt.clip(lambda_i, 1e-12, 1e12)

            integral_i = mu_i * dt_i + rho * h_prev * (1.0 - e_i)
            ll_i = pt.log(lambda_i) - integral_i

            h_new = h_decay + 1.0
            return h_new, ll_i, lambda_i

        (h_seq, ll_seq, lambda_seq), _ = pytensor.scan(
            fn=step,
            sequences=[dt_data, mu],
            outputs_info=[
                pt.as_tensor_variable(np.array(0.0, dtype=np.float64)),  # h_0
                None,
                None,
            ],
            non_sequences=[rho, decay],
        )

        # Deterministics for inspection / PSIS-LOO
        pm.Deterministic("history_state", h_seq, dims="interval")
        pm.Deterministic("lambda_event", lambda_seq, dims="interval")
        pm.Deterministic("ll_interval", ll_seq, dims="interval")

        # Total likelihood
        pm.Potential("hawkes_exog_loglik", ll_seq.sum())

        # Sample
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            init="adapt_diag",
            return_inferencedata=True,
        )

    return hawkes_exog_model, idata, X_mean, X_std


def _get_stat_mean(ss, names):
    """Return mean of the first existing var in `names`, else np.nan."""
    for n in names:
        if n in ss:
            return float(ss[n].mean().values)
    return float("nan")

def _get_stat_sum(ss, names):
    """Return sum of the first existing var in `names`, else 0."""
    for n in names:
        if n in ss:
            return float(ss[n].sum().values)
    return 0.0

def _get_step_summary(ss):
    """Try to summarize step size; fall back to 'scaling' if that's what exists."""
    for cand in ("step_size", "step", "stepsize", "scaling"):
        if cand in ss:
            vals = np.asarray(ss[cand].mean("draw"))
            return float(vals.mean()), float(vals.min()), float(vals.max()), cand
    return float("nan"), float("nan"), float("nan"), None

def diag_table(idata, runtime_s, name="model"):
    # basic sizes
    ch = int(idata.posterior.dims.get("chain", 1))
    dr = int(idata.posterior.dims.get("draw",  idata.posterior.sizes.get("draw", 0)))
    ss = idata.sample_stats

    # divergences (NUTS) or 0 if not available
    divs = _get_stat_sum(ss, ["diverging"])

    # acceptance: try NUTS field, else fall back to generic fields
    acc  = _get_stat_mean(ss, ["acceptance_rate", "accept", "accepted"])

    # step size summary (or scaling)
    step_mean, step_min, step_max, step_name = _get_step_summary(ss)

    # tree depth (NUTS only)
    tdepth = float(ss["tree_depth"].max().values) if "tree_depth" in ss else float("nan")

    # ArviZ diagnostics (robust)
    try:
        rhat_max = float(az.rhat(idata, method="rank").to_array().max().values)
    except Exception:
        rhat_max = float("nan")

    try:
        ess_bulk_min = float(az.ess(idata, method="bulk").to_array().min().values)
        ess_tail_min = float(az.ess(idata, method="tail").to_array().min().values)
    except Exception:
        ess_bulk_min = ess_tail_min = float("nan")

    try:
        bfmi_mean = float(az.bfmi(idata).mean().values)
    except Exception:
        bfmi_mean = float("nan")

    df = pd.DataFrame([{
        "model": name,
        "chains": ch, "draws/chain": dr, "total_draws": ch*dr,
        "divergences": int(divs),
        "accept_rate": acc,
        "step_metric": step_name if step_name else "",
        "step_mean": step_mean, "step_min": step_min, "step_max": step_max,
        "max_tree_depth": tdepth,
        "rhat_max": rhat_max,
        "ess_bulk_min": ess_bulk_min, "ess_tail_min": ess_tail_min,
        "bfmi_mean": bfmi_mean,
        "runtime_sec": float(runtime_s), "runtime_min": float(runtime_s)/60.0
    }])
    return df

def tail_loo(idata, times, tau, prefer="interval"):
    mask, vname, obs_dim = tail_mask_from_idata(idata, times, tau, prefer=prefer)
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise ValueError("No tail observations under this τ; choose a smaller τ.")

    # Copy and subset ONLY the log_likelihood group
    id2 = idata.copy()
    ll_full = id2.log_likelihood[vname]
    ll_tail = ll_full.isel({obs_dim: idx})

    # replace the whole group with the sliced dataset (keeps chain/draw intact)
    id2.log_likelihood = xr.Dataset({vname: ll_tail})

    # LOO on the restricted set
    return az.loo(id2, var_name=vname, pointwise=True)


def _bands(S_draws, qs=(5,50,95), axis=0):
    # S_draws: [n_draws, len(tau)]
    return np.percentile(S_draws, qs, axis=axis)   # [3, len(tau)]
    
def band(x, band, label, color, lw=1.7, alpha=0.15):
    m5,m50,m95 = band
    plt.fill_between(x, m5, m95, alpha=alpha, color=color)
    plt.plot(x, m50, color=color, lw=lw, label=label)
    

def _monotone_decreasing(v):
    # enforce non-increasing (numerical cosmetic)
    return np.maximum.accumulate(v[::-1])[::-1]

def S_poisson(id_pois, taus):
    mu = _draws(id_pois, _get_first(id_pois, ["mu", "lam", "lambda"]))[:, None]  # [n,1]
    S = np.exp(-mu * taus[None, :])                                              # [n,len(taus)]
    return S, _bands(S, qs=(5,50,95), axis=0)

def S_ip(id_ip, X_row, taus):
    beta0 = _draws(id_ip, _get_first(id_ip, ["beta0", "intercept"]))[:, None]    # [n,1]
    bname = _get_first(id_ip, ["b", "beta", "w"])
    b     = _draws(id_ip, bname)                                                 # [n,p] or [n,1,p]
    b     = b.reshape(b.shape[0], -1)
    lin   = beta0 + b.dot(X_row.astype(float))[:, None]                           # [n,1]
    lam   = np.exp(np.clip(lin, -40, 40))
    S     = np.exp(-lam * taus[None, :])
    return S, _bands(S, qs=(5,50,95), axis=0)

def S_gamma(id_gamma, taus):
    k    = _draws(id_gamma, _get_first(id_gamma, ["k", "alpha"]))[:, None]
    rate = _draws(id_gamma, _get_first(id_gamma, ["rate", "beta"]))[:, None]
    S = gammaincc(k, rate * taus[None, :])
    return S, _bands(S, qs=(5,50,95), axis=0)

def S_weibull(id_weib, taus):
    k   = _draws(id_weib, _get_first(id_weib, ["k", "c"]))[:, None]
    lam = _draws(id_weib, _get_first(id_weib, ["lambda", "scale"]))[:, None]
    S = np.exp(- (taus[None, :] / lam)**k)
    return S, _bands(S, qs=(5,50,95), axis=0)

def S_hawkes(id_hawkes, times, taus):
    mu   = _draws(id_hawkes, "mu")     # [n]
    beta = _draws(id_hawkes, "beta")   # [n]
    rho  = _draws(id_hawkes, "rho")    # [n]
    alpha = rho * beta                 # [n]

    # Keep only valid draws
    ok = np.isfinite(mu) & np.isfinite(beta) & np.isfinite(rho) & (beta > 1e-6)
    mu = mu[ok]
    beta = beta[ok]
    rho = rho[ok]
    alpha = alpha[ok]

    if len(mu) == 0:
        raise ValueError("S_hawkes: no valid posterior draws after beta filtering.")

    times = np.asarray(times, dtype=float)
    taus = np.asarray(taus, dtype=float)

    # History term s0
    if len(times) == 0:
        s0 = np.zeros_like(mu)
    else:
        gaps_to_last = times[-1] - times[:-1]   # [N-1]
        expo = -np.outer(beta, gaps_to_last)
        expo = np.clip(expo, -700, 700)
        s0 = np.exp(expo).sum(axis=1)

    tauM = taus[None, :]   # [1, m]

    # Stable version of (1 - exp(-beta*tau)) / beta
    phi = -np.expm1(-beta[:, None] * tauM) / beta[:, None]

    expo2 = -mu[:, None] * tauM - alpha[:, None] * s0[:, None] * phi
    expo2 = np.clip(expo2, -700, 700)

    S = np.exp(expo2)
    S = np.clip(S, 0.0, 1.0)

    return S, _bands(S, qs=(5, 50, 95), axis=0)


def score_hawkes_model(dt, event_times, id_hawkes, taus, tail_thresholds=(10, 30), thin=10):
    dt = np.asarray(dt, dtype=float)
    event_times = np.asarray(event_times, dtype=float)
    taus = np.asarray(taus, dtype=float)

    crps_vals = []
    twcrps_vals = {u: [] for u in tail_thresholds}
    bs_vals = {u: [] for u in tail_thresholds}

    for i, y in enumerate(dt):
        S_draws, _ = S_hawkes(
            id_hawkes=id_hawkes,
            times=event_times[:i],
            taus=taus
        )

        S_draws = S_draws[::thin]

        if S_draws.size == 0:
            raise ValueError(f"score_hawkes_model: no draws left after thinning at i={i}.")

        S_tau = np.nanmean(S_draws, axis=0)

        if not np.all(np.isfinite(S_tau)):
            raise ValueError(f"score_hawkes_model: invalid S_tau at i={i}.")

        S_tau = np.clip(S_tau, 0.0, 1.0)

        crps_vals.append(crps_from_survival(y, taus, S_tau))

        for u in tail_thresholds:
            S_u = np.interp(u, taus, S_tau)
            twcrps_vals[u].append(twcrps_from_survival(y, taus, S_tau, u))
            bs_vals[u].append(brier_exceedance(y, u, S_u))

    return summarize_scores("Hawkes", crps_vals, twcrps_vals, bs_vals)

def S_hawkes_exog_new(id_hawkes_exog, X_row, times, taus):
    # Posterior draws
    intercept = _draws(id_hawkes_exog, "intercept")   # [n]
    beta_x    = _draws(id_hawkes_exog, "beta_x")      # [n, p]
    rho       = _draws(id_hawkes_exog, "rho")         # [n]
    decay     = _draws(id_hawkes_exog, "decay")       # [n]

    X_row = np.asarray(X_row, dtype=float)
    taus = np.asarray(taus, dtype=float)
    t = np.asarray(times, dtype=float)

    # Covariate-driven baseline intensity
    mu_exo = np.exp(np.clip(intercept + beta_x.dot(X_row), -40, 40))   # [n]
    alpha = rho * decay                                                 # [n]

    # Hawkes history state at the last observed event time
    if len(t) == 0:
        s0 = np.zeros_like(mu_exo)                                      # [n]
    else:
        gaps_to_last = t[-1] - t[:-1]                                  # [n_events-1]
        expo = -np.outer(decay, gaps_to_last)
        expo = np.clip(expo, -700, 700)                                # numerical guard
        s0 = np.exp(expo).sum(axis=1)                                  # [n]

    tauM = taus[None, :]
    expo2 = (
        -mu_exo[:, None] * tauM
        - (alpha[:, None] * s0[:, None] / decay[:, None])
          * (1.0 - np.exp(-decay[:, None] * tauM))
    )
    expo2 = np.clip(expo2, -700, 700)
    S = np.exp(expo2)

    return S, _bands(S, qs=(5, 50, 95), axis=0)


def to_seconds(x):
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)

    # Heuristics: Bitcoin median Δt ≈ 600 s
    if 1 < med < 20:         # looks like minutes
        return x * 60.0
    if 1e4 < med < 1e6:      # looks like milliseconds
        return x / 1e3
    if 1e7 < med < 1e9:      # microseconds
        return x / 1e6
    if 1e10 < med < 1e12:    # nanoseconds
        return x / 1e9
    # else: assume already seconds (200–2000 typical)
    return x

def _draws(idata, name):
    """
    Return posterior draws as a NumPy array with chain/draw flattened.
    """
    arr = np.asarray(idata.posterior[name])  # ensure ndarray
    if arr.ndim == 2:
        # scalar param per draw
        return arr.reshape(-1)
    elif arr.ndim >= 3:
        # vector/matrix param per draw
        n = arr.shape[0] * arr.shape[1]
        rest = int(np.prod(arr.shape[2:], dtype=int))
        return arr.reshape(n, rest)
    else:
        # already (n,) — rare, but handle
        return arr.reshape(-1)

# ---------- small utilities ----------
def _get_first(idata, candidates):
    # pick the first var name that exists in posterior
    names = set(idata.posterior.data_vars)
    for n in candidates:
        if n in names:
            return n
    raise KeyError(f"None of {candidates} found. Posterior vars: {sorted(list(names))[:20]}")

def ip_exceed(id_ip, X_future_row, taus):
    """Return P(Δt > τ | posterior) for an inhomogeneous Poisson (IP) model."""
    x = np.asarray(X_future_row, float)
    x = np.where(np.isfinite(x), x, 0.0)  # fill NaNs with 0 (inputs are z-scored)
    p = x.shape[0]

    post = id_ip.posterior
    b_name, beta0_name = _detect_ip_params(id_ip, x)
    if b_name is None:
        raise RuntimeError("Could not find a coefficient vector in id_ip.posterior")

    # coefficients
    b_da = post[b_name]
    if "feat" in b_da.dims:  # (chain, draw, feat)
        b = b_da.transpose("chain","draw","feat").values.reshape(-1, p)
    elif b_da.ndim >= 3:     # (chain, draw, something) with last dim == p
        b = b_da.transpose("chain","draw", b_da.dims[-1]).values.reshape(-1, p)
    elif b_da.ndim == 2 and p == 1:  # scalar slope, 1-feature model
        b = b_da.values.reshape(-1, 1)
    else:
        raise RuntimeError(f"Param '{b_name}' has unexpected shape {b_da.shape} for p={p}")

    # intercept (optional)
    if beta0_name is not None:
        beta0 = _flatten_draws(post[beta0_name])  # (ndraws,)
    else:
        beta0 = np.zeros(b.shape[0], dtype=float)

    # check alignment
    if beta0.shape[0] != b.shape[0]:
        # trim to common ndraws if needed (rare but can happen if groups differ)
        nd = min(beta0.shape[0], b.shape[0])
        beta0 = beta0[:nd]
        b     = b[:nd, :]

    # λ = exp(beta0 + x·b)
    lin = beta0 + b @ x
    lam = np.exp(np.clip(lin, -40, 40))  # (ndraws,)

    taus = np.asarray(taus, float)
    return np.array([np.mean(np.exp(-lam * t)) for t in taus])

def _detect_ip_params(id_ip, x_future):
    post = id_ip.posterior
    names = list(post.data_vars)

    # try common names
    beta0_name = next((n for n in ("beta0","intercept","const","b0","alpha0") if n in names), None)
    b_name     = next((n for n in ("b","beta","w","coef","coefs","x_diff","x_tx") if n in names), None)

    p = len(x_future)

    # if a vector param with feat dim exists, prefer it
    for n in names:
        da = post[n]
        if "feat" in da.dims and da.sizes["feat"] == p:
            b_name = n
            break

    # if still not found, look for any param whose last dim == p
    if b_name is None:
        for n in names:
            da = post[n]
            last_dim = da.dims[-1] if da.ndim >= 3 else None
            if last_dim and da.sizes[last_dim] == p:
                b_name = n
                break

    # if STILL not found and p==1, allow scalar b
    if b_name is None and p == 1:
        # pick any scalar var (chain, draw) that looks like a slope
        scalars = [n for n in names if (post[n].ndim == 2 and not n.endswith("_log__"))]
        if scalars:
            b_name = scalars[0]

    return b_name, beta0_name

def _flatten_draws(da):
    arr = np.asarray(da)
    if arr.ndim == 2:        # (chain, draw)
        return arr.reshape(-1)
    elif arr.ndim >= 3:      # (chain, draw, ...)
        return arr.reshape(arr.shape[0]*arr.shape[1], *arr.shape[2:])
    else:
        return arr


def poisson_exceed(id_pois, taus):
    mu = _get_any(id_pois, ["mu", "lam", "lambda"])
    assert mu is not None
    # P(Δt > τ | μ) = exp(-μ τ)
    return np.array([np.mean(np.exp(-mu * t)) for t in taus])

def gamma_exceed(id_gamma, taus):
    k    = _get_any(id_gamma, ["k", "alpha", "shape"])
    rate = _get_any(id_gamma, ["rate", "beta"])  # PyMC Gamma(alpha=k, beta=rate)
    assert k is not None and rate is not None
    # Survival = Q(k, rate*τ) = gammaincc(k, rate*τ)
    return np.array([np.mean(gammaincc(k, rate*t)) for t in taus])

def weibull_exceed(id_weib, taus):
    k     = _get_any(id_weib, ["k", "alpha", "shape"])
    scale = _get_any(id_weib, ["lam", "lambda", "beta", "scale"])
    assert k is not None and scale is not None
    # Survival = exp(-(τ/scale)^k)
    return np.array([np.mean(np.exp(- (t/scale)**k)) for t in taus])

# --- Hawkes (simulate next gap via thinning; conditional on history) ---
def hawkes_nextgap_exceed(id_hawkes, times_data, taus, n_sims_per_draw=50, max_propose=10000, seed=123):
    rng = np.random.default_rng(seed)
    mu   = _get_any(id_hawkes, ["mu"])
    beta = _get_any(id_hawkes, ["beta"])
    alpha= _get_any(id_hawkes, ["alpha"])
    rho  = _get_any(id_hawkes, ["rho"])
    assert (mu is not None) and (beta is not None) and (alpha is not None or rho is not None)
    if alpha is None:
        # some fits store rho and beta; alpha = rho*beta
        alpha = rho * beta

    times = np.asarray(times_data, float)
    dt    = np.diff(times)
    N     = dt.size

    def s_last_post(beta_val):
        # s_{i} recursion after event i : s_i = e^{-β Δt_i} (1 + s_{i-1}), s_0=0
        s = 0.0
        for d in dt:
            s = np.exp(-beta_val * d) * (1.0 + s)
        return s

    def next_gap_sim(mu_val, alpha_val, beta_val, s0, rng):
        # intensity(t) = mu + alpha * s0 * exp(-beta t), decreasing in t
        # Ogata thinning for monotone decreasing: propose from Exp(M), M = current intensity
        t = 0.0
        M = mu_val + alpha_val * s0
        if M <= 0:
            # degenerate safeguard
            return np.inf
        for _ in range(max_propose):
            w = rng.exponential(1.0 / M)
            t += w
            lam_t = mu_val + alpha_val * s0 * np.exp(-beta_val * t)
            if rng.uniform() * M <= lam_t:
                return t
            # decrease bound to the current (smaller) intensity
            M = lam_t
        return t  # fallback (should not happen with sensible params)

    # Monte Carlo across posterior draws
    res = np.zeros_like(taus, dtype=float)
    for j in range(len(mu)):
        s0 = s_last_post(beta[j])
        for _ in range(n_sims_per_draw):
            gap = next_gap_sim(mu[j], alpha[j], beta[j], s0, rng)
            res += (gap > taus).astype(float)
    res /= (len(mu) * n_sims_per_draw)
    return res

import numpy as np

def hawkes_exog_exceed(id_hawkes, id_ip, X_row, times, taus, pair_mode="min",
                       q_bands=(5, 50, 95), return_bands=True):
    taus = np.asarray(taus, dtype=float)
    X_row = np.asarray(X_row, dtype=float).ravel()

    # --- Pull posterior draws (you already have helpers; using your naming conventions) ---
    mu_h   = _draws(id_hawkes, "mu")
    beta_h = _draws(id_hawkes, "beta")
    rho_h  = _draws(id_hawkes, "rho")
    alpha_h = rho_h * beta_h

    beta0 = _draws(id_ip, _get_first(id_ip, ["beta0", "intercept"]))
    bname = _get_first(id_ip, ["b", "beta", "w"])
    b     = _draws(id_ip, bname)
    if b.ndim == 1:
        b = b[:, None]
    b = b.reshape(-1, X_row.size)

    # --- Align number of draws ---
    n_h = len(mu_h)
    n_i = len(beta0)

    if pair_mode == "min":
        n = min(n_h, n_i)
        mu_h, beta_h, alpha_h = mu_h[:n], beta_h[:n], alpha_h[:n]
        beta0, b              = beta0[:n], b[:n, :]
    elif pair_mode == "hawkes":
        n = n_h
        if n_i < n:
            reps = int(np.ceil(n / n_i))
            beta0 = np.tile(beta0, reps)[:n]
            b     = np.tile(b, (reps, 1))[:n, :]
        else:
            beta0, b = beta0[:n], b[:n, :]
    elif pair_mode == "ip":
        n = n_i
        if n_h < n:
            reps = int(np.ceil(n / n_h))
            mu_h   = np.tile(mu_h, reps)[:n]
            beta_h = np.tile(beta_h, reps)[:n]
            alpha_h= np.tile(alpha_h, reps)[:n]
        else:
            mu_h, beta_h, alpha_h = mu_h[:n], beta_h[:n], alpha_h[:n]
    else:
        raise ValueError("pair_mode must be one of {'min','hawkes','ip'}")

    # --- Exogenous baseline (per draw) ---
    lin = np.clip(beta0 + (b * X_row[None, :]).sum(axis=1), -40, 40)
    mu_exo = np.exp(lin)  # shape (n,)

    # --- Hawkes excitation summary term based on history ---
    t = np.asarray(times, dtype=float)
    gaps_to_last = t[-1] - t[:-1]                       # shape (m,)
    s0 = np.exp(-np.outer(beta_h, gaps_to_last)).sum(axis=1)  # shape (n,)

    # --- Survival for next gap under hybrid: S(τ)=P(Δt>τ) ---
    tauM = taus[None, :]  # (1, T)
    S = np.exp(
        -mu_exo[:, None] * tauM
        - (alpha_h[:, None] * s0[:, None] / beta_h[:, None]) * (1.0 - np.exp(-beta_h[:, None] * tauM))
    )  # shape (n, T)

    # Convert to exceedance bands across posterior draws
    qs = np.percentile(S, q_bands, axis=0)
    # qs rows correspond to q_bands
    if return_bands:
        lo, med, hi = qs[0], qs[1], qs[2]
        return med, lo, hi
    else:
        return qs[1]

def _get_any(idata, candidates):
    post = idata.posterior
    for name in candidates:
        if name in post:
            return np.asarray(post[name]).reshape(-1)  # flatten chain,draw
    return None

# ---- Wilson binomial CI for a proportion ----
def wilson_ci(k, n, alpha=0.10):
    # two-sided (e.g., alpha=0.10 -> 90% CI)
    from scipy.stats import norm
    if n == 0:
        return (0.0, 1.0)
    z = norm.ppf(1 - alpha/2)
    p = k / n
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    half = z * sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return center - half, center + half

# ---- Empirical survival + CI over a grid of taus (seconds) ----
def empirical_survival_with_ci(times, taus, alpha=0.10):
    
    times = np.asarray(times)
    taus = np.asarray(taus)
    n = len(times)
    p_hat = np.zeros_like(taus, dtype=float)
    lo = np.zeros_like(taus, dtype=float)
    hi = np.zeros_like(taus, dtype=float)
    for j, t in enumerate(taus):
        k = np.count_nonzero(times > t)
        p_hat[j] = k / n
        lo[j], hi[j] = wilson_ci(k, n, alpha=alpha)
    return p_hat, lo, hi

def extract_ll_array(idata):
    """
    Standardize a model's log-likelihood into dims ('chain','draw','obs').
    """
    if not hasattr(idata, "log_likelihood"):
        raise ValueError("idata missing log_likelihood group")

    ll_varnames = list(idata.log_likelihood.data_vars)
    if len(ll_varnames) == 0:
        raise ValueError("no log_likelihood variables in idata.log_likelihood")

    # pick a sensible ll var
    if "interval" in ll_varnames:
        key = "interval"
    elif "y" in ll_varnames:
        key = "y"
    else:
        key = ll_varnames[0]

    ll_raw = idata.log_likelihood[key]

    # find obs dim = anything not chain/draw
    obs_dims = [d for d in ll_raw.dims if d not in ("chain","draw")]
    if len(obs_dims) != 1:
        raise ValueError(
            f"Expected exactly one observation dim, got {obs_dims} for var {key}"
        )
    obs_dim = obs_dims[0]

    # rename obs dim to "obs" for uniformity
    ll_std = ll_raw.rename({obs_dim: "obs"})
    return ll_std  # dims ('chain','draw','obs')

def make_tail_idata(idata, dt_full, tau):
    
    ll_std = extract_ll_array(idata)  # ('chain','draw','obs')

    dt_full = np.asarray(dt_full)
    n_obs_ll = ll_std.sizes["obs"]
    if dt_full.shape[0] != n_obs_ll:
        raise ValueError(
            f"dt length {dt_full.shape[0]} != ll obs dim {n_obs_ll}. "
            "dt_full must match the per-interval likelihood length for this model."
        )

    # choose only tail events (Δt > tau)
    mask_tail = dt_full > tau
    idx_tail = np.nonzero(mask_tail)[0]
    if idx_tail.size == 0:
        raise ValueError(f"No tail observations found above tau={tau}")

    # slice down
    ll_tail = ll_std.isel(obs=idx_tail).astype(np.float32)

    chains = ll_tail.sizes["chain"]
    draws  = ll_tail.sizes["draw"]
    obs_n  = ll_tail.sizes["obs"]

    # --- make a dummy posterior Dataset with *one* param ---
    dummy_param = np.zeros((chains, draws), dtype=np.float32)
    dummy_post = xr.Dataset(
        data_vars={
            "_dummy_param": (("chain","draw"), dummy_param)
        },
        coords={
            "chain": np.arange(chains),
            "draw":  np.arange(draws),
        },
    )

    # --- and a log_likelihood Dataset ---
    ll_ds = xr.Dataset(
        data_vars={
            "interval": (("chain","draw","obs"), ll_tail.values)
        },
        coords={
            "chain": np.arange(chains),
            "draw":  np.arange(draws),
            "obs":   np.arange(obs_n),
        },
    )

    tail_idata = az.InferenceData(
        posterior=dummy_post,
        log_likelihood=ll_ds,
    )
    return tail_idata

def compare_tail_models(idatas, dt_full, tau):
    tail_idatas = {}
    for name, idata in idatas.items():
        if idata is None:
            continue
        tail_idatas[name] = make_tail_idata(idata, dt_full, tau)

    cmp_tail = az.compare(
        tail_idatas,
        ic="loo",
        method="stacking",
        var_name="interval",  # matches the key we just used in ll_ds
    )
    return cmp_tail

def build_weibull_model(dt, coords):
    with pm.Model(coords=coords) as model:
        dt_c = pm.Data("dt", dt, dims=["interval"])

        k = pm.Exponential("k", 1.0)
        lambda_ = pm.Exponential("lambda", 1.0)

        y = pm.Weibull(
            "y",
            alpha=k,
            beta=lambda_,
            observed=dt_c,
            dims=["interval"],
        )
    return model

def build_gamma_model(dt,coords):
    with pm.Model(coords=coords) as model:
        dt_c = pm.Data("dt", dt, dims=["interval"])
    
        alpha = pm.Exponential("alpha", 1.0)
        beta  = pm.Exponential("beta", 1.0)
    
        y = pm.Gamma(
            "y",
            alpha=alpha,
            beta=beta,
            observed=dt_c,
            dims=["interval"],
        )
    return model

def hawkes_ll_interval_pt(dt, mu, alpha, beta):
    dt = pt.as_tensor_variable(dt)

    def step(dt_i, r_prev, mu, alpha, beta):
        decayed = r_prev * pt.exp(-beta * dt_i)
        lam_end = mu + decayed
        H_i = mu * dt_i + (r_prev / beta) * (1.0 - pt.exp(-beta * dt_i))
        ll_i = pt.log(lam_end) - H_i
        r_next = decayed + alpha
        return r_next, ll_i

    r0 = pt.constant(0.0, dtype=dt.dtype)

    (_, ll_seq), _ = scan(
        fn=step,
        sequences=dt,
        outputs_info=[r0, None],
        non_sequences=[mu, alpha, beta],
        strict=True,
    )

    return ll_seq
    

def hawkes_logp(value, dt_template, mu, alpha, beta):
    # dt_template is only there to give PyMC the support shape
    return pt.sum(hawkes_ll_interval_pt(value, mu, alpha, beta))

def hawkes_random(dt_template, mu, alpha, beta, rng=None, size=None):
    if rng is None:
        rng = np.random.default_rng()

    # Determine number of intervals n from template
    n = int(np.asarray(dt_template).shape[-1])

    # If PyMC asks for multiple draws, it will pass size=(n_draws, n)
    if size is None or size == () or size == (0,):
        # Single vector draw
        batch = 1
    elif isinstance(size, tuple):
        if len(size) == 1:
            batch = size[0]
        else:
            batch = size[0]
    else:
        batch = int(size)

    def simulate_one(mu, alpha, beta, n, rng):
        out = np.empty(n, dtype=float)
        r = 0.0

        for i in range(n):
            u = rng.uniform()
            target = -np.log1p(-u)

            def g(t):
                return mu * t + (r / beta) * (1.0 - np.exp(-beta * t)) - target

            lo, hi = 0.0, 1.0
            while g(hi) < 0:
                hi *= 2.0
                if hi > 1e6:
                    break

            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if g(mid) < 0:
                    lo = mid
                else:
                    hi = mid

            dt_i = 0.5 * (lo + hi)
            out[i] = dt_i
            r = r * np.exp(-beta * dt_i) + alpha

        return out

    # If only one draw requested
    if batch == 1:
        return simulate_one(mu, alpha, beta, n, rng)

    # If multiple draws requested
    out = np.empty((batch, n), dtype=float)
    for b in range(batch):
        out[b] = simulate_one(mu, alpha, beta, n, rng)

    return out


def build_hawkes_model(dt):
    coords = {"interval": np.arange(len(dt))}
    
    with pm.Model(coords=coords) as hawkes_m:
        dt_c = pm.Data("dt", np.asarray(dt, dtype=float), dims="interval")
    
        mu = pm.HalfNormal("mu", 1.0)
        beta = pm.HalfNormal("beta", 1.0)
        rho = pm.Beta("rho", 2, 5)
        alpha = pm.Deterministic("alpha", rho * beta)
    
        # optional soft stationarity constraint
        pm.Potential("stability", pt.switch(alpha / beta < 0.98, 0.0, -np.inf))
    
        def hawkes_logp(value, mu, alpha, beta):
            ll_i = hawkes_ll_interval_pt(value, mu, alpha, beta)
            return ll_i
    
        y = pm.CustomDist(
            "y",
            mu,
            alpha,
            beta,
            logp=hawkes_logp,
            random=hawkes_random,
            observed=dt_c,
            dims="interval",
            ndim_supp=0,
        )
    
        id_hawkes = pm.sample(
            1200,
            tune=1200,
            chains=4,
            step=pm.Metropolis(),
            return_inferencedata=True,
            init="adapt_diag",
            idata_kwargs={"log_likelihood": True},
            random_seed=42,
        )

    return hawkes_m

def loo_pit_hawkes_covariates(idata, dt, X, X_mean, X_std, ll_name="ll_interval"):
    
    dt = np.asarray(dt, dtype=float)
    X = np.asarray(X, dtype=float)
    Xs = (X - X_mean) / X_std

    N, p = Xs.shape

    # Posterior draws
    intercept = (
        idata.posterior["intercept"]
        .stack(sample=("chain", "draw"))
        .values
    )  # (S,)

    beta_x = (
        idata.posterior["beta_x"]
        .stack(sample=("chain", "draw"))
        .transpose("feature", "sample")
        .values
    )  # (p, S)

    rho = (
        idata.posterior["rho"]
        .stack(sample=("chain", "draw"))
        .values
    )  # (S,)

    decay = (
        idata.posterior["decay"]
        .stack(sample=("chain", "draw"))
        .values
    )  # (S,)

    # Pointwise log-likelihood from posterior deterministic
    ll = (
        idata.posterior[ll_name]
        .stack(sample=("chain", "draw"))
        .transpose("interval", "sample")
        .values
    )  # (N, S)

    S = ll.shape[1]

    # Covariate-driven baseline for every interval and posterior draw
    eta = intercept[None, :] + Xs @ beta_x   # (N, S)
    eta = np.clip(eta, -40, 40)
    mu = np.exp(eta)                         # (N, S)

    # PSIS-smoothed LOO weights
    raw_log_weights = -ll
    smoothed_log_weights, pareto_k = az.psislw(raw_log_weights)

    row_max = np.max(smoothed_log_weights, axis=1, keepdims=True)
    w = np.exp(smoothed_log_weights - row_max)
    w /= w.sum(axis=1, keepdims=True)

    # Rebuild h_prev for each interval and posterior draw
    h_prev_mat = np.zeros((N, S), dtype=float)
    h = np.zeros(S, dtype=float)

    for i in range(N):
        h_prev_mat[i, :] = h
        e_i = np.exp(-decay * dt[i])
        h = h * e_i + 1.0

    # Conditional CDF evaluated at the observed dt_i
    dt_mat = dt[:, None]
    e_obs = np.exp(-decay[None, :] * dt_mat)

    cum_hazard = mu * dt_mat + rho[None, :] * h_prev_mat * (1.0 - e_obs)
    cdf_vals = 1.0 - np.exp(-cum_hazard)
    cdf_vals = np.clip(cdf_vals, 1e-12, 1 - 1e-12)

    loo_pit = np.sum(w * cdf_vals, axis=1)
    return loo_pit, pareto_k

def loo_pit_hawkes_covariates_chunked(idata, dt, X, X_mean, X_std, ll_name="ll_interval"):
    dt = np.asarray(dt, dtype=float)
    X = np.asarray(X, dtype=float)
    Xs = (X - X_mean) / X_std

    N, p = Xs.shape

    # Posterior draws
    intercept = idata.posterior["intercept"].stack(sample=("chain","draw")).values
    beta_x = (
        idata.posterior["beta_x"]
        .stack(sample=("chain","draw"))
        .transpose("feature","sample")
        .values
    )
    rho = idata.posterior["rho"].stack(sample=("chain","draw")).values
    decay = idata.posterior["decay"].stack(sample=("chain","draw")).values

    # Pointwise log-likelihood
    ll = (
        idata.posterior[ll_name]
        .stack(sample=("chain","draw"))
        .transpose("interval","sample")
        .values
    )

    S = ll.shape[1]

    # PSIS weights
    raw_log_weights = -ll
    smoothed_log_weights, pareto_k = az.psislw(raw_log_weights)

    # Normalize weights row-wise
    row_max = np.max(smoothed_log_weights, axis=1, keepdims=True)
    w = np.exp(smoothed_log_weights - row_max)
    w /= w.sum(axis=1, keepdims=True)

    # Rebuild Hawkes history state sequentially
    h = np.zeros(S)
    loo_pit = np.zeros(N)

    for i in range(N):
        # Compute mu_i for all posterior draws
        eta = intercept + Xs[i] @ beta_x
        eta = np.clip(eta, -40, 40)
        mu_i = np.exp(eta)

        # Conditional CDF at observed dt_i
        e_i = np.exp(-decay * dt[i])
        cum_hazard = mu_i * dt[i] + rho * h * (1.0 - e_i)
        cdf_vals = 1.0 - np.exp(-cum_hazard)

        # LOO-PIT using PSIS weights
        loo_pit[i] = np.sum(w[i] * cdf_vals)

        # Update Hawkes history
        h = h * e_i + 1.0

        if i % 2000 == 0:
            print(f"Processed interval {i}/{N}")

    return loo_pit, pareto_k


def crps_from_survival(y_obs, taus, S_tau):
    
    taus = np.asarray(taus, dtype=float)
    S_tau = np.asarray(S_tau, dtype=float)
    indicator = (y_obs > taus).astype(float)
    return np.trapz((S_tau - indicator) ** 2, taus)


def twcrps_from_survival(y_obs, taus, S_tau, u_tail):
    """
    Tail-weighted CRPS with weight 1{t >= u_tail}.
    """
    taus = np.asarray(taus, dtype=float)
    S_tau = np.asarray(S_tau, dtype=float)
    indicator = (y_obs > taus).astype(float)
    weight = (taus >= u_tail).astype(float)
    return np.trapz(weight * (S_tau - indicator) ** 2, taus)


def brier_exceedance(y_obs, u, S_u):
    """
    Brier score for event Y > u.
    """
    z = float(y_obs > u)
    return (S_u - z) ** 2


def summarize_scores(model_name, crps_vals, twcrps_vals_dict, bs_vals_dict):
    out = {
        "model": model_name,
        "CRPS": np.mean(crps_vals),
    }
    for u, vals in twcrps_vals_dict.items():
        out[f"twCRPS>{u}"] = np.mean(vals)
    for u, vals in bs_vals_dict.items():
        out[f"BS(Y>{u})"] = np.mean(vals)
    return out

def score_poisson_model(
    dt,
    lambda_draws,
    taus,
    tail_thresholds=(600, 900, 1200)
):
    dt = np.asarray(dt, dtype=float)
    lambda_draws = np.asarray(lambda_draws, dtype=float)
    taus = np.asarray(taus, dtype=float)

    crps_vals = []
    twcrps_vals = {u: [] for u in tail_thresholds}
    bs_vals = {u: [] for u in tail_thresholds}

    # Posterior predictive survival is common to every observation.
    # This was already outside your loop.
    S_tau = np.exp(
        -lambda_draws[:, None] * taus[None, :]
    ).mean(axis=0)

    # OPTIMISATION:
    # S_u depends on model draws and threshold only,
    # not on observation y. Compute it once.
    S_u_cache = {
        u: np.exp(-lambda_draws * u).mean()
        for u in tail_thresholds
    }

    for y in dt:

        crps_vals.append(
            crps_from_survival(
                y,
                taus,
                S_tau
            )
        )

        for u in tail_thresholds:

            S_u = S_u_cache[u]

            twcrps_vals[u].append(
                twcrps_from_survival(
                    y,
                    taus,
                    S_tau,
                    u
                )
            )

            bs_vals[u].append(
                brier_exceedance(
                    y,
                    u,
                    S_u
                )
            )

    summary = summarize_scores(
        "Poisson",
        crps_vals,
        twcrps_vals,
        bs_vals
    )

    pointwise = {
        "CRPS": np.asarray(crps_vals, dtype=float),

        "twCRPS": {
            u: np.asarray(twcrps_vals[u], dtype=float)
            for u in tail_thresholds
        },

        "Brier": {
            u: np.asarray(bs_vals[u], dtype=float)
            for u in tail_thresholds
        }
    }

    return summary, pointwise

def score_ip_covariates_model(
    dt,
    X,
    intercept,
    beta,
    taus,
    tail_thresholds=(10, 30)
):
    dt = np.asarray(dt, dtype=float)
    X = np.asarray(X, dtype=float)
    taus = np.asarray(taus, dtype=float)

    crps_vals = []
    twcrps_vals = {u: [] for u in tail_thresholds}
    bs_vals = {u: [] for u in tail_thresholds}

    for i, y in enumerate(dt):
        lam_i = np.exp(
            np.clip(
                intercept + np.dot(beta, X[i]),
                -40,
                40
            )
        )

        S_tau = np.exp(
            -lam_i[:, None] * taus[None, :]
        ).mean(axis=0)

        crps_vals.append(
            crps_from_survival(
                y,
                taus,
                S_tau
            )
        )

        for u in tail_thresholds:

            S_u = np.exp(
                -lam_i * u
            ).mean()

            twcrps_vals[u].append(
                twcrps_from_survival(
                    y,
                    taus,
                    S_tau,
                    u
                )
            )

            bs_vals[u].append(
                brier_exceedance(
                    y,
                    u,
                    S_u
                )
            )

    # -------------------------------------------------------
    # Aggregate summary - same as before
    # -------------------------------------------------------

    summary = summarize_scores(
        "IP + covariates",
        crps_vals,
        twcrps_vals,
        bs_vals
    )

    # -------------------------------------------------------
    # Pointwise scores for bootstrap uncertainty
    # -------------------------------------------------------

    pointwise = {
        "CRPS": np.asarray(
            crps_vals,
            dtype=float
        ),

        "twCRPS": {
            u: np.asarray(
                twcrps_vals[u],
                dtype=float
            )
            for u in tail_thresholds
        },

        "Brier": {
            u: np.asarray(
                bs_vals[u],
                dtype=float
            )
            for u in tail_thresholds
        },
    }

    return summary, pointwise

def score_weibull_model(
    dt,
    shape_k,
    scale_lambda,
    taus,
    tail_thresholds=(10, 30)
):
    dt = np.asarray(dt, dtype=float)
    shape_k = np.asarray(shape_k, dtype=float)
    scale_lambda = np.asarray(scale_lambda, dtype=float)
    taus = np.asarray(taus, dtype=float)

    crps_vals = []
    twcrps_vals = {u: [] for u in tail_thresholds}
    bs_vals = {u: [] for u in tail_thresholds}

    # OPTIMISATION:
    # Your original function recomputed this identical survival
    # curve once for every y. It does not depend on y.
    #
    # Formula is UNCHANGED.
    S_tau = np.exp(
        -scale_lambda[:, None] * taus[None, :]
    ).mean(axis=0)

    # Same optimisation for threshold survival values.
    S_u_cache = {
        u: np.exp(-scale_lambda * u).mean()
        for u in tail_thresholds
    }

    for y in dt:

        crps_vals.append(
            crps_from_survival(
                y,
                taus,
                S_tau
            )
        )

        for u in tail_thresholds:

            S_u = S_u_cache[u]

            twcrps_vals[u].append(
                twcrps_from_survival(
                    y,
                    taus,
                    S_tau,
                    u
                )
            )

            bs_vals[u].append(
                brier_exceedance(
                    y,
                    u,
                    S_u
                )
            )

    summary = summarize_scores(
        "Weibull",
        crps_vals,
        twcrps_vals,
        bs_vals
    )

    pointwise = {
        "CRPS": np.asarray(crps_vals, dtype=float),

        "twCRPS": {
            u: np.asarray(twcrps_vals[u], dtype=float)
            for u in tail_thresholds
        },

        "Brier": {
            u: np.asarray(bs_vals[u], dtype=float)
            for u in tail_thresholds
        }
    }

    return summary, pointwise



def score_gamma_model(
    dt,
    shape_draws,
    scale_draws,
    taus,
    tail_thresholds=(600, 900, 1200)
):
    dt = np.asarray(dt, dtype=float)
    shape_draws = np.asarray(shape_draws, dtype=float)
    scale_draws = np.asarray(scale_draws, dtype=float)
    taus = np.asarray(taus, dtype=float)

    crps_vals = []
    twcrps_vals = {u: [] for u in tail_thresholds}
    bs_vals = {u: [] for u in tail_thresholds}

    # Posterior predictive survival on whole grid.
    # EXACT SAME formula.
    S_tau = gamma_dist.sf(
        taus[None, :],
        a=shape_draws[:, None],
        scale=scale_draws[:, None],
    ).mean(axis=0)

    # OPTIMISATION:
    # Your old function recalculated this 15,000 times
    # for every threshold although it is independent of y.
    S_u_cache = {
        u: gamma_dist.sf(
            u,
            a=shape_draws,
            scale=scale_draws,
        ).mean()
        for u in tail_thresholds
    }

    for y in dt:

        crps_vals.append(
            crps_from_survival(
                y,
                taus,
                S_tau
            )
        )

        for u in tail_thresholds:

            S_u = S_u_cache[u]

            twcrps_vals[u].append(
                twcrps_from_survival(
                    y,
                    taus,
                    S_tau,
                    u
                )
            )

            bs_vals[u].append(
                brier_exceedance(
                    y,
                    u,
                    S_u
                )
            )

    summary = summarize_scores(
        "Gamma",
        crps_vals,
        twcrps_vals,
        bs_vals
    )

    pointwise = {
        "CRPS": np.asarray(crps_vals, dtype=float),

        "twCRPS": {
            u: np.asarray(twcrps_vals[u], dtype=float)
            for u in tail_thresholds
        },

        "Brier": {
            u: np.asarray(bs_vals[u], dtype=float)
            for u in tail_thresholds
        }
    }

    return summary, pointwise

def score_hawkes_model(
    dt,
    event_times,
    id_hawkes,
    taus,
    tail_thresholds=(10, 30),
    thin=10
):
    dt = np.asarray(dt, dtype=float)
    event_times = np.asarray(event_times, dtype=float)
    taus = np.asarray(taus, dtype=float)

    crps_vals = []
    twcrps_vals = {u: [] for u in tail_thresholds}
    bs_vals = {u: [] for u in tail_thresholds}

    for i, y in enumerate(dt):

        # EXACT SAME Hawkes survival calculation as before
        S_draws, _ = S_hawkes(
            id_hawkes=id_hawkes,
            times=event_times[:i],
            taus=taus
        )

        # EXACT SAME thinning
        S_draws = S_draws[::thin]

        if S_draws.size == 0:
            raise ValueError(
                f"score_hawkes_model: "
                f"no draws left after thinning at i={i}."
            )

        # EXACT SAME posterior averaging
        S_tau = np.nanmean(S_draws, axis=0)

        if not np.all(np.isfinite(S_tau)):
            raise ValueError(
                f"score_hawkes_model: invalid S_tau at i={i}."
            )

        S_tau = np.clip(S_tau, 0.0, 1.0)

        crps_vals.append(
            crps_from_survival(
                y,
                taus,
                S_tau
            )
        )

        for u in tail_thresholds:

            # EXACT SAME interpolation
            S_u = np.interp(
                u,
                taus,
                S_tau
            )

            twcrps_vals[u].append(
                twcrps_from_survival(
                    y,
                    taus,
                    S_tau,
                    u
                )
            )

            bs_vals[u].append(
                brier_exceedance(
                    y,
                    u,
                    S_u
                )
            )

    summary = summarize_scores(
        "Hawkes",
        crps_vals,
        twcrps_vals,
        bs_vals
    )

    pointwise = {
        "CRPS": np.asarray(crps_vals, dtype=float),

        "twCRPS": {
            u: np.asarray(twcrps_vals[u], dtype=float)
            for u in tail_thresholds
        },

        "Brier": {
            u: np.asarray(bs_vals[u], dtype=float)
            for u in tail_thresholds
        }
    }

    return summary, pointwise

def score_hawkes_covariates_model(
    dt,
    X,
    event_times,
    id_hawkes_exog,
    taus,
    tail_thresholds=(10, 30),
):
    dt = np.asarray(dt, dtype=float)
    X = np.asarray(X, dtype=float)
    event_times = np.asarray(event_times, dtype=float)
    taus = np.asarray(taus, dtype=float)

    crps_vals = []
    twcrps_vals = {u: [] for u in tail_thresholds}
    bs_vals = {u: [] for u in tail_thresholds}

    for i, y in enumerate(dt):

        # EXACT SAME model-specific survival calculation.
        S_draws, _ = S_hawkes_exog_new(
            id_hawkes_exog=id_hawkes_exog,
            X_row=X[i],
            times=event_times[:i],
            taus=taus,
        )

        # EXACT SAME posterior averaging.
        S_tau = S_draws.mean(axis=0)

        crps_vals.append(
            crps_from_survival(
                y,
                taus,
                S_tau
            )
        )

        for u in tail_thresholds:

            # EXACT SAME interpolation
            S_u = np.interp(
                u,
                taus,
                S_tau
            )

            twcrps_vals[u].append(
                twcrps_from_survival(
                    y,
                    taus,
                    S_tau,
                    u
                )
            )

            bs_vals[u].append(
                brier_exceedance(
                    y,
                    u,
                    S_u
                )
            )

    summary = summarize_scores(
        "Hawkes + covariates",
        crps_vals,
        twcrps_vals,
        bs_vals
    )

    pointwise = {
        "CRPS": np.asarray(crps_vals, dtype=float),

        "twCRPS": {
            u: np.asarray(twcrps_vals[u], dtype=float)
            for u in tail_thresholds
        },

        "Brier": {
            u: np.asarray(bs_vals[u], dtype=float)
            for u in tail_thresholds
        }
    }

    return summary, pointwise

def fit_powerlaw_hawkes_covariates(
    dt,
    X,
    feature_names=None,
    draws=1200,
    tune=1200,
    chains=4,
    target_accept=0.95,
    random_seed=42,
    n_basis=12,
    min_timescale=None,
    max_timescale=None,
):
        # ---------------------------------------------------------------
    # Input validation
    # ---------------------------------------------------------------
    dt = np.asarray(dt, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    if dt.ndim != 1:
        raise ValueError("dt must be a 1D array.")

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    if len(dt) != X.shape[0]:
        raise ValueError("dt and X must contain the same number of rows.")

    if np.any(~np.isfinite(dt)):
        raise ValueError("dt contains non-finite values.")

    if np.any(dt <= 0):
        raise ValueError("All inter-arrival times must be positive.")

    if np.any(~np.isfinite(X)):
        raise ValueError("X contains non-finite values.")

    N, n_features = X.shape

    if feature_names is None:
        feature_names = [f"x{i}" for i in range(n_features)]

    if len(feature_names) != n_features:
        raise ValueError(
            "feature_names must have length equal to X.shape[1]."
        )

    # ---------------------------------------------------------------
    # Standardize covariates
    # ---------------------------------------------------------------
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0, ddof=0)

    # Avoid divide-by-zero for constant covariates
    X_std = np.where(X_std == 0.0, 1.0, X_std)

    Xs = (X - X_mean) / X_std

    # ---------------------------------------------------------------
    # Useful empirical scales
    # ---------------------------------------------------------------
    mean_dt = float(np.mean(dt))
    median_dt = float(np.median(dt))

    mean_rate = 1.0 / mean_dt

    # ---------------------------------------------------------------
    # Exponential basis for power-law memory
    #
    # Each basis rate s corresponds roughly to memory scale 1/s.
    #
    # We deliberately span a wide range because the fitted
    # exponential Hawkes model indicated ~28 hour memory.
    # ---------------------------------------------------------------

    if min_timescale is None:
        # Avoid pathological tiny observations controlling the grid.
        # Never shorter than 10 seconds.
        min_timescale = max(
            10.0,
            float(np.percentile(dt, 1))
        )

    if max_timescale is None:
        # Cover at least 30 days and substantially beyond observed
        # inter-arrival scales.
        max_timescale = max(
            30.0 * 24.0 * 3600.0,
            10.0 * float(np.max(dt))
        )

    if min_timescale <= 0:
        raise ValueError("min_timescale must be positive.")

    if max_timescale <= min_timescale:
        raise ValueError(
            "max_timescale must exceed min_timescale."
        )

    # Rate is inverse timescale.
    #
    # Small rates = long memory.
    # Large rates = short memory.
    rate_grid = np.geomspace(
        1.0 / max_timescale,
        1.0 / min_timescale,
        n_basis
    ).astype(np.float64)

    coords = {
        "interval": np.arange(N),
        "feature": feature_names,
        "basis": np.arange(n_basis),
    }

    with pm.Model(coords=coords) as model:

        # ===========================================================
        # DATA
        # ===========================================================

        dt_data = pm.Data(
            "dt",
            dt,
            dims="interval"
        )

        X_data = pm.Data(
            "X",
            Xs,
            dims=("interval", "feature")
        )

        s = pt.as_tensor_variable(rate_grid)

        # ===========================================================
        # COVARIATE-DEPENDENT BASELINE
        #
        # Same specification as existing exponential Hawkes model.
        # ===========================================================

        intercept = pm.Normal(
            "intercept",
            mu=np.log(mean_rate),
            sigma=1.5
        )

        beta_x = pm.Normal(
            "beta_x",
            mu=0.0,
            sigma=0.5,
            dims="feature"
        )

        eta = intercept + pt.dot(X_data, beta_x)

        mu = pm.Deterministic(
            "mu",
            pt.exp(eta),
            dims="interval"
        )

        # ===========================================================
        # HAWKES BRANCHING RATIO
        #
        # Same stability prior as existing exponential model.
        #
        # 0 < rho < 1 guarantees subcritical branching.
        # ===========================================================

        rho = pm.Beta(
            "rho",
            alpha=2.0,
            beta=5.0
        )

        # ===========================================================
        # POWER-LAW PARAMETERS
        #
        # Target form:
        #
        #       (t + c)^(-p)
        #
        # p > 1 is required for finite integrated excitation.
        # ===========================================================

        power_minus_one = pm.LogNormal(
            "power_minus_one",
            mu=np.log(1.0),
            sigma=0.5
        )

        power = pm.Deterministic(
            "power",
            1.0 + power_minus_one
        )

        # Shift parameter prevents singularity at t = 0.
        #
        # Prior centred around the empirical median inter-arrival
        # time but deliberately broad.
        kernel_shift = pm.LogNormal(
            "kernel_shift",
            mu=np.log(median_dt),
            sigma=1.0
        )

        # ===========================================================
        # POWER-LAW -> EXPONENTIAL-BASIS REPRESENTATION
        #
        # Using the identity that a shifted power law can be
        # represented as a continuous mixture of exponentials.
        #
        # On a logarithmic rate grid the unnormalised quadrature
        # weights are proportional to
        #
        #       s^p exp(-c s)
        #
        # Common constants cancel after normalization.
        # ===========================================================

        log_weight_raw = (
            power * pt.log(s)
            - kernel_shift * s
        )

        # Numerical stabilisation
        log_weight_raw = (
            log_weight_raw
            - pt.max(log_weight_raw)
        )

        weight_raw = pt.exp(log_weight_raw)

        # -----------------------------------------------------------
        # Normalize so
        #
        #       sum_k a_k / s_k = 1
        #
        # and therefore
        #
        #       integral g(t) dt = rho
        #
        # exactly for the finite basis.
        # -----------------------------------------------------------

        integral_raw = pt.sum(weight_raw / s)

        basis_amplitude = (
            weight_raw / integral_raw
        )

        pm.Deterministic(
            "basis_amplitude",
            basis_amplitude,
            dims="basis"
        )

        # Useful normalized mixture proportions for interpretation
        basis_mass = (
            (basis_amplitude / s)
            / pt.sum(basis_amplitude / s)
        )

        pm.Deterministic(
            "basis_mass",
            basis_mass,
            dims="basis"
        )

        # ===========================================================
        # RECURSIVE HAWKES LIKELIHOOD
        #
        # For basis k:
        #
        # h_k(t) = sum_j exp[-s_k (t - t_j)]
        #
        # Because every component is exponential, each history
        # state can be updated recursively.
        #
        # Complexity:
        #
        #       O(N * K)
        #
        # instead of O(N^2).
        # ===========================================================

        initial_history = pt.zeros(
            (n_basis,),
            dtype="float64"
        )

        def step(
            dt_i,
            mu_i,
            h_prev,
            rho,
            rates,
            amplitudes
        ):

            # Decay each exponential history component
            exp_decay = pt.exp(-rates * dt_i)

            h_decay = h_prev * exp_decay

            # -------------------------------------------------------
            # Conditional intensity immediately before event i
            #
            # lambda_i =
            #     mu_i
            #     + rho * sum_k a_k h_k
            # -------------------------------------------------------

            excitation = rho * pt.sum(
                amplitudes * h_decay
            )

            lambda_i = mu_i + excitation

            lambda_i = pt.clip(
                lambda_i,
                1e-12,
                1e12
            )

            # -------------------------------------------------------
            # Integrated excitation over current interval
            #
            # Integral exp(-s t) dt
            # =
            # (1 - exp(-s dt)) / s
            # -------------------------------------------------------

            excitation_integral = rho * pt.sum(
                amplitudes
                * h_prev
                * (1.0 - exp_decay)
                / rates
            )

            integral_i = (
                mu_i * dt_i
                + excitation_integral
            )

            ll_i = (
                pt.log(lambda_i)
                - integral_i
            )

            # Current event enters history at age zero
            h_new = h_decay + 1.0

            return (
                h_new,
                ll_i,
                lambda_i,
                excitation
            )

        (
            h_seq,
            ll_seq,
            lambda_seq,
            excitation_seq
        ), _ = pytensor.scan(

            fn=step,

            sequences=[
                dt_data,
                mu
            ],

            outputs_info=[
                initial_history,
                None,
                None,
                None,
            ],

            non_sequences=[
                rho,
                s,
                basis_amplitude
            ],
        )

        # ===========================================================
        # STORE POINTWISE RESULTS
        # ===========================================================

        pm.Deterministic(
            "ll_interval",
            ll_seq,
            dims="interval"
        )

        pm.Deterministic(
            "lambda_event",
            lambda_seq,
            dims="interval"
        )

        pm.Deterministic(
            "excitation_event",
            excitation_seq,
            dims="interval"
        )

        # Storing the complete N x K history matrix can materially
        # inflate idata. Avoid it unless you genuinely need it.
        #
        # pm.Deterministic(
        #     "history_state",
        #     h_seq,
        #     dims=("interval", "basis")
        # )

        # ===========================================================
        # TOTAL LIKELIHOOD
        # ===========================================================

        pm.Potential(
            "powerlaw_hawkes_loglik",
            pt.sum(ll_seq)
        )

        # ===========================================================
        # SAMPLING
        # ===========================================================

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=min(chains, 4),
            target_accept=target_accept,
            random_seed=random_seed,
            init="adapt_diag",
            return_inferencedata=True,
            progressbar=True,
        )

    return (
        model,
        idata,
        X_mean,
        X_std,
        rate_grid
    )

def prepare_loglik_for_loo(idata, ll_var="ll_interval", obs_name="obs"):
    
    idata_loo = idata.copy()

    if ll_var not in idata_loo.posterior:
        raise KeyError(
            f"{ll_var!r} not found in idata.posterior. "
            f"Available variables include:\n"
            f"{list(idata_loo.posterior.data_vars)}"
        )

    ll = idata_loo.posterior[ll_var]

    print(f"{ll_var} dims :", ll.dims)
    print(f"{ll_var} shape:", ll.shape)

    # Basic numerical check
    if not np.all(np.isfinite(ll.values)):
        n_bad = np.sum(~np.isfinite(ll.values))
        raise ValueError(
            f"{ll_var} contains {n_bad} non-finite values."
        )

    # Remove an existing group if this function is rerun
    if hasattr(idata_loo, "log_likelihood"):
        del idata_loo.log_likelihood

    # Rename the observation dimension consistently if necessary
    obs_dim = ll.dims[-1]

    if obs_dim != "interval":
        print(f"Warning: expected final dimension 'interval', got {obs_dim!r}")

    ll_for_loo = ll.rename(obs_name)

    loglik_ds = xr.Dataset({
        obs_name: ll_for_loo
    })

    idata_loo.add_groups({
        "log_likelihood": loglik_ds
    })

    return idata_loo


def score_hawkes_future(
    idata,
    dt_train,
    dt_test,
    X_test,
    X_mean_train,
    X_std_train,
):
    dt_train = np.asarray(dt_train, dtype=float)
    dt_test = np.asarray(dt_test, dtype=float)
    X_test = np.asarray(X_test, dtype=float)

    Xs_test = (
        (X_test - X_mean_train)
        / X_std_train
    )

    # -----------------------------------------------------------
    # Posterior draws
    # -----------------------------------------------------------

    intercept = (
        idata.posterior["intercept"]
        .stack(sample=("chain", "draw"))
        .values
    )

    beta_x = (
        idata.posterior["beta_x"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "feature")
        .values
    )

    rho = (
        idata.posterior["rho"]
        .stack(sample=("chain", "draw"))
        .values
    )

    decay = (
        idata.posterior["decay"]
        .stack(sample=("chain", "draw"))
        .values
    )

    S = len(rho)
    N_test = len(dt_test)

    # -----------------------------------------------------------
    # Reconstruct history state at end of training period
    #
    # h_new = h_prev exp(-decay * dt) + 1
    # -----------------------------------------------------------

    h = np.zeros(S, dtype=float)

    for dti in dt_train:
        h = h * np.exp(-decay * dti) + 1.0

    # -----------------------------------------------------------
    # Covariate-dependent future baseline intensity
    # -----------------------------------------------------------

    eta_test = (
        intercept[:, None]
        + beta_x @ Xs_test.T
    )

    mu_test = np.exp(eta_test)

    ll = np.empty((S, N_test), dtype=float)

    # -----------------------------------------------------------
    # Sequential future prediction
    # -----------------------------------------------------------

    for i, dti in enumerate(dt_test):

        e = np.exp(-decay * dti)

        # history immediately before event
        h_decay = h * e

        lambda_i = (
            mu_test[:, i]
            + rho * decay * h_decay
        )

        lambda_i = np.clip(
            lambda_i,
            1e-300,
            None
        )

        integral_i = (
            mu_test[:, i] * dti
            + rho * h * (1.0 - e)
        )

        ll[:, i] = (
            np.log(lambda_i)
            - integral_i
        )

        # only now does the realised test event enter history
        h = h_decay + 1.0

    lpd_i = (
        logsumexp(ll, axis=0)
        - np.log(S)
    )

    return lpd_i


THRESHOLDS = [600, 1800, 2640,3036]


def summarize_future_scores(
    dt_test,
    lpd_ip,
    lpd_hawkes,
):
    rows = []

    # Full future window
    mask_all = np.ones(len(dt_test), dtype=bool)

    eval_sets = [
        ("Full", mask_all)
    ]

    for u in THRESHOLDS:
        eval_sets.append(
            (f">{u}s", dt_test > u)
        )

    for label, mask in eval_sets:

        n = int(mask.sum())

        ip_score = float(lpd_ip[mask].sum())
        hawkes_score = float(
            lpd_hawkes[mask].sum()
        )

        rows.append({
            "region": label,
            "n": n,
            "IP_covariates": ip_score,
            "Hawkes_covariates": hawkes_score,
            "difference_Hawkes_minus_IP":
                hawkes_score - ip_score,
            "winner":
                "Hawkes + covariates"
                if hawkes_score > ip_score
                else "IP + covariates",
        })

    return rows


def fit_ip_covariates(
    dt,
    X,
    feature_names=None,
    draws=600,
    tune=600,
    chains=2,
    target_accept=0.95,
    random_seed=42,
):
    dt = np.asarray(dt, dtype=float)
    X = np.asarray(X, dtype=float)

    if dt.ndim != 1:
        raise ValueError("dt must be one-dimensional.")

    if X.ndim != 2:
        raise ValueError("X must be two-dimensional.")

    if len(dt) != X.shape[0]:
        raise ValueError("dt and X must have the same number of rows.")

    if np.any(dt <= 0):
        raise ValueError("All dt values must be positive.")

    if np.any(~np.isfinite(dt)):
        raise ValueError("dt contains non-finite values.")

    if np.any(~np.isfinite(X)):
        raise ValueError("X contains non-finite values.")

    N, p = X.shape

    if feature_names is None:
        feature_names = [f"x{i}" for i in range(p)]

    if len(feature_names) != p:
        raise ValueError(
            "feature_names must have length equal to X.shape[1]."
        )

    # -------------------------------------------------------
    # Standardize using TRAINING WINDOW ONLY
    # -------------------------------------------------------
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0, ddof=0)
    X_std = np.where(X_std == 0.0, 1.0, X_std)

    Xs = (X - X_mean) / X_std

    coords = {
        "interval": np.arange(N),
        "feat": feature_names,
    }

    with pm.Model(coords=coords) as model:

        dt_c = pm.Data(
            "dt",
            dt,
            dims="interval"
        )

        X_c = pm.Data(
            "X",
            Xs,
            dims=("interval", "feat")
        )

        # Same priors as original model
        beta0 = pm.Normal(
            "beta0",
            mu=0.0,
            sigma=1.0
        )

        b = pm.Normal(
            "b",
            mu=0.0,
            sigma=1.0,
            dims="feat"
        )

        lin = pm.math.clip(
            beta0 + pm.math.dot(X_c, b),
            -40.0,
            40.0
        )

        lam = pm.math.exp(lin)

        ll_i = (
            pm.math.log(lam)
            - lam * dt_c
        )

        pm.Deterministic(
            "ll_interval",
            ll_i,
            dims="interval"
        )

        pm.Potential(
            "sum_ll",
            ll_i.sum()
        )

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=min(chains, 4),
            target_accept=target_accept,
            return_inferencedata=True,
            init="adapt_diag",
            random_seed=random_seed,
        )

    # Expose pointwise likelihood for ArviZ
    try:
        idata.add_groups({
            "log_likelihood": {
                "interval": idata.posterior["ll_interval"]
            }
        })
    except ValueError:
        idata.log_likelihood["interval"] = (
            idata.posterior["ll_interval"]
        )

    return model, idata, X_mean, X_std
 

def score_ip_future(
    idata,
    dt_test,
    X_test,
    X_mean,
    X_std,
):
    dt_test = np.asarray(dt_test, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    X_mean = np.asarray(X_mean, dtype=float)
    X_std = np.asarray(X_std, dtype=float)

    if dt_test.ndim != 1:
        raise ValueError("dt_test must be one-dimensional.")

    if X_test.ndim != 2:
        raise ValueError("X_test must be two-dimensional.")

    if len(dt_test) != X_test.shape[0]:
        raise ValueError(
            "dt_test and X_test must have the same number of rows."
        )

    if np.any(dt_test <= 0):
        raise ValueError("All dt_test values must be positive.")

    if np.any(~np.isfinite(dt_test)):
        raise ValueError("dt_test contains non-finite values.")

    if np.any(~np.isfinite(X_test)):
        raise ValueError("X_test contains non-finite values.")

    # -------------------------------------------------------
    # Standardize TEST covariates using TRAINING statistics
    # -------------------------------------------------------
    Xs_test = (X_test - X_mean) / X_std

    # -------------------------------------------------------
    # Posterior draws
    # -------------------------------------------------------
    beta0 = (
        idata.posterior["beta0"]
        .stack(sample=("chain", "draw"))
        .values
    )

    b = (
        idata.posterior["b"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "feat")
        .values
    )

    # -------------------------------------------------------
    # Linear predictor
    # shape: posterior samples x test observations
    # -------------------------------------------------------
    eta = beta0[:, None] + b @ Xs_test.T

    eta = np.clip(
        eta,
        -40.0,
        40.0
    )

    lam = np.exp(eta)

    # -------------------------------------------------------
    # Pointwise held-out log likelihood
    # -------------------------------------------------------
    ll = (
        np.log(lam)
        - lam * dt_test[None, :]
    )

    # -------------------------------------------------------
    # Integrate over posterior uncertainty
    # -------------------------------------------------------
    lpd_i = (
        logsumexp(ll, axis=0)
        - np.log(ll.shape[0])
    )

    return lpd_i


def summarize_future_scores_with_uncertainty(
    dt_test,
    lpd_ip,
    lpd_hawkes,
    thresholds=(600, 1800, 2640,3036),
):
    rows = []

    eval_sets = [("Full", np.ones(len(dt_test), dtype=bool))]

    for u in thresholds:
        eval_sets.append((f">{u}s", dt_test > u))

    for label, mask in eval_sets:

        ip_i = np.asarray(lpd_ip)[mask]
        hawkes_i = np.asarray(lpd_hawkes)[mask]

        diff_i = hawkes_i - ip_i
        n = len(diff_i)

        total_ip = ip_i.sum()
        total_hawkes = hawkes_i.sum()
        total_diff = diff_i.sum()

        mean_ip = ip_i.mean()
        mean_hawkes = hawkes_i.mean()
        mean_diff = diff_i.mean()

        if n > 1:
            se_mean_diff = diff_i.std(ddof=1) / np.sqrt(n)
            se_total_diff = diff_i.std(ddof=1) * np.sqrt(n)
        else:
            se_mean_diff = np.nan
            se_total_diff = np.nan

        rows.append({
            "region": label,
            "n": n,

            "IP_total_LPD": total_ip,
            "Hawkes_total_LPD": total_hawkes,
            "total_diff_H_minus_IP": total_diff,

            "IP_mean_LPD": mean_ip,
            "Hawkes_mean_LPD": mean_hawkes,
            "mean_diff_H_minus_IP": mean_diff,

            "SE_mean_diff": se_mean_diff,
            "SE_total_diff": se_total_diff,

            "z_like_ratio": (
                total_diff / se_total_diff
                if np.isfinite(se_total_diff) and se_total_diff > 0
                else np.nan
            ),

            "winner": (
                "Hawkes + covariates"
                if total_diff > 0
                else "IP + covariates"
            ),
        })

    return rows
def paired_block_bootstrap_ci(
    score_a,
    score_b,
    block_length=50,
    n_boot=5000,
    ci=0.95,
    random_seed=42,
):
    
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)

    if a.shape != b.shape:
        raise ValueError(
            "score_a and score_b must have identical shapes."
        )

    if a.ndim != 1:
        raise ValueError(
            "Pointwise scores must be one-dimensional."
        )

    d = a - b
    N = len(d)

    if N % block_length != 0:
        raise ValueError(
            f"For the fast bootstrap, N={N} must be divisible "
            f"by block_length={block_length}."
        )

    rng = np.random.default_rng(random_seed)

    # --------------------------------------------------------
    # Precompute sums for every possible moving block.
    #
    # This makes 5000 bootstrap replications extremely cheap.
    # --------------------------------------------------------

    csum = np.concatenate([
        [0.0],
        np.cumsum(d)
    ])

    block_sums = (
        csum[block_length:]
        -
        csum[:-block_length]
    )

    n_possible_blocks = len(block_sums)
    n_blocks = N // block_length

    observed_diff = d.mean()

    boot_diff = np.empty(n_boot)

    for j in range(n_boot):

        starts = rng.integers(
            0,
            n_possible_blocks,
            size=n_blocks
        )

        boot_diff[j] = (
            block_sums[starts].sum()
            / N
        )

    alpha = 1.0 - ci

    lower, upper = np.quantile(
        boot_diff,
        [
            alpha / 2.0,
            1.0 - alpha / 2.0
        ]
    )

    return {
        "score_A": a.mean(),
        "score_B": b.mean(),
        "difference_A_minus_B": observed_diff,
        "bootstrap_SE": boot_diff.std(ddof=1),
        "CI_lower": lower,
        "CI_upper": upper,
        "n_boot": n_boot,
        "block_length": block_length,
    }


def _tail_scores_all_thresholds(y_obs, taus, S_tau, tail_thresholds):
    
    taus = np.asarray(taus, dtype=float)
    S_tau = np.asarray(S_tau, dtype=float)
    thresholds = np.asarray(tail_thresholds, dtype=float)

    indicator = (y_obs > taus).astype(float)
    squared_error = (S_tau - indicator) ** 2

    # Segment areas for ordinary trapezoidal integration.
    segment_areas = (
        0.5 * (squared_error[:-1] + squared_error[1:])
        * np.diff(taus)
    )

    # tail_area_from[j] = sum of full trapezoid areas from j onward.
    tail_area_from = np.concatenate([
        np.cumsum(segment_areas[::-1])[::-1],
        np.array([0.0]),
    ])

    # First tau value included by weight I(tau >= u).
    first_idx = np.searchsorted(taus, thresholds, side="left")

    tw_scores = np.zeros(len(thresholds), dtype=float)

    valid = first_idx < len(taus)
    j = first_idx[valid]

    # Full trapezoids from j onward.
    tw_scores[valid] = tail_area_from[j]

    # Exact boundary contribution created by np.trapz when the
    # preceding value is zeroed by the indicator weight.
    boundary = j > 0
    if np.any(boundary):
        j_boundary = j[boundary]
        tw_scores[np.where(valid)[0][boundary]] += (
            0.5
            * squared_error[j_boundary]
            * (taus[j_boundary] - taus[j_boundary - 1])
        )

    # Same exceedance probability interpolation as before.
    S_u = np.interp(thresholds, taus, S_tau)
    brier_scores = (
        ((y_obs > thresholds).astype(float) - S_u) ** 2
    )

    return tw_scores, brier_scores

def score_hawkes_model_sensitivity(
    dt,
    event_times,
    id_hawkes,
    taus,
    tail_thresholds,
    thin=10,
):
    dt = np.asarray(dt, dtype=float)
    event_times = np.asarray(event_times, dtype=float)
    taus = np.asarray(taus, dtype=float)
    tail_thresholds = np.asarray(tail_thresholds, dtype=int)

    crps_vals = []
    twcrps_vals = {int(u): [] for u in tail_thresholds}
    bs_vals = {int(u): [] for u in tail_thresholds}

    for i, y in enumerate(dt):

        S_draws, _ = S_hawkes(
            id_hawkes=id_hawkes,
            times=event_times[:i],
            taus=taus,
        )

        S_draws = S_draws[::thin]

        if S_draws.size == 0:
            raise ValueError(
                "score_hawkes_model_sensitivity: "
                "no draws left after thinning."
            )

        S_tau = np.clip(np.nanmean(S_draws, axis=0), 0.0, 1.0)

        crps_vals.append(
            crps_from_survival(y, taus, S_tau)
        )

        tw_scores, brier_scores = _tail_scores_all_thresholds(
            y_obs=y,
            taus=taus,
            S_tau=S_tau,
            tail_thresholds=tail_thresholds,
        )

        for position, u in enumerate(tail_thresholds):
            twcrps_vals[int(u)].append(tw_scores[position])
            bs_vals[int(u)].append(brier_scores[position])

    summary = summarize_scores(
        "Hawkes",
        crps_vals,
        twcrps_vals,
        bs_vals,
    )

    pointwise = {
        "CRPS": np.asarray(crps_vals, dtype=float),
        "twCRPS": {
            int(u): np.asarray(twcrps_vals[int(u)], dtype=float)
            for u in tail_thresholds
        },
        "Brier": {
            int(u): np.asarray(bs_vals[int(u)], dtype=float)
            for u in tail_thresholds
        },
    }

    return summary, pointwise

def score_hawkes_covariates_model_sensitivity(
    dt,
    X,
    event_times,
    id_hawkes_exog,
    taus,
    tail_thresholds,
):
    dt = np.asarray(dt, dtype=float)
    X = np.asarray(X, dtype=float)
    event_times = np.asarray(event_times, dtype=float)
    taus = np.asarray(taus, dtype=float)
    tail_thresholds = np.asarray(tail_thresholds, dtype=int)

    crps_vals = []
    twcrps_vals = {int(u): [] for u in tail_thresholds}
    bs_vals = {int(u): [] for u in tail_thresholds}

    for i, y in enumerate(dt):

        S_draws, _ = S_hawkes_exog_new(
            id_hawkes_exog=id_hawkes_exog,
            X_row=X[i],
            times=event_times[:i],
            taus=taus,
        )

        S_tau = np.clip(S_draws.mean(axis=0), 0.0, 1.0)

        crps_vals.append(
            crps_from_survival(y, taus, S_tau)
        )

        tw_scores, brier_scores = _tail_scores_all_thresholds(
            y_obs=y,
            taus=taus,
            S_tau=S_tau,
            tail_thresholds=tail_thresholds,
        )

        for position, u in enumerate(tail_thresholds):
            twcrps_vals[int(u)].append(tw_scores[position])
            bs_vals[int(u)].append(brier_scores[position])

    summary = summarize_scores(
        "Hawkes + covariates",
        crps_vals,
        twcrps_vals,
        bs_vals,
    )

    pointwise = {
        "CRPS": np.asarray(crps_vals, dtype=float),
        "twCRPS": {
            int(u): np.asarray(twcrps_vals[int(u)], dtype=float)
            for u in tail_thresholds
        },
        "Brier": {
            int(u): np.asarray(bs_vals[int(u)], dtype=float)
            for u in tail_thresholds
        },
    }

    return summary, pointwise


