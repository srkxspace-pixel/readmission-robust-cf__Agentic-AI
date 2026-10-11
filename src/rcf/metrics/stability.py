"""Stability metrics exactly as defined in the gap-source thesis (docs/M0_REPORT.md §1.2), quality metrics
(DESIGN D6) and the patient-level bootstrap. A CF table has columns [patient, source, cf] + raw features.

Every metric returns (patient_scores: Series indexed by group, units: array of unit-level values), so both
the thesis's patient-level means (Table 7) and its unit-level descriptive means (Tables 3-6) can be reported.
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def logit_matrix(models, df, cols):
    """Shifted logits of every CF row under every model (encode once; all models share the encoder)."""
    if df.empty:  # a method may find no CFs at all (e.g. ElliCE on a hard cohort)
        return pd.DataFrame(columns=[m.model_id for m in models], index=df.index, dtype=float)
    Z = models[0].encoder.transform(df[cols])
    return pd.DataFrame({m.model_id: m.logit(Z) for m in models}, index=df.index)


def deltas(cf, test, vary):
    """cf_f - x_f for each actionable feature (numeric), aligned to cf rows."""
    orig = test.loc[cf.patient, vary].to_numpy(float)
    return pd.DataFrame(cf[vary].to_numpy(float) - orig, columns=vary, index=cf.index)


def changed(cf, test, vary):
    """Boolean matrix: did the CF change feature f (works for numeric and categorical features)."""
    orig = test.loc[cf.patient, vary].astype(str).to_numpy()
    cur = cf[vary].astype(float).astype(int).astype(str).to_numpy() if cf[vary].dtypes.ne(object).all() \
        else cf[vary].astype(str).to_numpy()
    return pd.DataFrame(cur != orig, columns=vary, index=cf.index)


# ---- thesis metrics -------------------------------------------------------------------------------------

def cross_model_validity(cf, L, evaluators, group="patient"):
    """Eq. 1: V(x') = (1/K) sum_k 1[m_k(x') = 0] over the evaluator models. Patient = mean over its CFs."""
    v = (L[evaluators] < 0).mean(axis=1)
    return v.groupby(cf[group]).mean(), v.to_numpy()


def _cross_pairs(src):
    a, b = np.triu_indices(len(src), 1)
    keep = src[a] != src[b]
    return a[keep], b[keep]


def feature_set_consistency(cf, ch, group="patient", src="source", pair_ok=None):
    """Eqs. 2-4: Jaccard of changed-feature sets over all CF pairs from different sources; empty-empty = 1."""
    scores, units = {}, []
    for g, idx in cf.groupby(group).indices.items():
        s = cf[src].to_numpy()[idx]
        a, b = _cross_pairs(s)
        if pair_ok is not None:
            ok = np.array([pair_ok(s[i], s[j]) for i, j in zip(a, b)], bool)
            a, b = a[ok], b[ok]
        if not len(a):
            continue
        F = ch.to_numpy()[idx]
        inter, union = (F[a] & F[b]).sum(1), (F[a] | F[b]).sum(1)
        j = np.where(union == 0, 1.0, inter / np.maximum(union, 1))
        scores[g] = j.mean()
        units.extend(j)
    return pd.Series(scores, dtype=float), np.array(units)


def direction_consistency(cf, dl, group="patient", src="source"):
    """Eqs. 5-6: per (patient, feature) share of changes in the dominant direction, pooled over all CFs of the
    patient; a feature counts only if CFs from >= 2 distinct sources change it. Patient = mean over features."""
    scores, units = {}, []
    for g, idx in cf.groupby(group).indices.items():
        s, D = cf[src].to_numpy()[idx], dl.to_numpy()[idx]
        vals = []
        for f in range(D.shape[1]):
            nz = D[:, f] != 0
            if len(set(s[nz])) < 2:
                continue
            up, down = (D[nz, f] > 0).sum(), (D[nz, f] < 0).sum()
            vals.append(max(up, down) / (up + down))
        if vals:
            scores[g] = np.mean(vals)
            units.extend(vals)
    return pd.Series(scores, dtype=float), np.array(units)


def magnitude_consistency(cf, dl, width, group="patient", src="source"):
    """Eqs. 7-8: G = max(0, 1 - |Δa - Δb|), Δ = |change| / (u_f - l_f), over cross-source pairs changing f in the
    same direction. Patient-feature = mean over pairs; patient = mean over eligible features."""
    scores, units = {}, []
    for g, idx in cf.groupby(group).indices.items():
        s, D = cf[src].to_numpy()[idx], dl.to_numpy()[idx]
        a, b = _cross_pairs(s)
        vals = []
        for f in range(D.shape[1]):
            da, db = D[a, f], D[b, f]
            ok = (da != 0) & (db != 0) & (np.sign(da) == np.sign(db))
            if ok.any():
                G = np.maximum(0, 1 - np.abs(np.abs(da[ok]) - np.abs(db[ok])) / width[f])
                vals.append(G.mean())
                units.extend(G)
        if vals:
            scores[g] = np.mean(vals)
    return pd.Series(scores, dtype=float), np.array(units)


# ---- quality --------------------------------------------------------------------------------------------

def proximity(cf, dl, width, group="patient"):
    """L1 distance over actionable features, each normalised by its permitted range."""
    p = (dl.abs() / width).sum(axis=1)
    return p.groupby(cf[group]).mean(), p.to_numpy()


def sparsity(cf, ch, group="patient"):
    s = ch.sum(axis=1)
    return s.groupby(cf[group]).mean(), s.to_numpy()


def plausibility(cf, models, train, cols, group="patient"):
    """Euclidean distance (encoded space) from each CF to the nearest training patient that its source model
    predicts low-risk. Lower = more realistic."""
    by_id = {m.model_id: m for m in models}
    enc = models[0].encoder
    Ztr = enc.transform(train[cols])
    dist = pd.Series(np.nan, index=cf.index)
    for sid, rows in cf.groupby("source").groups.items():
        m = by_id[sid]
        nn_ = NearestNeighbors(n_neighbors=1).fit(Ztr[m.logit(Ztr) < 0])
        dist[rows] = nn_.kneighbors(enc.transform(cf.loc[rows, cols]))[0][:, 0]
    return dist.groupby(cf[group]).mean(), dist.to_numpy()


# ---- uncertainty ----------------------------------------------------------------------------------------

def bootstrap(scores, n=5000, seed=0):
    """Patient-level percentile bootstrap of the mean (thesis §3.5). Returns mean, se, 95% CI, n patients."""
    x = np.asarray(pd.Series(scores).dropna(), float)
    if not len(x):
        return dict(mean=np.nan, se=np.nan, lo=np.nan, hi=np.nan, n=0)
    boots = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(axis=1)
    return dict(mean=x.mean(), se=boots.std(ddof=1), lo=np.percentile(boots, 2.5), hi=np.percentile(boots, 97.5),
                n=len(x))


def paired_diff(a, b, n=5000, seed=0):
    """Bootstrap CI of mean(a - b) over patients present in both (paired comparison of two methods)."""
    both = pd.concat([pd.Series(a, dtype=float), pd.Series(b, dtype=float)], axis=1).dropna()
    return bootstrap(both.iloc[:, 0] - both.iloc[:, 1], n, seed)


if __name__ == "__main__":
    # Self-check on a hand-made example (thesis definitions).
    cf = pd.DataFrame({"patient": [1, 1, 1, 1], "source": ["a", "a", "b", "b"]})
    ch = pd.DataFrame({"x": [1, 0, 1, 0], "y": [0, 1, 1, 0]}, dtype=bool)
    fs, u = feature_set_consistency(cf, ch)
    # cross pairs: (0,2) {x}vs{x,y}=1/2, (0,3) {x}vs{}=0, (1,2) {y}vs{x,y}=1/2, (1,3) {y}vs{}=0
    assert np.isclose(fs[1], 0.25) and len(u) == 4
    dl = pd.DataFrame({"x": [2.0, 0, -1.0, 0], "y": [0, 1.0, 3.0, 0]})
    dc, _ = direction_consistency(cf, dl)
    assert np.isclose(dc[1], (0.5 + 1.0) / 2)   # x: one up one down -> 0.5; y: two ups from 2 sources -> 1
    mc, _ = magnitude_consistency(cf, dl, np.array([10.0, 10.0]))
    assert np.isclose(mc[1], 1 - 0.2)            # only y pair (1,2) same direction: 1 - |1-3|/10
    b = bootstrap(np.ones(10))
    assert b["mean"] == 1 and b["lo"] == 1
    print("stability self-check ok")
