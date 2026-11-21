"""Independent high-precision Monte Carlo check of cutoffs on Belgian ArNet2 records."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from patient_thresholds import cp_limits

REPEATS = 20000
STREAM = "holter-review-independent-mc-20000-v1"
BLOCK = 500


def uniform_subsets(rng, population, k, size):
    # Uniform independent sampling of ordered tuples, retaining only distinct
    # values within each tuple. Rejection leaves every ordered permissible
    # tuple with equal probability. Use uniform random priorities for strata
    # near census, where proposal collisions would be frequent.
    if k * k > population and population <= 512:
        u = rng.random((size, population))
        return np.argpartition(u, kth=k - 1, axis=1)[:, :k]
    result = np.empty((size, k), dtype=np.int64)
    pending = np.arange(size)
    while pending.size:
        candidate = rng.integers(population, size=(pending.size, k))
        ok = np.all(np.diff(np.sort(candidate, axis=1), axis=1) != 0, axis=1)
        result[pending[ok]] = candidate[ok]
        pending = pending[~ok]
    return result


def simulate(y, strata, prediction, prior_variance, sizes, model, seed):
    counts = np.bincount(strata, minlength=6)
    assert np.all(sizes <= counts) and sizes.sum() == 20
    groups = [np.flatnonzero(strata == h) for h in range(len(counts))]
    weights = counts / len(y)
    residual = y - prediction
    truth = y.mean()
    census = np.array_equal(counts, sizes)
    fixed_radius = 0. if census else model["radius"]
    rng = np.random.default_rng(seed)
    covered = np.zeros(4, dtype=np.int64)
    for start in range(0, REPEATS, BLOCK):
        n = min(BLOCK, REPEATS - start)
        center = np.full(n, prediction.mean())
        v_obs = np.zeros(n)
        v_fit = 0.
        for h, members in enumerate(groups):
            if members.size == 0:
                continue
            if sizes[h] == members.size:
                center += weights[h] * residual[members].mean()
                continue
            chosen = uniform_subsets(rng, members.size, sizes[h], n)
            values = residual[members][chosen]
            center += weights[h] * values.mean(axis=1)
            factor = weights[h] ** 2 * (1. - sizes[h] / members.size) / sizes[h]
            v_obs += factor * values.var(axis=1, ddof=1)
            v_fit += factor * prior_variance[h]
        ctr = np.clip(center,0,1)
        scale = np.maximum(.001, np.sqrt(.5*(v_obs+v_fit)))
        adaptive_radius = 0. if census else model["multiplier"]*scale
        # The reference burden and center are in [0,1]. Endpoint intersection
        # with that range does not affect interval membership.
        f = np.abs(ctr - truth) <= fixed_radius + 1e-12
        a = np.abs(ctr - truth) <= adaptive_radius + 1e-12
        covered += np.bincount(2*f.astype(int)+a.astype(int),minlength=4)
    assert covered.sum() == REPEATS
    return covered


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    root = args.study_root.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0,str(root))
    import study
    from pilot_estimators import stable_seed, levels, allocate

    model=json.loads((root / "results/model_ArNet2.json").read_text())
    p, variance = np.array(model["p"]), np.array(model["residual_variance"])
    records=[]
    for rec,pt,y,raw,st in study.read_inputs("ArNet2","IRIDIA"):
        k=allocate(np.bincount(st,minlength=6),np.ones(6),20)
        seed=stable_seed(STREAM,"IRIDIA","ArNet2",rec)
        none,only_a,only_f,both=simulate(y,st,p[levels(raw)],variance,k,model,seed)
        records.append(dict(record=rec,patient=pt,draws=REPEATS,seed=str(seed),
                            both_covered=both,only_fixed_covered=only_f,
                            only_adaptive_covered=only_a,neither_covered=none))
    records=pd.DataFrame(records)
    assert len(records)==167
    records.to_csv(args.out / 'independent_mc_records.csv',index=False)
    original=pd.read_csv(root / 'results/patients.csv').query(
        "detector=='ArNet2' and cohort=='IRIDIA'"
    ).pivot(index='patient',columns='method',values='coverage')
    patients=[]
    for pt,df in records.groupby('patient'):
        r=len(df)
        row=dict(patient=pt,records=r,draws_per_record=REPEATS)
        for label,key,only in [('fixed','fixed_radius','only_fixed_covered'),
                               ('adaptive','scaled_radius','only_adaptive_covered')]:
            hits=df.both_covered+df[only]
            limits=np.array([cp_limits(k,REPEATS,.05/r) for k in hits])
            row[label+'_primary_pct']=100*original.loc[pt,key]
            row[label+'_independent_pct']=100*hits.sum()/(r*REPEATS)
            row[label+'_mc95_low_pct'],row[label+'_mc95_high_pct']=100*limits.mean(axis=0)
        patients.append(row)
    patients=pd.DataFrame(patients)
    assert len(patients)==152
    patients.to_csv(args.out / 'independent_mc_patients.csv',index=False,float_format='%.10f')
    cross=[]
    for t in [70,75,80,85,90]:
        f,a=(patients[x+'_independent_pct']<t for x in ['fixed','adaptive'])
        cross.append(dict(threshold_pct=t,below_fixed=f.sum(),below_adaptive=a.sum(),
                          both_below=(f & a).sum(),only_fixed_below=(f & ~a).sum(),
                          only_adaptive_below=(~f & a).sum(),both_at_or_above=(~f & ~a).sum()))
    cross=pd.DataFrame(cross)
    cross.to_csv(args.out / 'independent_mc_thresholds.csv',index=False)
    print(cross.to_string(index=False))
    for label in ['fixed','adaptive']:
        orig=patients[label+'_primary_pct']<80
        new=patients[label+'_independent_pct']<80
        overlap=(patients[label+'_mc95_low_pct']<=80)&(patients[label+'_mc95_high_pct']>=80)
        print(label,'IDs changing sides at 80:',patients.loc[orig!=new,'patient'].tolist(),
              'including 80 in independent interval:',patients.loc[overlap,'patient'].tolist())
    print(patients.query("fixed_primary_pct>=80 and adaptive_primary_pct<80").to_string(index=False))

if __name__=='__main__':
    main()
