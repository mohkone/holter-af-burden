"""Sensitivity of the Belgian coverage classifications to cutoffs and Monte Carlo draws."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import beta

DETECTORS = ("ArNet2", "COSEn")
METHODS = ("fixed_radius", "scaled_radius")
DRAWS = 1000
ALPHA = 0.05
CUTOFFS = (70, 75, 80, 85, 90)


def cp_limits(successes, n, alpha):
    """Equal-tailed Clopper-Pearson interval for a binomial probability."""
    lo = 0.0 if successes == 0 else beta.ppf(alpha / 2, successes, n - successes + 1)
    hi = 1.0 if successes == n else beta.ppf(1 - alpha / 2, successes + 1, n - successes)
    return lo, hi


def per_patient(study_path):
    all_rows = []
    stored_patients = pd.read_csv(study_path / "results/patients.csv")
    for detector in DETECTORS:
        source = pd.read_csv(study_path / "results" / f"records_IRIDIA_{detector}.csv")
        source = source[source.method.isin(METHODS)]
        for (patient, method), rows in source.groupby(["patient", "method"]):
            r = len(rows)
            counts = np.rint(rows.coverage * DRAWS).astype(int)
            np.testing.assert_allclose(rows.coverage * DRAWS, counts, rtol=0, atol=1e-9)
            ci = [cp_limits(k, DRAWS, ALPHA / r) for k in counts]
            lo, hi = np.mean(ci, axis=0)
            coverage = counts.sum() / (r * DRAWS)
            match = stored_patients.query(
                "detector==@detector and method==@method and cohort=='IRIDIA' and patient==@patient"
            )
            assert len(match) == 1 and abs(match.coverage.iloc[0] - coverage) < 1e-12
            all_rows.append(dict(patient=patient, detector=detector, method=method,
                                 records=r, record_ids="|".join(rows.record),
                                 covered_draws=counts.sum(), draws_per_record=DRAWS,
                                 coverage_pct=100 * coverage,
                                 mc95_low_pct=100 * lo, mc95_high_pct=100 * hi))
    return pd.DataFrame(all_rows)


def threshold_table(patient):
    rows = []
    for detector, site in patient.groupby("detector", sort=False):
        fixed = site[site.method == METHODS[0]].set_index("patient").sort_index()
        adaptive = site[site.method == METHODS[1]].set_index("patient").sort_index()
        assert fixed.index.equals(adaptive.index) and len(fixed) == 152
        for threshold in CUTOFFS:
            def below(frame):
                # Compare integer counts so equality always belongs to the upper bin.
                return 100 * frame.covered_draws < threshold * frame.draws_per_record * frame.records
            f, a = below(fixed), below(adaptive)
            f_cross = (fixed.mc95_low_pct <= threshold) & (threshold <= fixed.mc95_high_pct)
            a_cross = (adaptive.mc95_low_pct <= threshold) & (threshold <= adaptive.mc95_high_pct)
            rows.append(dict(detector=detector, threshold_pct=threshold, patients=len(f),
                below_fixed=f.sum(), below_adaptive=a.sum(), both_below=(f & a).sum(),
                only_fixed_below=(f & ~a).sum(), only_adaptive_below=(~f & a).sum(),
                both_at_or_above=(~f & ~a).sum(),
                below_fixed_with_mc95_including_cutoff=(f & f_cross).sum(),
                below_adaptive_with_mc95_including_cutoff=(a & a_cross).sum(),
                at_or_above_fixed_with_mc95_including_cutoff=(~f & f_cross).sum(),
                at_or_above_adaptive_with_mc95_including_cutoff=(~a & a_cross).sum()))
    return pd.DataFrame(rows)


def crossing_checks(study_path, patient):
    """Regenerate paired interval outcomes, conditional on the same 1,000 random draws."""
    sys.path.insert(0, str(study_path))
    import study
    import pilot_estimators

    arnet = patient[patient.detector == "ArNet2"]
    fixed = arnet[arnet.method == METHODS[0]].set_index("patient")
    adaptive = arnet[arnet.method == METHODS[1]].set_index("patient")
    ids = set(fixed.index[(fixed.coverage_pct >= 80) & (adaptive.coverage_pct < 80)])
    assert len(ids) == 4
    model = json.loads((study_path / "results/model_ArNet2.json").read_text())
    prob, variance = np.array(model["p"]), np.array(model["residual_variance"])
    rows = []
    for record, person, y, raw, st in study.read_inputs("ArNet2", "IRIDIA"):
        if person not in ids:
            continue
        assert fixed.loc[person, "records"] == adaptive.loc[person, "records"] == 1
        p = prob[pilot_estimators.levels(raw)]
        seed = pilot_estimators.stable_seed("holter-interval", "IRIDIA", record)
        state = study.draw(y, st, p, variance, seed)
        bands = study.intervals(state, model)
        truth = y.mean()
        covered = {}
        for name in METHODS:
            lo, hi = bands[name]
            covered[name] = (lo <= truth + 1e-12) & (hi >= truth - 1e-12)
        f, a = (covered[name] for name in METHODS)
        assert f.sum() == fixed.loc[person, "covered_draws"]
        assert a.sum() == adaptive.loc[person, "covered_draws"]
        gain, loss = (~f & a).sum(), (f & ~a).sum()
        # Difference in coverage is the difference between the two mutually
        # exclusive discordance probabilities. Simultaneously bound them.
        gain_lo, gain_hi = cp_limits(gain, DRAWS, ALPHA / 2)
        loss_lo, loss_hi = cp_limits(loss, DRAWS, ALPHA / 2)
        row = dict(patient=person, record=record, paired_draws=DRAWS,
                   fixed_coverage_pct=100 * f.mean(),
                   fixed_mc95_low_pct=fixed.loc[person, "mc95_low_pct"],
                   fixed_mc95_high_pct=fixed.loc[person, "mc95_high_pct"],
                   adaptive_coverage_pct=100 * a.mean(),
                   adaptive_mc95_low_pct=adaptive.loc[person, "mc95_low_pct"],
                   adaptive_mc95_high_pct=adaptive.loc[person, "mc95_high_pct"],
                   both_cover=(f & a).sum(), only_fixed_covers=loss,
                   only_adaptive_covers=gain, neither_covers=(~f & ~a).sum(),
                   adaptive_minus_fixed_pp=100*(int(gain)-int(loss))/DRAWS,
                   paired_diff_mc95_low_pp=100*(gain_lo-loss_hi),
                   paired_diff_mc95_high_pp=100*(gain_hi-loss_lo))
        rows.append(row)
    assert len(rows) == len(ids)
    return pd.DataFrame(rows).sort_values("patient")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    root = args.study_root.resolve()
    args.out.mkdir(exist_ok=True, parents=True)
    patients = per_patient(root)
    assert len(patients) == 152 * len(METHODS) * len(DETECTORS)
    threshold = threshold_table(patients)
    crossings = crossing_checks(root, patients)
    for key, frame in [("patient_mc_intervals", patients),
                       ("threshold_sensitivity", threshold),
                       ("four_crossings_mc", crossings)]:
        frame.to_csv(args.out / (key + ".csv"), index=False, float_format="%.10f")
    print(threshold.to_string(index=False))
    print(crossings.to_string(index=False, float_format="%.2f"))


if __name__ == "__main__":
    main()
