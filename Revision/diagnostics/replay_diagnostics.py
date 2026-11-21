"""Replay frozen Holter draws and describe errors, tiny widths, and a variance-ratio flag.

python replay_diagnostics.py --study /path/to/holter_intervals --output /path/to/results
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def weighted_auc(outcome, score, weight):
    """Weighted Mann-Whitney AUC, scores tied exactly receive half credit."""
    order = np.argsort(score, kind="stable")
    y, x, w = outcome[order], score[order], weight[order]
    groups = np.r_[0, 1 + np.flatnonzero(np.diff(x) != 0)]
    pos = np.add.reduceat(w * y, groups)
    neg = np.add.reduceat(w * (~y), groups)
    if pos.sum() == 0 or neg.sum() == 0:
        return np.nan
    return (pos * (np.cumsum(neg) - .5 * neg)).sum() / (pos.sum() * neg.sum())


def tiny_summary(frame):
    rows = []
    for (det, method), a in frame.groupby(["detector", "method"]):
        for band, include in [
            ("all", a.truth >= 0), ("zero", a.truth == 0),
            ("positive", a.truth > 0), ("interior", (a.truth > 0) & (a.truth < 1)),
            ("full", a.truth == 1)
        ]:
            b = a.loc[include]
            by_patient = b.groupby("patient")[["empty", "zero", "tiny"]].mean()
            rows.append(dict(detector=det, method=method, burden=band,
                             patients=len(by_patient), records=len(b),
                             pct_empty=100*by_patient["empty"].mean(),
                             pct_zero=100*by_patient["zero"].mean(), pct_below_0_1pp=100*by_patient.tiny.mean(),
                             draws_below_0_1pp=int(np.rint(1000*b.tiny.sum()))))
    return pd.DataFrame(rows)


def ratio_analysis(panel):
    rows = []
    for (cohort, det), b in panel.groupby(["cohort", "detector"]):
        # Distinct sets of draws for the predictor and the patient-level low-coverage target.
        b = b.copy()
        b["half"] = (b.draw >= 500).astype(int)
        cover = (1 - b.groupby(["patient", "record", "half"]).miss.mean()).groupby(["patient","half"]).mean()
        other_half = pd.MultiIndex.from_arrays([b.patient,1-b.half])
        b["other_half_low"] = cover.reindex(other_half).to_numpy() < .8
        w = (1 / b.groupby("patient").draw.transform("size")).to_numpy()
        ratio = b.ratio.to_numpy()
        for label, target in [("current_interval_misses", b.miss.to_numpy()),
                              ("patient_other_half_coverage_below_80", b.other_half_low.to_numpy())]:
            auc = weighted_auc(target, -ratio, w)
            for cutoff in [.25, 1.]:
                flag = ratio <= cutoff
                rows.append(dict(cohort=cohort, detector=det, target=label,
                    flag_ratio_at_most=cutoff, patients=b.patient.nunique(), records=b.record.nunique(),
                    flag_fraction=np.average(flag,weights=w), prevalence=np.average(target,weights=w),
                    sensitivity=np.average(flag[target],weights=w[target]),
                    false_positive_rate=np.average(flag[~target],weights=w[~target]),
                    positive_predictive_value=np.average(target[flag],weights=w[flag]),
                    negative_predictive_value=np.average(~target[~flag],weights=w[~flag]),
                    auc_lower_ratio_more_risk=auc))
    frame = pd.DataFrame(rows)
    # A complementary flag R>c is often easier to interpret at c=1.
    for output, opposite in [
        ("gt_flag_fraction","flag_fraction"), ("gt_sensitivity","sensitivity"),
        ("gt_false_positive_rate","false_positive_rate"),
        ("gt_positive_predictive_value","negative_predictive_value"),
        ("gt_negative_predictive_value","positive_predictive_value"),
        ("auc_higher_ratio_more_risk","auc_lower_ratio_more_risk")
    ]:
        frame[output] = 1 - frame[opposite]
    return frame


def describe_case_flags(case_draws):
    frame = case_draws.loc[case_draws["method"] == "scaled_radius"].copy()
    frame["ratio_above_one"] = frame.v_obs_over_v_fit > 1
    return (frame.groupby(["patient","record","ratio_above_one"])
            .agg(draws=("covered","size"),coverage=("covered","mean"),
                 mean_width_pp=("width_pp","mean"),median_center_pp=("center_pp","median"))
            .reset_index())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    sys.path.insert(0,str(args.study.resolve()))
    import study
    out=args.output
    out.mkdir(exist_ok=True, parents=True)
    root=args.study
    patients=pd.read_csv(root/"results/patients.csv",dtype={"patient":str})
    relevant=patients.query("cohort == 'IRIDIA' and detector == 'ArNet2' and method == 'scaled_radius' and coverage < .5")
    case_ids=set(relevant.patient)
    assert len(case_ids)==5
    widths, cases, case_draws, flags = [], [], [], []
    checks=0
    for co in ("IRIDIA","SHDB"):
        for det in ("ArNet2","COSEn"):
            model=json.loads((root/f"results/model_{det}.json").read_text())
            fitted=np.array(model["residual_variance"])
            lookup=np.array(model["p"])
            old=pd.read_csv(root/f"results/records_{co}_{det}.csv",
                            dtype={"record":str,"patient":str}).set_index(["record","method"])
            for rec, pt, y, raw, strata in study.read_inputs(det,co):
                p=lookup[study.levels(raw)]
                seed=study.stable_seed("holter-interval",co,rec)
                st=study.draw(y,strata,p,fitted,seed)
                ctr=np.clip(st["mu"],0,1)
                truth=y.mean()
                n,k=st["counts"],st["sizes"]
                active=n>k
                vfit=np.sum((n[active]/n.sum())**2*(1-k[active]/n[active])*fitted[active]/k[active])
                assert vfit>0
                np.testing.assert_allclose(st["scale"],np.maximum(.001,np.sqrt((st["var"]+vfit)/2)),atol=1e-14)
                intervals=study.intervals(st,model)
                for method,(lower,upper) in intervals.items():
                    l,u=np.broadcast_arrays(lower,upper)
                    width=np.maximum(0,u-l)
                    over=truth<l-1e-12
                    under=truth>u+1e-12
                    covered=~(over|under)
                    emp=(l>u+1e-12).mean()
                    zero=(width<=1e-12).mean()
                    ref=old.loc[(rec,method)]
                    np.testing.assert_allclose([covered.mean(),100*width.mean(),over.mean(),under.mean(),emp,zero],
                        ref[["coverage","width_pp","miss_low","miss_high","empty","zero"]].astype(float),rtol=1e-11,atol=2e-12)
                    checks+=1
                    if co=="IRIDIA":
                        widths.append(dict(detector=det,method=method,patient=pt,record=rec,truth=truth,
                                           empty=emp,zero=zero,tiny=(width<.001).mean()))
                    if method=="scaled_radius":
                        flags.append(pd.DataFrame(dict(cohort=co,detector=det,patient=pt,record=rec,
                            draw=np.arange(len(ctr)),ratio=st["var"]/vfit,miss=~covered)))
                    if co=="IRIDIA" and det=="ArNet2" and pt in case_ids and method in ("scaled_radius","fixed_radius"):
                        row=dict(patient=pt,record=rec,method=method,truth_pp=truth*100,
                                 mean_center_pp=ctr.mean()*100, signed_mean_error_pp=(ctr.mean()-truth)*100,
                                 signed_median_error_pp=(np.median(ctr)-truth)*100,
                                 realized_center_sd_pp=ctr.std(ddof=1)*100,
                                 mean_scale_pp=st["scale"].mean()*100, mean_width_pp=100*width.mean(),coverage=covered.mean(),
                                 frac_interval_entirely_above_truth=over.mean(),frac_interval_entirely_below_truth=under.mean())
                        for tag,arr in [("center",ctr),("width",width),("lower",l),("upper",u)]:
                            for q, val in zip(("q05","q50","q95"),np.quantile(100*arr,[.05,.5,.95])):
                                row[tag+"_"+q+"_pp"]=val
                        cases.append(row)
                        case_draws.append(pd.DataFrame(dict(patient=pt,record=rec,method=method,draw=np.arange(1,len(ctr)+1),
                            truth_pp=100*truth,center_pp=ctr*100,lower_pp=l*100,upper_pp=u*100,width_pp=100*width,
                            covered=covered,v_obs_over_v_fit=st["var"]/vfit)))
    assert checks==3180
    tiny_summary(pd.DataFrame(widths)).to_csv(out/"near_zero_widths.csv",index=False)
    caseframe=pd.DataFrame(cases).sort_values(["patient","method"])
    assert len(caseframe)==10 and caseframe.record.nunique()==5
    caseframe.to_csv(out/"five_cases_summary.csv",index=False)
    individual_draws=pd.concat(case_draws,ignore_index=True)
    describe_case_flags(individual_draws).to_csv(out/"five_cases_ratio_by_draw.csv",index=False)
    individual_draws.to_csv(out/"five_cases_draws.csv.gz",index=False,
        compression={"method":"gzip","mtime":0},float_format="%.12g")
    ratio_analysis(pd.concat(flags,ignore_index=True)).to_csv(out/"variance_ratio_flag.csv",index=False)
    print(f"Matched all six archived recording summaries in {checks} detector/record/method combinations.")
    print("Original thresholds and intervals were retained. Case-level subset uses frozen reporting criteria.")


if __name__=="__main__":
    main()
