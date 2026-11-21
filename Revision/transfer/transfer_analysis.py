"""Secondary detector, calibration transfer and clock-stratification analyses.

Usage: python transfer_analysis.py --study /path/to/holter_intervals --out new_folder
Requires the dependencies in the original study archive.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import hashlib
import json
import numpy as np
import pandas as pd

@dataclass
class Record:
    key: str
    unit: str
    idx: np.ndarray
    y: np.ndarray
    score: np.ndarray
    strata: np.ndarray

def read_sets(root, detector, site, alg):
    manifest = pd.read_csv(root/'inputs'/f'{detector}_{site}.csv', dtype={'record':str,'patient':str})
    for row in manifest.itertuples():
        if not row.eligible:
            continue
        path = root/row.path
        if hashlib.sha256(path.read_bytes()).hexdigest() != row.sha256:
            raise ValueError(path)
        with np.load(path) as arr:
            idx = arr['idx'].astype(int)
            raw = arr['raw12'].astype(float)/12
            assert np.all(idx < 1440)
            yield Record(row.record, row.patient, idx, arr['y'].copy(), raw[idx],
                         alg.strata_for_minutes(np.arange(len(raw)), raw)[idx])

def block(r):
    return r.idx//360

def groups(frame):
    yield 'overall', frame
    for label, f in frame.groupby('burden_band'):
        yield label, f

def burden_band(percent):
    if percent < 1e-10: return 'zero'
    if percent <= 5+1e-10: return 'low'
    if percent >= 100-1e-10: return 'full'
    return 'intermediate'

def sample_row(base, error, truth, center, radius, method):
    lo = np.maximum(0, center-radius)
    hi = np.minimum(1, center+radius)
    return dict(**base, method=method, coverage=((lo<=truth+1e-12)&(hi>=truth-1e-12)).mean(),
                width_pp=100*(hi-lo).mean(), center_mae_pp=100*error.mean())

def fit_and_draw(records, study, alg, predict, var, cohort, strata):
    for r in records:
        p = predict[alg.levels(r.score)]
        out = study.draw(r.y,strata(r),p,var,alg.stable_seed('holter-interval',cohort,r.key))
        center = np.clip(out['mu'],0,1)
        yield r,center,out['scale'],np.abs(center-r.y.mean())

def summaries(records, key_cols, metrics):
    output = []
    for key,f in records.groupby(key_cols, sort=True):
        if not isinstance(key,tuple): key=(key,)
        for band,subset in groups(f):
            units = subset.groupby('patient')[metrics].mean()
            row=dict(zip(key_cols,key), burden_band=band, units=len(units), recordings=len(subset))
            row.update(units.mean().to_dict())
            if 'coverage' in metrics:
                row['below80']=int((units.coverage<.8).sum())
                row['below50']=int((units.coverage<.5).sum())
            if 'lag1' in metrics: row['units_with_lag']=int(units.lag1.count())
            output.append(row)
    return pd.DataFrame(output)

def temporal_metrics(r, error, s, predicted_var, alg):
    counts = np.bincount(s,minlength=6)
    takes = alg.allocate(counts,np.ones(6),20)
    centered = np.zeros_like(error)
    known,trained=0.,0.
    residual_metrics=[]
    for h,n in enumerate(counts):
        if n==0:continue
        here=s==h
        vals=error[here]
        centered[here]=vals-vals.mean()
        v=vals.var(ddof=1) if n>=2 else np.nan
        if n>takes[h]:
            weight=(n/len(error))**2*(1-takes[h]/n)/takes[h]
            known+=weight*v
            trained+=weight*predicted_var[h]
        residual_metrics.append(dict(stratum=h,n=int(n),fraction=n/len(error),
                                      mean_residual_pp=100*vals.mean(),within_variance=v,
                                      minute_abs_error_pp=100*np.abs(vals).mean()))
    pairs=(s[1:]==s[:-1])&(np.diff(r.idx)==1)
    lag=np.nan
    if pairs.sum()>=2:
        a,b=centered[:-1][pairs],centered[1:][pairs]
        if a.std()>1e-12 and b.std()>1e-12: lag=float(np.corrcoef(a,b)[0,1])
    return dict(lag1=lag,adjacent_pairs=int(pairs.sum()),true_design_sd_pp=100*np.sqrt(known),
                predicted_design_sd_pp=100*np.sqrt(trained),minute_abs_error_pp=100*np.abs(error).mean(),
                signed_detector_bias_pp=-100*error.mean(),abs_detector_bias_pp=100*abs(error.mean())),residual_metrics

def execute(args):
    root=args.study.resolve()
    sys.path.insert(0,str(root))
    import study
    import pilot_estimators as alg
    interval_rows=[]; diag=[]; levels=[]; models=[]
    for det in ['ArNet2','COSEn']:
        primary=json.loads((root/'results'/f'model_{det}.json').read_text())
        all_lta={r.key:r for r in read_sets(root,det,'LTA',alg)}
        fit=[all_lta[k] for k in primary['fit_records']]
        calibration=[all_lta[k] for k in primary['calibration_records']]
        assert len(fit)==len(calibration)==42 and set(primary['fit_records']).isdisjoint(primary['calibration_records'])
        pred=np.array(primary['p']); fitted_var=np.array(primary['residual_variance'])
        model,new_var=alg.calibrate(np.stack([alg.record_stats(r.score,block(r),r.y) for r in fit]).sum(axis=0))
        np.testing.assert_allclose(pred,model,atol=2e-13,rtol=0)
        for name,get_groups,variances in [('original',lambda r:r.strata,fitted_var),('six_hour_blocks',block,new_var)]:
            old=list(fit_and_draw(calibration,study,alg,pred,variances,'LTA',get_groups))
            abs_errors=np.array([err for _,_,_,err in old])
            scores=np.array([err/s for _,_,s,err in old])
            radius=study.hquant(abs_errors,42);mult=study.hquant(scores,42)
            if name=='original':
                np.testing.assert_allclose([radius,mult],[primary['radius'],primary['multiplier']],atol=5e-13,rtol=0)
            models.append(dict(detector=det,design=name,fixed_halfwidth_pp=100*radius,multiplier=mult,
                               fit_variances=variances.tolist(),fit_n=len(fit),calibration_n=len(calibration)))
            belgium=list(read_sets(root,det,'IRIDIA',alg))
            for cohort,rows in [('LTA',old),('IRIDIA',fit_and_draw(belgium,study,alg,pred,variances,'IRIDIA',get_groups))]:
                for j,(r,center,scale,err) in enumerate(rows):
                    truth=r.y.mean()
                    base=dict(detector=det,cohort=cohort,design=name,record=r.key,patient=r.unit,
                              burden_band=burden_band(100*truth),true_burden_pp=100*truth)
                    if cohort=='LTA':
                        comparators=np.arange(len(old))!=j
                        rad=study.hquant(abs_errors[comparators],41)
                        multiplier=study.hquant(scores[comparators],41)
                    else: rad,multiplier=radius,mult
                    interval_rows.extend([sample_row(base,err,truth,center,rad,'fixed_radius'),
                                          sample_row(base,err,truth,center,multiplier*scale,'scaled_radius')])
                    if name=='original':
                        residuals=r.y-pred[alg.levels(r.score)]
                        d,hs=temporal_metrics(r,residuals,r.strata,fitted_var,alg)
                        diag.append(dict(**base,**d))
                        levels.extend(dict(**base,**h) for h in hs)
    dest=args.out;dest.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(interval_rows)
    for det in ['ArNet2','COSEn']:
        old=pd.read_csv(root/'results'/f'records_IRIDIA_{det}.csv',dtype={'patient':str,'record':str})
        old=old[old.method.isin(['fixed_radius','scaled_radius'])].set_index(['record','method']).sort_index()
        repeat=frame[(frame.cohort=='IRIDIA')&(frame.design=='original')&(frame.detector==det)].set_index(['record','method']).sort_index()
        assert old.index.equals(repeat.index)
        np.testing.assert_allclose(old.coverage,repeat.coverage,rtol=0,atol=1e-12)
        np.testing.assert_allclose(old.width_pp,repeat.width_pp,rtol=0,atol=1e-10)
    frame.to_csv(dest/'interval_records.csv',index=False)
    summaries(frame,['detector','cohort','design','method'],['coverage','width_pp','center_mae_pp']).to_csv(dest/'interval_summary.csv',index=False)
    pd.DataFrame(models).to_csv(dest/'calibration_parameters.csv',index=False)
    d=pd.DataFrame(diag)
    d.to_csv(dest/'residual_by_record.csv',index=False)
    residual_cols=['lag1','true_design_sd_pp','predicted_design_sd_pp','minute_abs_error_pp','signed_detector_bias_pp','abs_detector_bias_pp']
    summaries(d,['detector','cohort'],residual_cols).to_csv(dest/'residual_summary.csv',index=False)
    sv=pd.DataFrame(levels)
    sv.to_csv(dest/'strata_by_record.csv',index=False)
    keys=['detector','cohort','stratum']
    # Stratum-specific means conditional on its presence. Singleton strata have undefined finite-population variance.
    aggregated=sv.groupby(keys+['patient'])[['within_variance','mean_residual_pp','minute_abs_error_pp','fraction']].mean().reset_index()
    strata=aggregated.groupby(keys).agg(units=('patient','count'),variance_units=('within_variance','count'),
        avg_within_variance=('within_variance','mean'),median_within_variance=('within_variance','median'),
        avg_minute_abs_error_pp=('minute_abs_error_pp','mean'),avg_residual_pp=('mean_residual_pp','mean'),
        avg_fraction_conditional_present=('fraction','mean')).reset_index()
    strata.to_csv(dest/'strata_summary.csv',index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    execute(p.parse_args())
