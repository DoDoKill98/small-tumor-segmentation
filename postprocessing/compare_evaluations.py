"""Compare two evaluation JSONs on exactly the same patients and definitions."""
import argparse
import json
import os
from pathlib import Path
import numpy as np


def compare_reports(before,after):
    for key in ('fold','split_sha256','surface_tolerance_mm','checkpoint_label','definitions'):
        if before[key]!=after[key]: raise ValueError(f'Evaluation settings differ: {key}')
    if Path(before['reference_dir']).resolve()!=Path(after['reference_dir']).resolve():
        raise ValueError('Reference annotation directories differ')
    a={r['patient']:r for r in before['patients']}; b={r['patient']:r for r in after['patients']}
    if a.keys()!=b.keys(): raise ValueError('Patient IDs differ')
    if len(a)!=len(before['patients']) or len(b)!=len(after['patients']): raise ValueError('Duplicate patients')
    metrics={'dice':1,'surface_dice':1,'hd95_mm':-1,'voxel_recall':1,'voxel_precision':1,'absolute_volume_error_ml':-1,
             'regions_completely_missed':-1,'regions_unmatched_prediction':-1}
    rows=[]; summary={}; rng=np.random.default_rng(12345)
    for patient in sorted(a):
        changes={}
        for key in metrics:
            x,y=a[patient][key],b[patient][key]
            changes[key]={'before':x,'after':y,'delta':y-x if x is not None and y is not None else None}
        rows.append({'patient':patient,'metrics':changes})
    for key,direction in metrics.items():
        valid=[r['metrics'][key] for r in rows if r['metrics'][key]['delta'] is not None]
        delta=np.array([r['delta'] for r in valid],float)
        s={'n_pairs':len(delta),'n_undefined_pairs':len(rows)-len(delta),'better_direction':'increase' if direction==1 else 'decrease'}
        if len(delta):
            s.update(mean_before=float(np.mean([r['before'] for r in valid])),mean_after=float(np.mean([r['after'] for r in valid])),
                     mean_delta=float(delta.mean()),improved=int(np.count_nonzero(delta*direction>1e-10)),
                     degraded=int(np.count_nonzero(delta*direction < -1e-10)),unchanged=int(np.count_nonzero(np.abs(delta)<=1e-10)))
            if len(delta)>1:
                means=rng.choice(delta,size=(2000,len(delta)),replace=True).mean(axis=1)
                s['paired_bootstrap_mean_delta_ci95']=np.percentile(means,[2.5,97.5]).tolist()
        summary[key]=s
    return {'patients':rows,'summary':summary,
            'bootstrap':{'unit':'paired patient differences','samples':2000,'seed':12345},
            'limitations':['Exploratory postprocessing comparison on the same validation fold used for model development; not independent test performance']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before',type=Path,required=True)
    parser.add_argument('--after',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    before=json.loads(args.before.read_text()); after=json.loads(args.after.read_text())
    try: report=compare_reports(before,after)
    except ValueError as exc: parser.error(str(exc))
    report.update(before_report=str(args.before.resolve()),after_report=str(args.after.resolve()))
    report['before_postprocessing']=before.get('postprocessing')
    report['after_postprocessing']=after.get('postprocessing')
    if any((r.get('postprocessing') or {}).get('uses_ground_truth_for_selection') for r in (before,after)):
        report['limitations'].append('Comparison includes ORACLE selection using ground truth; not a comparison of two deployable automatic methods')
    output=args.output or args.after.parent/'comparison_to_raw.json'
    if output.resolve() in (args.before.resolve(),args.after.resolve()): parser.error('Output must not overwrite an input report')
    temporary=output.with_suffix('.json.tmp'); temporary.write_text(json.dumps(report,indent=2,allow_nan=False)); os.replace(temporary,output)
    for row in report['patients']:
        d=row['metrics']['dice']; h=row['metrics']['hd95_mm']
        print(f'{row["patient"]}: Dice {d["before"]:.4f} -> {d["after"]:.4f}; HD95 {h["before"]} -> {h["after"]}',flush=True)
    for key in ['dice','surface_dice','hd95_mm']: print(key,report['summary'][key],flush=True)
    print('Report:',output,flush=True)


if __name__=='__main__': main()
