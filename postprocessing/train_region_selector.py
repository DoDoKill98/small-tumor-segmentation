"""Train an oracle-supervised selector; validate with leave-one-patient-out only."""
import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import joblib
import numpy as np
from region_selector import FEATURE_NAMES,MODEL_CONFIG,file_hash,load_feature_rows,leave_one_patient_out,fit_selector,export_component
from region_probability_stats import DEFAULT_INPUT


def main():
    project=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction-dir',type=Path,default=DEFAULT_INPUT)
    parser.add_argument('--oracle-manifest',type=Path)
    parser.add_argument('--output-dir',type=Path,default=Path.home()/'Datasets/nnUNet/experiments/region_selector_v1')
    args=parser.parse_args(); root=args.prediction_dir; output=args.output_dir
    if output.exists(): parser.error('Experiment already exists; choose a fresh output directory')
    stats_path=root/'region_probability_stats.json'
    oracle_path=args.oracle_manifest or root/'oracle_best_overlap/postprocessing_manifest.json'
    oracle=json.loads(oracle_path.read_text())
    if not oracle.get('uses_ground_truth_for_selection') or oracle.get('method')!='oracle_max_dice_whole_component':
        parser.error('Expected the whole-component oracle training labels')
    if Path(oracle['input_dir']).resolve()!=root.resolve(): parser.error('Oracle source differs from feature source')
    X,groups,rows,stats=load_feature_rows(stats_path)
    oracle_by_id={r['patient']:r for r in oracle['patients']}
    patients=[p['patient'] for p in stats['patients']]
    if set(oracle_by_id)!=set(patients): parser.error('Oracle and feature patients differ')
    for p in stats['patients']:
        target=oracle_by_id[p['patient']]
        if target['input_sha256']!=p['mask_sha256']: parser.error('Oracle target derived from different prediction')
        if {r['component_id'] for r in target['components']}!={r['component_id'] for r in p['regions']}:
            parser.error('Candidate component IDs differ')
    y=np.array([int(row['component_id']==oracle_by_id[row['patient']]['selected_component_id']) for row in rows])
    if len(np.unique(groups))<3: parser.error('At least three patient groups required')
    started=time.perf_counter()
    scores,audit=leave_one_patient_out(X,y,groups)
    final_model=fit_selector(X,y,groups)
    training_seconds=time.perf_counter()-started
    output.mkdir(parents=True); masks=output/'loo_predictions'; masks.mkdir()
    decisions=[]
    for patient in patients:
        indices=np.flatnonzero(groups==patient)
        if len(indices):
            selected=int(indices[np.argmax(scores[indices])]); component=rows[selected]['component_id']
            candidate_scores=[{'component_id':rows[i]['component_id'],'selector_score':float(scores[i])} for i in indices]
        else: component=None; candidate_scores=[]
        export_component(root/(patient+'.nii.gz'),component,masks/(patient+'.nii.gz'))
        match=component==oracle_by_id[patient]['selected_component_id']
        decisions.append({'patient':patient,'selected_component_id':component,
                          'oracle_component_id_for_evaluation_only':oracle_by_id[patient]['selected_component_id'],
                          'matches_oracle':bool(match),'candidates':candidate_scores,
                          'training_patients':next(r['training_patients'] for r in audit if r['held_out_patient']==patient) if len(indices) else []})
        print(patient,'selected',component,'oracle',oracle_by_id[patient]['selected_component_id'],'match',match,flush=True)
    bundle={'schema_version':1,'pipeline':final_model,'feature_names':FEATURE_NAMES,'model_config':MODEL_CONFIG,
            'training_patients':patients,'backbone_checkpoint_sha256':stats['checkpoint_sha256'],
            'probability_stats_schema_version':stats['schema_version']}
    joblib.dump(bundle,output/'selector.joblib')
    table=[{**row,'oracle_target':int(target)} for row,target in zip(rows,y)]
    (output/'training_candidates.json').write_text(json.dumps(table,indent=2,allow_nan=False))
    limitations=['Development prototype: 13 patients, not 64 independent patients',
                 'Backbone BEST checkpoint selected on these same 13 patients; leave-one-out of selector does not remove this upstream selection bias',
                 'No independent test estimate or hyperparameter search; no claim of clinical reliability',
                 'Selector always chooses one existing component; cannot recover a missed tumor or refine boundaries',
                 'Classifier scores are not calibrated probabilities of clinical tumor presence']
    manifest={'method':'oracle_supervised_logistic_selector_v1_loo','uses_ground_truth_for_selection':False,
              'uses_ground_truth_for_training':True,'selection_protocol':'leave_one_patient_out; held-out labels excluded from fit',
              'feature_names':FEATURE_NAMES,'config':MODEL_CONFIG,'patients':decisions,'fold_audit':audit,
              'limitations':limitations,'source_probability_stats_sha256':file_hash(stats_path),
              'source_oracle_manifest_sha256':file_hash(oracle_path),'backbone_checkpoint_sha256':stats['checkpoint_sha256']}
    (masks/'postprocessing_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False))
    coefficients=final_model.named_steps['classifier'].coef_[0]
    summary={'created_utc':datetime.now(timezone.utc).isoformat(),'patients':len(patients),'candidate_regions':len(rows),
             'positive_targets':int(y.sum()),'negative_targets':int(len(y)-y.sum()),'model_fit_seconds':training_seconds,
             'loo_oracle_selection_matches':sum(d['matches_oracle'] for d in decisions),
             'final_model_training_patients':patients,'final_model_path':str((output/'selector.joblib').resolve()),
             'final_model_not_used_for_loo_predictions':True,'feature_names':FEATURE_NAMES,'model_config':MODEL_CONFIG,
             'standardized_coefficients':dict(zip(FEATURE_NAMES,coefficients.tolist())),
             'source_probability_stats_sha256':file_hash(stats_path),'source_oracle_manifest_sha256':file_hash(oracle_path),
             'versions':{name:importlib.metadata.version(name) for name in ['scikit-learn','numpy','joblib','scipy']},'limitations':limitations}
    (output/'training_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    checkpoint_name=Path(stats['checkpoint']).name
    subprocess.run([sys.executable,'-u',str(project/'postprocessing/evaluate.py'),'--prediction-dir',str(masks),
                    '--checkpoint-label',checkpoint_name],check=True)
    subprocess.run([sys.executable,'-u',str(project/'postprocessing/compare_evaluations.py'),
                    '--before',str(root/'evaluation_metrics.json'),'--after',str(masks/'evaluation_metrics.json')],check=True)
    print('Training and held-out evaluation completed:',output,flush=True)
    print('Oracle selections matched:',summary['loo_oracle_selection_matches'],'/',len(patients),flush=True)


if __name__=='__main__': main()
