"""Apply the fitted selector to NEW patients only; no reference labels required."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from region_selector import FEATURE_NAMES,load_feature_rows,export_component,file_hash,score_candidate_rows


def validate_application(bundle,stats,groups):
    if bundle.get('feature_names')!=FEATURE_NAMES: raise ValueError('Model feature schema differs')
    if bundle.get('backbone_checkpoint_sha256')!=stats['checkpoint_sha256']:
        raise ValueError('Backbone checkpoint differs from the selector training data')
    application_patients={p['patient'] for p in stats.get('patients',[])} or set(groups)
    repeated=set(bundle['training_patients']) & application_patients
    if repeated: raise ValueError('These patients trained the final selector. Use their saved leave-one-out masks instead: '+str(sorted(repeated)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stats',type=Path,required=True,help='region_probability_stats.json of new patients')
    parser.add_argument('--model',type=Path,default=Path.home()/'Datasets/nnUNet/experiments/region_selector_v1/selector.joblib',help='Trusted local joblib model only')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists(): parser.error('Output already exists; choose a fresh directory')
    # joblib loads executable Python objects: only load a model you created or trust.
    bundle=joblib.load(args.model)
    X,groups,rows,stats=load_feature_rows(args.stats)
    try: validate_application(bundle,stats,groups)
    except ValueError as exc: parser.error(str(exc))
    scores=score_candidate_rows(bundle['pipeline'],X)
    args.output_dir.mkdir(parents=True); decisions=[]
    for patient in stats['patients']:
        name=patient['patient']; indices=np.flatnonzero(groups==name)
        best=int(indices[np.argmax(scores[indices])]) if len(indices) else None
        component=rows[best]['component_id'] if best is not None else None
        export_component(args.stats.parent/(name+'.nii.gz'),component,args.output_dir/(name+'.nii.gz'))
        decisions.append({'patient':name,'selected_component_id':component,
                          'candidates':[{'component_id':rows[i]['component_id'],'selector_score':float(scores[i])} for i in indices]})
        print(name,'selected',component,flush=True)
    report={'method':'oracle_supervised_selector_final_model_new_patients',
            'uses_ground_truth_for_selection':False,'uses_ground_truth_for_training':True,
            'model_sha256':file_hash(args.model),'source_statistics_sha256':file_hash(args.stats),
            'patients':decisions,'limitations':['Prototype trained on 13 patients; external performance unvalidated',
                                               'Scores are not calibrated probabilities of tumor presence']}
    (args.output_dir/'postprocessing_manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__': main()
