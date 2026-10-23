"""Optional ensemble evidence, survival investigation and synthetic sensitivity."""
from pathlib import Path
import argparse
import json
import sys
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent/'lib'))
from monitoring_v3.config import load
from monitoring_v3.data import load_sources
from monitoring_v3.experiments import ensemble_features, survival_experiment, sensitivity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment',choices=['ensemble','survival','sensitivity'])
    p.add_argument('--config',required=True);p.add_argument('--unit',required=True)
    p.add_argument('--model');p.add_argument('--sensor');p.add_argument('--start');p.add_argument('--as-of')
    a=p.parse_args();cfg=load(a.config);contexts,_=load_sources(cfg);c=contexts[a.unit]
    root=Path(cfg['output']);dest=root/'investigations'/a.unit/a.experiment;dest.mkdir(parents=True,exist_ok=True)
    if a.experiment=='ensemble':
        ensemble_features(root,a.unit).to_parquet(dest/'ensemble_features.parquet')
    elif a.experiment=='survival':
        if not a.as_of:
            p.error('--as-of is required')
        f=ensemble_features(root,a.unit)
        f=f.loc[c['eligible'].reindex(f.index).fillna(False)]
        print(json.dumps(survival_experiment(f,c['events'],cfg,pd.Timestamp(a.as_of),c['frame'].index[-1],dest),indent=2))
    else:
        if not (a.model and a.sensor and a.start):
            p.error('--model, --sensor and --start are required')
        from saved_model_runner import _runtime
        with _runtime(Path(a.model),'incremental') as (model,_):
            if pd.Timestamp(a.start)<model.state['cutoff']:
                p.error('Injection must start after the model cutoff')
            sensitivity(model,c['frame'],a.sensor,a.start,cfg,dest)


if __name__=='__main__':
    main()
