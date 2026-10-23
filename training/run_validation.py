"""Generate a clearly synthetic demonstration and exercise every supported normal model."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd
import openpyxl

sys.path.insert(0,str(Path(__file__).resolve().parent/'lib'))
from monitoring_v3.config import load as load_config
from monitoring_v3.pipeline import run
from monitoring_v3.data import load_sources


def generate(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(2026)
    ix=pd.date_range('2024-01-01',periods=3300,freq='h')
    t=np.arange(len(ix));load=100+15*np.sin(t/40)+rng.normal(0,.5,len(t))
    state=np.full(len(t),'Работа',dtype=object)
    for start in (1650,2500):
        state[start:start+24]='Аварийный ремонт';load[start:start+24]=0
    temp=200+.3*load+2*np.where(t%100<60,1.,0.)+rng.normal(0,.2,len(t))
    temp[1600:1650]+=np.linspace(0,6,50)
    rare=np.zeros(len(t));rare[2100:]=1.
    frame=pd.DataFrame({'TimeValue':ix,'Состояние':state,'KA-1 Расход пара':load,
        'KA-1 Температура':temp,'KA-1 Давление':12+.02*load+rng.normal(0,.03,len(t)),
        'KA-1 Положение':rare,'KA-1 Механизм':np.where(t%100<60,'Включен','Отключен')})
    folder=root/'source'/'К-1';folder.mkdir(parents=True)
    frame.to_parquet(folder/'hourly.parquet',index=False)
    wb=openpyxl.Workbook();s=wb.active;s.title='КА-1'
    s.append(['col','роль','общий параметр','единица'])
    for row in [('KA-1 Расход пара','нагрузка','нагрузка','т/ч'),
                ('KA-1 Температура','проверять','температура','C'),
                ('KA-1 Давление','проверять','давление','МПа'),
                ('KA-1 Положение','проверять','положение',''),
                ('KA-1 Механизм','состояние','механизм','')]:s.append(row)
    wb.save(root/'roles.xlsx')
    (root/'rules.json').write_text(json.dumps({'sensors':{'положение':{'kind':'rare'},
        'температура':{'kind':'continuous','group':'water_steam'},
        'давление':{'kind':'continuous','group':'water_steam'}}},ensure_ascii=False),encoding='utf-8')
    cfg=load_config(overrides=dict(groups={'synthetic':['К-1']},data_root=str(root/'source'),
        sensor_file=str(root/'roles.xlsx'),sensor_rules=str(root/'rules.json'),output=str(root/'results'),
        first_test='2024-03-01',startup_hours=2,min_train_rows=150,min_sensor_rows=24,
        calibration_rows=48,validate_rows=48,memory_rows=100,replay_rows=40,update_rows=500,
        models=['analogs','AAKR','MSET','HistGradientBoosting','CatBoost','CatBoostCorrection','CatBoostMulti'],
        feature_sets=['conditions','forecast'],horizons=[1],cb_initial_trees=15,cb_update_trees=5,
        cb_max_trees=100,cb_depth=2,threads=1))
    (root/'settings.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    (root/'SYNTHETIC_ONLY.txt').write_text('Generated toy data. Results verify implementation and are not evidence of performance on real boilers.',encoding='utf-8')
    return cfg


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='runs/synthetic_validation')
    a=p.parse_args();cfg=generate(Path(a.output).resolve());result=run(cfg)
    root=Path(cfg['output'])
    from monitoring_v3.experiments import sensitivity,ensemble_features,survival_experiment
    from saved_model_runner import _runtime
    context=load_sources(cfg)[0]['К-1']
    export=next((root/'experiments').glob('*__CatBoost__conditions__0/К-1/model'))
    with _runtime(export,'incremental') as (model,_):
        sensitivity(model,context['frame'],'температура',model.state['cutoff']+pd.Timedelta('48h'),cfg,root/'sensitivity')
    ensemble=ensemble_features(root,'К-1');ensemble.to_parquet(root/'ensemble_features.parquet')
    survival_experiment(ensemble,context['events'],cfg,context['frame'].index[-1],context['frame'].index[-1],root/'survival')
    from train_interpretation import train
    experiment=export.parent
    train(cfg,experiment,'К-1',context['frame'].index[-1])
    print(result['status'].to_string(index=False))
    print('SYNTHETIC VALIDATION ONLY:',root)


if __name__=='__main__':main()
