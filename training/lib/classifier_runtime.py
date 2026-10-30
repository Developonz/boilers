"""Portable telemetry-only inference; copied into each classifier export."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .preparation import clean,regimes
from .regime_evidence import evidence_features
from .input_grid import normalize_grid
from .feature_functions import direct_features,direct_native


class Model:
    def __init__(self,folder):
        from catboost import CatBoostClassifier
        self.folder=Path(folder)
        self.config=json.loads((self.folder/'config.json').read_text(encoding='utf-8'))
        self.model=CatBoostClassifier();self.model.load_model(str(self.folder/'classifier.cbm'))
        if list(self.model.feature_names_)!=self.config['columns']:raise ValueError('Classifier feature schema mismatch')

    def features(self,raw,unit,at,start=None,aliases=None):
        config=self.config
        if unit not in config['units']:raise ValueError('Котёл отсутствует в экспорте классификатора: '+str(unit))
        spec=config['units'][unit];at=pd.Timestamp(at)
        if at<pd.Timestamp(config['available_at']):raise ValueError('Модель ещё не была доступна на выбранную дату')
        raw=raw.copy();raw.columns=[str(c).strip() for c in raw]
        if aliases:raw=raw.rename(columns=aliases)
        if 'TimeValue' in raw:
            time=pd.to_datetime(raw['TimeValue'],errors='coerce');raw=raw.loc[time.le(at)]
        else:raw=raw.loc[:at]
        roles=spec['roles'];mapping=spec['aliases']
        used={};numeric=[];states=[]
        for canonical,role in roles.items():
            hits=[c for c in raw if mapping.get(c,c)==canonical]
            if len(hits)>1:raise ValueError('Неоднозначный канал классификатора: '+canonical)
            if hits:
                used[hits[0]]=canonical
                (states if role=='state' else numeric).append(hits[0])
        frame,audit=normalize_grid(raw[[c for c in raw if c in used or c=='TimeValue']],config['preparation']['step'],numeric,states)
        frame=frame.rename(columns=used).reindex(columns=list(roles))
        if frame.empty or at not in frame.index:raise ValueError('Нет данных классификатора за выбранный час')
        if frame.index[0]>pd.Timestamp(spec['history_start']):
            raise ValueError('Для восстановления истории режимов нужна телеметрия с '+spec['history_start'])
        frame=frame.loc[pd.Timestamp(spec['history_start']):at]
        missing=[c for c in roles if c not in used.values()]
        if len(missing)==len(roles):raise ValueError('Не сопоставлены датчики классификатора')
        cfg=config['preparation']
        checked=[c for c,r in roles.items() if r in ('load','condition','target')]
        cleaned,flags,changes=clean(frame,checked,config['sensor_rules'],cfg['step'])
        del flags,changes
        evidence=evidence_features(cleaned,spec['regime_spec'])
        mode=regimes(cleaned[spec['load']],spec['threshold'],cfg,evidence)
        context=dict(frame=frame,cleaned=cleaned,roles=roles,unit=unit,regime=mode,
            threshold_source=spec['threshold_source'],preparation_reference_end=spec['reference_end'],
            regime_spec=spec['regime_spec'],declared_units=spec['declared_units'])
        features,meta,_=direct_features(context,cfg,config['options'],start=start or at)
        absent=set(config['columns'])-set(features)
        if absent:raise ValueError('Не сформированы признаки: '+str(sorted(absent)))
        return features,meta,missing

    def predict(self,raw,unit,at,start=None,aliases=None):
        from catboost import Pool
        features,meta,missing=self.features(raw,unit,at,start,aliases)
        result=meta.copy();result['score']=np.nan
        valid=meta.eligible&(meta.index>=pd.Timestamp(self.config['available_at']))
        if valid.any():
            x=direct_native(features.loc[valid],self.config['columns'],self.config['categorical'])
            result.loc[valid,'score']=self.model.predict_proba(Pool(x,cat_features=self.config['categorical']),
                thread_count=int(self.config['options']['threads']))[:,1]
        return result,missing
