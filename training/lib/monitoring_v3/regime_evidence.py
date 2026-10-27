"""Motor currents corroborate fuel-system operation; they never replace steam flow."""
import re
import numpy as np
import pandas as pd


def _line(name):
    match=re.search(r'\b(?:ммт|мв|мельница|псу)[\s-]*(?:#|\d+)?([а-яa-z])\b',name,re.I)
    return match.group(1).lower() if match else None


def reference_end(frame,cfg,load=None):
    first=pd.Timestamp(cfg['first_test'])
    if not cfg['history_policy']['bootstrap']:return first
    anchor=frame.index[0]
    if load in frame:
        working=frame.get('Состояние',pd.Series('',index=frame.index)).eq('Работа')
        good=frame.index[(frame.index<first)&working&pd.to_numeric(frame[load],errors='coerce').gt(0)]
        if len(good):anchor=good[0]
    return min(first,anchor+pd.Timedelta(days=cfg['history_policy']['reference_days']))


def fit_evidence(frame,roles,unit,cfg):
    end=reference_end(frame,cfg,next((c for c,r in roles.items() if r=='load'),None))
    policy=cfg['regime_policy']
    configured=policy['current_thresholds']
    if any(isinstance(v,dict) for v in configured.values()) and not all(isinstance(v,dict) for v in configured.values()):
        raise ValueError('Use either common current thresholds or dictionaries by unit, not a mixture')
    manual=configured.get(unit,{}) if any(isinstance(v,dict) for v in configured.values()) else configured
    channels={};audit=[]
    for name,role in roles.items():
        if role!='condition' or not re.search(r'\bток\b',name,re.I):continue
        if 'управлен' in name:group='control'
        elif re.search(r'\bпсу\b',name):group='feeder'
        elif re.search(r'\b(ммт|мв|мельниц)',name):group='mill'
        elif re.search(r'\b(дв|дс|дрг|вгд)\b',name):group='draft'
        else:group='other'
        x=pd.to_numeric(frame.loc[frame.index<end,name],errors='coerce').replace([np.inf,-np.inf],np.nan)
        positive=x[x>0]
        typical=float(positive.quantile(.75)) if len(positive) else np.nan
        threshold=manual.get(name) if isinstance(manual,dict) else None
        source='manual' if threshold is not None else 'early_past'
        if threshold is None and len(positive)>=cfg['min_sensor_rows'] and np.isfinite(typical):
            threshold=typical*policy['current_fraction']
        if threshold is not None and (not np.isfinite(threshold) or threshold<0):
            raise ValueError('Invalid current threshold: '+name)
        channels[name]=dict(group=group,line=_line(name),threshold=float(threshold) if threshold is not None else None,
                            typical=typical if np.isfinite(typical) else None)
        audit.append(dict(sensor=name,group=group,line=_line(name),threshold=threshold,source=source,
                          reference_end=str(end),observed=int(x.notna().sum()),positive_observed=len(positive),
                          status='available' if threshold is not None else 'insufficient_reference'))
    if any(roles.get(c)!='condition' for c in policy['oil_channels']):
        raise ValueError('Oil evidence channels must have role condition; target-derived inputs would leak their own values')
    oils=[c for c,r in roles.items() if r=='condition' and (c in policy['oil_channels'] or ('расход' in c and 'мазут' in c))]
    unknown=set(manual)-set(channels)
    if unknown:raise ValueError('Unknown current_thresholds channels for '+unit+': '+str(sorted(unknown)))
    pairs=[]
    manual_pairs=policy.get('current_pairs',{}).get(unit)
    if manual_pairs is not None:
        for n,pair in enumerate(manual_pairs):
            if len(pair)!=2 or any(c not in channels for c in pair) or pair[0]==pair[1]:
                raise ValueError('current_pairs must contain two distinct current condition channels: '+str(pair))
            pairs.append(dict(line=str(n+1),mill=pair[0],feeder=pair[1],source='manual'))
    else:
        for line in sorted({d['line'] for d in channels.values()}-{None}):
            mill=[c for c,d in channels.items() if d['line']==line and d['group']=='mill']
            feeder=[c for c,d in channels.items() if d['line']==line and d['group']=='feeder']
            if len(mill)==len(feeder)==1:
                pairs.append(dict(line=line,mill=mill[0],feeder=feeder[0],source='name_inference_check_locally'))
    for row in audit:
        row['paired_with']=';'.join(p['feeder'] if p['mill']==row['sensor'] else p['mill']
            for p in pairs if row['sensor'] in (p['mill'],p['feeder']))
    return dict(channels=channels,pairs=pairs,oil_channels=oils,reference_end=str(end),audit=audit)


def evidence_features(frame,spec):
    result={}
    for group in ('mill','feeder','draft','control','other'):
        cols={name:pd.to_numeric(frame[name] if name in frame else pd.Series(np.nan,index=frame.index),errors='coerce').replace([np.inf,-np.inf],np.nan)
              for name,d in spec['channels'].items() if d['group']==group and d['threshold'] is not None}
        if cols:
            data=pd.DataFrame(cols,index=frame.index)
            on=pd.DataFrame({c:data[c].gt(spec['channels'][c]['threshold']).where(data[c].notna()) for c in data})
            result[group+'_active']=on.sum(axis=1,min_count=1)
            result[group+'_coverage']=data.notna().mean(axis=1)
            result[group+'_level']=pd.DataFrame({c:data[c]/spec['channels'][c]['typical'] for c in data
                if spec['channels'][c]['typical'] is not None and spec['channels'][c]['typical']>0},index=frame.index).mean(axis=1)
        else:
            result[group+'_active']=pd.Series(np.nan,index=frame.index)
            result[group+'_coverage']=pd.Series(0.,index=frame.index)
            result[group+'_level']=pd.Series(np.nan,index=frame.index)
    oils=[c for c in spec['oil_channels'] if c in frame]
    oil=frame[oils].apply(pd.to_numeric,errors='coerce') if oils else None
    result['oil_active']=oil.gt(0).any(axis=1).where(oil.notna().any(axis=1)).astype(float) if oils else pd.Series(np.nan,index=frame.index)
    out=pd.DataFrame(result,index=frame.index)
    coal=out.mill_active.gt(0)&out.feeder_active.gt(0)
    coal_known=out.mill_active.notna()&out.feeder_active.notna()
    paired={};mill_only={};feeder_only={}
    for p in spec.get('pairs',[]):
        m,f=p['mill'],p['feeder']
        if any(c not in frame or spec['channels'][c]['threshold'] is None for c in (m,f)):continue
        mx=pd.to_numeric(frame[m],errors='coerce');fx=pd.to_numeric(frame[f],errors='coerce')
        known=mx.notna()&fx.notna()
        ma=mx.gt(spec['channels'][m]['threshold']);fa=fx.gt(spec['channels'][f]['threshold'])
        paired[p['line']]=(ma&fa).astype(float).where(known)
        mill_only[p['line']]=(ma&~fa).astype(float).where(known)
        feeder_only[p['line']]=(~ma&fa).astype(float).where(known)
        out['pair_'+p['line']+'_active']=paired[p['line']]
    if spec.get('pairs'):
        pair_data=pd.DataFrame(paired,index=frame.index)
        out['paired_active']=pair_data.sum(axis=1,min_count=1)
        out['paired_coverage']=pair_data.notna().sum(axis=1)/len(spec['pairs'])
        out['mill_without_feeder']=pd.DataFrame(mill_only,index=frame.index).sum(axis=1,min_count=1)
        out['feeder_without_mill']=pd.DataFrame(feeder_only,index=frame.index).sum(axis=1,min_count=1)
        coal=out.paired_active.gt(0)
        coal_known=coal|out.paired_coverage.eq(1.)
    out['coal_evidence']=coal.where(coal_known).astype(float)
    out['fuel_mode']=np.select([coal&out.oil_active.eq(1),coal,out.oil_active.eq(1)],
                               ['mixed','coal','oil'],default='unknown')
    return out
