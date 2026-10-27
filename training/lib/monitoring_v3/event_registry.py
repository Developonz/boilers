"""Enumerate load-confirmed physical stops first, attach documents second."""
import numpy as np
import pandas as pd
import json

EVENT_COLUMNS=['event_id','start','stop','end','type','reason','source','label_available_at','verified',
    'open_ended','issue','stop_detected_at','document_types','document_count','observed_stop',
    'documents','review_type','review_available_at','reviews']


def empty_registry():
    result=pd.DataFrame(columns=EVENT_COLUMNS)
    for c in ('start','stop','end','label_available_at','stop_detected_at','review_available_at'):
        result[c]=pd.to_datetime(result[c])
    return result


def event_type_at(event,as_of=None):
    get=lambda key,default=None:event.get(key,default) if isinstance(event,dict) else getattr(event,key,default)
    history=get('reviews','')
    history=json.loads(history) if isinstance(history,str) and history else []
    future_review=False
    if history:
        history=sorted(history,key=lambda r:pd.Timestamp(r['available_at']))
        known=[r for r in history if as_of is None or pd.Timestamp(r['available_at'])<=pd.Timestamp(as_of)]
        if known:return known[-1]['type'],pd.Timestamp(known[-1]['available_at'])
        future_review=True
    reviewed=get('review_type','')
    if not history and isinstance(reviewed,str) and reviewed and (as_of is None or pd.Timestamp(get('review_available_at'))<=pd.Timestamp(as_of)):
        return reviewed,pd.Timestamp(get('review_available_at'))
    if not history and isinstance(reviewed,str) and reviewed:future_review=True
    documents=get('documents','')
    docs=json.loads(documents) if isinstance(documents,str) and documents else []
    if not docs:
        return ('Неизвестно',get('stop_detected_at',pd.NaT)) if future_review else (get('type','Неизвестно'),get('label_available_at',pd.NaT))
    docs=[d for d in docs if as_of is None or pd.Timestamp(d['start'])<=pd.Timestamp(as_of)]
    if not docs:return 'Неизвестно',pd.NaT
    types={d['type'] for d in docs}
    if types=={'Авария','Резерв'}:
        failure=min(pd.Timestamp(d['start']) for d in docs if d['type']=='Авария')
        reserve=min(pd.Timestamp(d['start']) for d in docs if d['type']=='Резерв')
        if failure<=reserve:return 'Авария',failure
    if len(types)==1:return next(iter(types)),min(pd.Timestamp(d['start']) for d in docs)
    return 'Неизвестно',max(pd.Timestamp(d['start']) for d in docs)


def build_registry(frame,mode,cfg,unit=''):
    step=pd.Timedelta(cfg['step']);ix=frame.index;rows=[]
    state=frame['Состояние'].fillna('Нет данных').astype(str)
    on=mode['mode'].isin(['работа','пуск'])
    if 'stop_confirmed' in mode:
        candidates=[(pd.Timestamp(r.physical_stop_at),t) for t,r in mode.loc[mode.stop_confirmed].iterrows()]
    else:
        mask=mode['mode'].eq('стоит')&on.shift(fill_value=False)
        candidates=[(t,t) for t in ix[mask]]
    for stop,detected in candidates:
        future=ix[(ix>stop)&on]
        end=future[0] if len(future) else ix[-1]+step
        rows.append(dict(event_id=f'{unit}::stop::{stop.isoformat()}',start=stop,stop=stop,end=end,
            type='Неизвестно',reason='',source='load',label_available_at=detected,
            stop_detected_at=detected,verified=False,open_ended=not len(future),
            issue='unlabelled_load_stop',document_types='',document_count=0,observed_stop=True,
            documents='[]',review_type='',review_available_at=pd.NaT))
    kinds={'Аварийный ремонт':'Авария','Плановый ремонт':'План','Резерв':'Резерв'}
    for _,s in state.groupby(state.ne(state.shift()).cumsum()):
        kind=kinds.get(s.iloc[0])
        if kind is None:continue
        start,end=s.index[0],s.index[-1]+step
        matches=[r for r in rows if pd.notna(r['stop']) and abs(r['stop']-start)<=pd.Timedelta('72h')]
        before=[r for r in matches if r['stop']<=start]
        match=max(before,key=lambda r:r['stop']) if before else min(matches,key=lambda r:r['stop']) if matches else None
        if match is None:
            match=dict(event_id=f'{unit}::document::{start.isoformat()}',start=start,stop=pd.NaT,end=end,
                type=kind,reason='',source='Состояние',label_available_at=start,stop_detected_at=pd.NaT,
                verified=False,open_ended=end>=ix[-1]+step,issue='missing_load_stop',
                document_types='',document_count=0,observed_stop=False,documents='[]',review_type='',review_available_at=pd.NaT)
            rows.append(match)
        types=set(filter(None,match['document_types'].split(';')));types.add(kind)
        docs=json.loads(match['documents']);docs.append(dict(start=str(start),end=str(end),type=kind))
        match['documents']=json.dumps(docs,ensure_ascii=False)
        match.update(start=min(start,match['start']) if match['document_count'] else start,
            end=max(end,match['end']),
            type=kind if len(types)==1 else 'Неизвестно',document_types=';'.join(sorted(types)),
            document_count=match['document_count']+1,source='load+Состояние' if match['observed_stop'] else 'Состояние',
            label_available_at=max(start,match['label_available_at']) if match['document_count'] else start,
            issue='conflicting_document_types' if len(types)>1 else 'reason_unverified',
            open_ended=max(end,match['end'])>=ix[-1]+step)
        resolved,available=event_type_at(match)
        match['type']=resolved;match['label_available_at']=available
        if resolved=='Авария' and len(types)>1:match['issue']='emergency_repair_then_reserve'
    for override in cfg.get('event_overrides',[]):
        if override['unit']!=unit:continue
        found=[r for r in rows if pd.notna(r['stop']) and r['stop']==pd.Timestamp(override['stop'])]
        if len(found)!=1:raise ValueError('Reviewed physical stop not found uniquely: '+str(override))
        row=found[0]
        history=json.loads(row.get('reviews','[]'))
        date=pd.Timestamp(override['label_available_at'])
        if pd.isna(date) or date.tz is not None:raise ValueError('Invalid review availability date')
        if any(pd.Timestamp(r['available_at'])==date and r['type']!=override['type'] for r in history):
            raise ValueError('Conflicting event reviews at the same availability date')
        history.append(dict(type=override['type'],available_at=str(date),reason=override.get('reason','')))
        history.sort(key=lambda r:pd.Timestamp(r['available_at']));latest=history[-1]
        row.update(type=latest['type'],review_type=latest['type'],verified=True,reviews=json.dumps(history,ensure_ascii=False),
                   review_available_at=pd.Timestamp(latest['available_at']),label_available_at=pd.Timestamp(latest['available_at']),
                   reason=latest['reason'],issue='reviewed',source=row['source'].replace('+review','')+'+review')
    result=pd.DataFrame(rows,columns=EVENT_COLUMNS)
    for c in ('start','stop','end','label_available_at','stop_detected_at','review_available_at'):
        result[c]=pd.to_datetime(result[c])
    return result.sort_values(['start','stop'],kind='stable').reset_index(drop=True)
