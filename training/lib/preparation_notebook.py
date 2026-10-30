"""Read-only EDA using the same preparation and event registry as notebooks 02/05."""
from pathlib import Path
from html import escape
import ast
import json
import numpy as np
import pandas as pd
from detector_notebook import table,details,figure_html,show_html


def saved_settings(path):
    """Read literals only; opening a notebook never executes its cells."""
    path=Path(path);out={}
    if not path.exists():return out
    for cell in json.loads(path.read_text(encoding='utf-8-sig')).get('cells',[]):
        if cell.get('cell_type')!='code':continue
        try:tree=ast.parse(''.join(cell['source']))
        except SyntaxError:continue
        for node in tree.body:
            if not isinstance(node,ast.Assign):continue
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ('SETTINGS','SENSOR_RULES'):
                    try:out[target.id]=ast.literal_eval(node.value)
                    except (ValueError,TypeError):
                        raise ValueError('Настройка '+target.id+' содержит выражения; укажите её явно в DATA_SETTINGS.')
    return out


def emit(title,body,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    html=details(title,body)
    (folder/'report.html').write_text(html,encoding='utf-8');show_html(html)


def save_table(frame,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(path,index=False,encoding='utf-8-sig')


def stop_tables(context,cfg):
    """Keep physical stop, detection, restart and document dates separate."""
    c=context;ix=c['frame'].index;step=pd.Timedelta(cfg['step'])
    running=ix[c['regime']['mode'].isin(['работа','пуск'])]
    rows=[];documents=[]
    for event in c['events'].itertuples():
        physical=bool(event.observed_stop) and pd.notna(event.stop)
        future=running[running>event.stop] if physical else pd.DatetimeIndex([])
        restart=future[0] if len(future) else pd.NaT
        right=restart if pd.notna(restart) else ix[-1]+step
        span=(ix>=event.stop)&(ix<right) if physical else np.zeros(len(ix),bool)
        docs=json.loads(event.documents) if isinstance(event.documents,str) and event.documents else []
        first_document=min((pd.Timestamp(d['start']) for d in docs),default=pd.NaT)
        rows.append(dict(unit=c['unit'],event_id=event.event_id,type=event.type,
            physical_stop=event.stop,detected_at=event.stop_detected_at,restart=restart,
            observed_stop=physical,no_ticket=bool(physical and not docs),
            missing_physical_stop=not physical,conflicting_tickets=event.issue=='conflicting_document_types',
            documents=len(docs),first_document=first_document,
            document_delay_hours=float((first_document-event.stop)/pd.Timedelta('1h')) if physical and pd.notna(first_document) else np.nan,
            physical_duration_hours=float((right-event.stop)/pd.Timedelta('1h')) if physical else np.nan,
            open_physical_stop=bool(physical and pd.isna(restart)),
            work_label_while_stopped_hours=float((span&c['frame']['Состояние'].eq('Работа')).sum()*step/pd.Timedelta('1h')),
            unknown_regime_hours=float((span&c['regime']['mode'].eq('неизвестно')).sum()*step/pd.Timedelta('1h')),
            issue=event.issue,reviewed=bool(event.verified)))
        for doc in docs:
            documents.append(dict(event_id=event.event_id,type=doc['type'],document_start=pd.Timestamp(doc['start']),
                document_end=pd.Timestamp(doc['end']),physical_stop=event.stop,
                document_delay_hours=float((pd.Timestamp(doc['start'])-event.stop)/pd.Timedelta('1h')) if physical else np.nan))
    columns=['unit','event_id','type','physical_stop','detected_at','restart','observed_stop','no_ticket',
             'missing_physical_stop','conflicting_tickets','documents','first_document','document_delay_hours',
             'physical_duration_hours','open_physical_stop','work_label_while_stopped_hours','unknown_regime_hours','issue','reviewed']
    return pd.DataFrame(rows,columns=columns),pd.DataFrame(documents)


def stops_report(contexts,cfg,output,options):
    import matplotlib.pyplot as plt
    all_rows=[]
    for unit,c in contexts.items():
        folder=Path(output)/unit/'stops';stops,docs=stop_tables(c,cfg);all_rows.append(stops)
        save_table(stops,folder/'stops.csv');save_table(docs,folder/'documents.csv')
        no_ticket=stops.loc[stops.no_ticket.eq(True)]
        orphan=stops.loc[stops.missing_physical_stop.eq(True)]
        body=table('Остановы без заявок: все найденные',no_ticket)+table('Заявки без подтверждённого физического останова',orphan)
        body+=table('Все события: физические и документальные даты',stops)+table('Все привязанные отметки ремонтов',docs)
        body+='<p>Останов без заявки получает неизвестную причину. Он не становится аварией или нормой автоматически. '
        body+='physical_stop — первый низкий отсчёт подтверждённого останова, detected_at — когда накопилось подтверждение, '
        body+='restart — возврат рабочего сигнала. Разница дат заявки и останова сама по себе не доказывает ошибку часового пояса.</p>'
        candidates=stops.loc[stops.observed_stop.eq(True)].sort_values(['no_ticket','physical_duration_hours'],ascending=False)
        for event in candidates.head(options['max_stop_plots']).itertuples():
            left=event.physical_stop-pd.Timedelta(hours=options['stop_before_hours'])
            right=(event.restart if pd.notna(event.restart) else event.physical_stop)+pd.Timedelta(hours=options['stop_after_hours'])
            d=c['cleaned'].loc[left:right];regime=c['regime'].reindex(d.index)
            fig,axes=plt.subplots(3,1,figsize=(14,7),sharex=True)
            axes[0].plot(d.index,d[c['load']],lw=.9,label=c['load'])
            axes[0].axhline(c['threshold'],color='red',ls='--',label=f'Порог работы {c["threshold"]:g}')
            axes[0].axhline(c['threshold']*cfg['regime_policy']['off_fraction'],color='gray',ls=':',label='Граница выключения')
            axes[0].axvline(event.physical_stop,color='black',label='Физический останов')
            if pd.notna(event.detected_at):axes[0].axvline(event.detected_at,color='purple',ls=':',label='Подтверждение')
            axes[0].set_title(f'{unit}: {event.type}; '+('без заявки' if event.no_ticket else 'есть отметка ремонта'))
            for key,label in [('mill','мельницы'),('feeder','питатели'),('draft','тяга')]:
                col='evidence::'+key+'_active'
                if col in regime and regime[col].notna().any():
                    axes[1].step(d.index,regime[col],where='post',label=label)
            if not axes[1].get_legend_handles_labels()[0]:
                axes[1].text(.5,.5,'Нет сопоставленных токов приводов',ha='center',transform=axes[1].transAxes)
            axes[1].set_ylabel('Включено приводов')
            axes[2].step(d.index,regime['mode'].map({'стоит':0,'неизвестно':1,'пуск':2,'работа':3}),where='post')
            axes[2].set_yticks([0,1,2,3],['стоит','неизвестно','пуск','работа'])
            linked=docs.loc[docs.event_id.eq(event.event_id)] if len(docs) else docs
            for doc in linked.itertuples():
                for ax in axes:ax.axvspan(doc.document_start,doc.document_end,color='red' if doc.type=='Авария' else 'gray',alpha=.13)
            for ax in axes:
                ax.grid(alpha=.2)
                if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=8,loc='upper left')
            fig.tight_layout();body+=details(str(event.physical_stop)+' · '+event.issue,
                figure_html(fig,folder/f'stop_{event.Index}.png'))
        emit(unit+' · остановы и заявки',body,folder)
    result=pd.concat(all_rows,ignore_index=True) if all_rows else pd.DataFrame()
    if len(result):
        overview=result.groupby('unit').agg(physical_stops=('observed_stop','sum'),stops_without_tickets=('no_ticket','sum'),
            tickets_without_stop=('missing_physical_stop','sum'),conflicts=('conflicting_tickets','sum')).reset_index()
        show_html(table('Итоги сопоставления остановов и заявок',overview,True))
    return result


def training_availability(context,cfg):
    from monitoring_v3.data import healthy
    from monitoring_v3.partitioning import partition_history
    c=context;ix=c['frame'].index
    reference=pd.Timestamp(c['preparation_reference_end'])
    begin=reference.to_period('M').to_timestamp()
    if begin<reference:begin+=pd.offsets.MonthBegin(1)
    begin=min(pd.Timestamp(cfg['first_test']),begin)
    cutoffs=([begin]+list(pd.date_range(begin+pd.offsets.MonthBegin(1),ix[-1],freq='MS'))) if begin<=ix[-1] else []
    rows=[]
    inputs=[k for k,r in c['roles'].items() if r in ('load','condition','state')]
    targets=c['cleaned'][c['targets']]
    coverage=targets.notna().mean(axis=1).ge(cfg['minimum_coverage'])
    for cutoff in cutoffs:
        stage=(ix<cutoff);counts={}
        def keep(name,mask):
            nonlocal stage
            before=int(stage.sum());stage&=mask
            counts[name+'_excluded']=before-int(stage.sum());counts[name+'_remaining']=int(stage.sum())
        total=int(stage.sum())
        keep('regime',c['eligible'])
        keep('inputs',c['cleaned'][inputs].notna().mean(axis=1).ge(cfg['minimum_coverage']))
        keep('work_label',c['frame']['Состояние'].eq('Работа'))
        keep('maturity',ix<cutoff-pd.Timedelta(days=cfg['admission_delay_days']))
        keep('events',healthy(c,cutoff,cfg))
        keep('target_coverage',coverage)
        index=pd.MultiIndex.from_arrays([ix[stage],np.repeat(c['unit'],int(stage.sum()))],names=['time','unit'])
        # Only the index is needed: no estimator or feature matrix is built.
        _,_,_,_,_,audit=partition_history(pd.DataFrame(index=index),cfg)
        status=('ready_for_initial_fit' if audit['train']>=cfg['min_train_rows'] and audit['calibration']>=cfg['calibration_rows'] else
                'insufficient_train' if audit['train']<cfg['min_train_rows'] else 'insufficient_calibration')
        rows.append(dict(unit=c['unit'],cutoff=cutoff,source_rows=total,admitted_rows=int(stage.sum()),
            train_rows=audit['train'],calibration_rows=audit['calibration'],status=status,**counts))
    return pd.DataFrame(rows)


def availability_report(contexts,cfg,output):
    import matplotlib.pyplot as plt
    rows=[]
    for unit,c in contexts.items():
        frame=training_availability(c,cfg);rows.append(frame);folder=Path(output)/unit/'availability'
        save_table(frame,folder/'availability.csv')
        body=table('Все месяцы и последовательный отсев строк',frame)
        body+='<p>Это проверка возможности первоначального local-обучения на каждом срезе с реальным healthy/partition_history из 02. '
        body+='Обучение не запускается, состояние обновлений и результаты проверки кандидата не имитируются. '
        body+='Для pooled нужны общие совместимые каналы и суммарная история группы. '
        body+='Причины отсева взаимоисключающие благодаря порядку проверок; строки и часы совпадают только при шаге 1 час.</p>'
        if len(frame):
            fig,ax=plt.subplots(figsize=(13,4))
            for col in ('source_rows','admitted_rows','train_rows','calibration_rows'):ax.plot(frame.cutoff,frame[col],marker='.',label=col)
            ax.axhline(cfg['min_train_rows'],color='gray',ls='--',label='Минимум train')
            ax.set_ylabel('Строк');ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout()
            body+=figure_html(fig,folder/'availability.png')
        emit(unit+' · доступная история для первоначального обучения',body,folder)
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()


def sensor_quality(context,cfg):
    c=context;operating=c['eligible'];rows=[]
    for sensor,role in c['roles'].items():
        if sensor not in c['cleaned']:continue
        raw=c['frame'][sensor].loc[operating];clean=c['cleaned'][sensor].loc[operating]
        counts=raw.value_counts(dropna=True)
        row=dict(sensor=sensor,role=role,operating_rows=len(raw),known_rows=int(raw.notna().sum()),
            unique_values=int(raw.nunique()),dominant_fraction=float(counts.iloc[0]/counts.sum()) if len(counts) else np.nan,
            raw_missing_fraction=float(raw.isna().mean()),clean_missing_fraction=float(clean.isna().mean()))
        held=c['changes'].get(sensor+'::held_hours')
        row['max_observed_plateau_hours']=float(held.loc[operating].max()) if held is not None else np.nan
        for flag in ('frozen','spike','masked'):
            key=(sensor,flag)
            row[flag+'_rows']=int(c['input_quality'].loc[operating,key].sum()) if key in c['input_quality'] else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def quality_report(contexts,cfg,output):
    import matplotlib.pyplot as plt
    rows=[]
    for unit,c in contexts.items():
        frame=sensor_quality(c,cfg);rows.append(frame.assign(unit=unit));folder=Path(output)/unit/'quality'
        save_table(frame,folder/'sensor_quality.csv')
        body=table('Все датчики: доступность, плато, скачки и реальная маскировка',frame)
        rare=frame.loc[frame.max_observed_plateau_hours.ge(24)|frame.dominant_fraction.ge(.98)]
        body+=table('Долго постоянные и редко меняющиеся каналы: требуют разбора, не исключаются автоматически',rare)
        numeric=c['cleaned'][c['targets']].where(c['eligible']).resample('MS').count()
        possible=c['eligible'].astype(int).resample('MS').sum().replace(0,np.nan)
        coverage=numeric.div(possible,axis=0)
        if len(coverage) and len(coverage.columns):
            selected=coverage.mean().nsmallest(min(30,len(coverage.columns))).index
            fig,ax=plt.subplots(figsize=(13,max(4,len(selected)*.2)))
            im=ax.imshow(coverage[selected].T.to_numpy(),aspect='auto',vmin=0,vmax=1,cmap='RdYlGn')
            ax.set_yticks(range(len(selected)),selected,fontsize=7)
            ticks=np.arange(0,len(coverage),max(1,len(coverage)//14))
            ax.set_xticks(ticks,coverage.index[ticks].strftime('%Y-%m'),rotation=45,ha='right')
            ax.set_title('Доступность измерений в допустимом рабочем режиме: 30 худших каналов')
            fig.colorbar(im,ax=ax,label='Доля известных значений');fig.tight_layout()
            body+=figure_html(fig,folder/'coverage.png')
        emit(unit+' · качество датчиков',body,folder)
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()


def reference_residuals(context,cfg):
    """A descriptive load-bin baseline, never a fitted anomaly detector."""
    from monitoring_v3.data import healthy
    c=context;normal=healthy(c,c['frame'].index[-1]+pd.Timedelta(cfg['step']),cfg)
    reference=normal&(c['frame'].index<c['preparation_reference_end'])
    load=c['cleaned'][c['load']];ref_load=load.loc[reference].dropna()
    if len(ref_load)<20:return pd.DataFrame(index=load.index),normal
    edges=np.unique(ref_load.quantile(np.linspace(0,1,6)).to_numpy())
    edges=np.r_[-np.inf,edges[1:-1],np.inf]
    if len(edges)<2:edges=np.array([-np.inf,np.inf])
    bins=pd.cut(load,edges,labels=False)
    numeric=c['cleaned'][c['targets']]
    centers=numeric.loc[reference].groupby(bins.loc[reference],observed=True).median()
    expected=centers.reindex(bins.to_numpy());expected.index=numeric.index
    spread=(numeric.loc[reference].quantile(.75)-numeric.loc[reference].quantile(.25))/1.349
    spread=spread.where(spread.gt(1e-10))
    z=(numeric-expected).div(spread).where(c['eligible'])
    return z,normal


def relationships_report(contexts,cfg,output,options):
    import matplotlib.pyplot as plt
    from monitoring_v3.data import healthy
    reports=[]
    for unit,c in contexts.items():
        folder=Path(output)/unit/'relationships'
        mask=healthy(c,c['frame'].index[-1]+pd.Timedelta(cfg['step']),cfg)
        d=c['cleaned'].loc[mask]
        if len(d)>options['max_analysis_rows']:d=d.iloc[::int(np.ceil(len(d)/options['max_analysis_rows']))]
        target=d[c['targets']].apply(pd.to_numeric,errors='coerce')
        corr=target.corr(method='spearman',min_periods=20)
        pairs=[]
        for i,a in enumerate(corr):
            for b in corr.columns[i+1:]:
                rho=corr.loc[a,b]
                if np.isfinite(rho) and abs(rho)>=options['correlation_threshold']:
                    pairs.append(dict(sensor_a=a,sensor_b=b,spearman=rho,paired_rows=int(target[[a,b]].notna().all(axis=1).sum())))
        pairs=pd.DataFrame(pairs,columns=['sensor_a','sensor_b','spearman','paired_rows'])
        if len(pairs):pairs=pairs.assign(absolute_spearman=pairs.spearman.abs()).sort_values('absolute_spearman',ascending=False)
        load=pd.to_numeric(d[c['load']],errors='coerce')
        load_corr=pd.DataFrame([dict(sensor=s,load_spearman=target[s].corr(load,method='spearman'),
            paired_rows=int((target[s].notna()&load.notna()).sum())) for s in target if target[s].notna().sum()>=20])
        save_table(pairs,folder/'correlated_pairs.csv');save_table(load_corr,folder/'load_correlations.csv')
        body=f'<p>Использовано {len(d)} строк допустимой нормы; для тяжёлых расчётов максимум {options["max_analysis_rows"]} '
        body+='равномерно выбранных по истории строк. Корреляция показывает совместное движение, а не причинность или тождество физических точек. '
        body+='Связи не записываются автоматически в related_inputs.</p>'
        body+=table('Связь датчиков с нагрузкой',load_corr)+table('Сильные связи и возможные дубли',pairs)
        selected=list(corr.columns[:options['max_heatmap_sensors']])
        if selected:
            fig,ax=plt.subplots(figsize=(10,8));im=ax.imshow(corr.loc[selected,selected],vmin=-1,vmax=1,cmap='coolwarm')
            ax.set_xticks(range(len(selected)),selected,rotation=90,fontsize=6);ax.set_yticks(range(len(selected)),selected,fontsize=6)
            fig.colorbar(im,ax=ax,label='Spearman');fig.tight_layout();body+=figure_html(fig,folder/'correlations.png')
        categories=[s for s,r in c['roles'].items() if r=='state' and s in d]
        category_rows=[]
        for state in categories:
            for value,g in d.groupby(state,dropna=False,observed=True):
                category_rows.append(dict(sensor=state,value=str(value),rows=len(g),load_median=float(pd.to_numeric(g[c['load']],errors='coerce').median())))
        body+=table('Состояния оборудования: частота и типичная нагрузка',pd.DataFrame(category_rows))
        emit(unit+' · связи датчиков и режимов',body,folder);reports.append(load_corr.assign(unit=unit))
    return pd.concat(reports,ignore_index=True) if reports else pd.DataFrame()


def drift_report(contexts,cfg,output,options):
    import matplotlib.pyplot as plt
    reports=[]
    for unit,c in contexts.items():
        folder=Path(output)/unit/'drift';z,normal=reference_residuals(c,cfg)
        if z.empty:
            emit(unit+' · изменение нормы','<p>В опорном периоде меньше 20 допустимых строк: описательную базу не строим.</p>',folder);continue
        monthly=z.where(normal).resample('MS').median()
        raw=c['cleaned'][c['targets']].where(normal).resample('MS').median()
        save_table(monthly.rename_axis('month').reset_index(),folder/'load_adjusted_monthly_shift.csv')
        body=table('Медианные значения по месяцам в единицах датчиков',raw.rename_axis('month').reset_index())
        body+=table('Сдвиг после грубой поправки на нагрузку',monthly.rename_axis('month').reset_index())
        body+='<p>Это описательная оценка: в начальном опорном периоде считаем медиану датчика в пяти диапазонах нагрузки; '
        body+='вычитаем её и делим на IQR/1.349 опорного периода. Не учитываем все состояния оборудования и вид топлива. '
        body+='Сдвиг не равен неисправности; он может отражать другой режим. Для постоянного опорного датчика нормированный сдвиг не определён. '
        body+='Эта оценка не является ошибкой модели из 02 и не меняет её пороги.</p>'
        selected=monthly.abs().max().dropna().nlargest(min(20,len(monthly.columns))).index
        if len(selected):
            fig,ax=plt.subplots(figsize=(13,max(4,.25*len(selected))))
            im=ax.imshow(monthly[selected].T,aspect='auto',cmap='coolwarm',vmin=-3,vmax=3)
            ax.set_yticks(range(len(selected)),selected,fontsize=7)
            ticks=np.arange(0,len(monthly),max(1,len(monthly)//14))
            ax.set_xticks(ticks,monthly.index[ticks].strftime('%Y-%m'),rotation=45,ha='right');fig.colorbar(im,ax=ax,label='Сдвиг / опорный разброс')
            fig.tight_layout();body+=figure_html(fig,folder/'monthly_shift.png')
        tails=[]
        segment=(~c['eligible']|~c['eligible'].shift(fill_value=False)|~z.index.to_series().diff().eq(pd.Timedelta(cfg['step']))).cumsum()
        for sensor in z:
            values=z[sensor].where(normal).dropna()
            if len(values)<20:continue
            row=dict(sensor=sensor,rows=len(values),absolute_q95=float(values.abs().quantile(.95)),
                     fraction_above_3=float(values.abs().gt(3).mean()))
            for hours in (1,24,168):
                lag=int(pd.Timedelta(hours=hours)/pd.Timedelta(cfg['step']))
                past=z[sensor].shift(lag).where(segment.eq(segment.shift(lag)))
                row[f'autocorrelation_{hours}h']=z[sensor].where(normal).corr(past.where(normal))
            tails.append(row)
        tail_frame=pd.DataFrame(tails);save_table(tail_frame,folder/'tails_and_memory.csv')
        body+=table('Хвосты распределения и память: приближённый остаток EDA',tail_frame)
        body+='<p>Автокорреляции не соединяют разные рабочие участки через остановы и разрывы. '
        body+='Доля выше 3 — наблюдаемая частота; теоретическое правило 3σ здесь не применяется для выбора тревог.</p>'
        emit(unit+' · изменение нормы, хвосты и память',body,folder);reports.append(tail_frame.assign(unit=unit))
    return pd.concat(reports,ignore_index=True) if reports else pd.DataFrame()


def classifier_label_readiness(context,cfg,horizons,options=None):
    """The inexpensive eligibility rules from 05, without its 9000-column matrix."""
    from direct_risk import direct_options
    from monitoring_v3.interpretation import risk_labels
    c=context;o=direct_options(options)
    numeric=[s for s,r in c['roles'].items() if r in ('target','condition','load')]
    coverage=c['cleaned'][numeric].notna().mean(axis=1)
    auto=any(a.get('source')=='early_past' and a.get('threshold') is not None and pd.notna(a.get('threshold'))
             for a in c.get('regime_spec',{}).get('audit',[]))
    reference=c.get('threshold_source')=='initial_past' or auto
    eligible=c['regime']['mode'].isin(['работа','пуск'] if o['include_startup'] else ['работа'])
    eligible&=~c['frame'].get('_Тип',pd.Series('',index=eligible.index)).isin(['ИСП','ЗРР'])
    eligible&=coverage.ge(o['minimum_numeric_coverage'])
    if reference:eligible&=eligible.index>=c['preparation_reference_end']
    rows=[]
    for horizon in horizons:
        end=c['frame'].index[-1]
        y,g,audit=risk_labels(eligible.index[eligible],c['events'],horizon,end,end,
            observed=c['cleaned'][c['load']].notna(),min_lead_hours=o['min_lead_hours'],return_audit=True)
        rows.append(dict(unit=c['unit'],horizon_hours=horizon,candidate_rows=int(eligible.sum()),
            positive_rows=int(y.eq(1).sum()),negative_rows=int(y.eq(0).sum()),unknown_rows=int(y.isna().sum()),
            positive_events=int(g.loc[y.eq(1)].nunique()),**{'audit_'+k:v for k,v in audit.items()}))
    return pd.DataFrame(rows)


def readiness_report(contexts,cfg,output,horizons,options=None):
    frames=[classifier_label_readiness(c,cfg,horizons,options) for c in contexts.values()]
    result=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    save_table(result,Path(output)/'classifier_labels.csv')
    show_html(table('Потенциальные исходы для 05: до разделения train/validation/test',result,True)+
        '<p>Это численность исходов при правилах допуска 05, без построения большого окна признаков и без обучения. '
        'Неизвестные остановы и неполное будущее не считаются отрицательным классом. '
        'Большое число положительных часов не заменяет несколько независимых аварий; '
        'реальная пригодность каждого train/validation/test определяется в 05.</p>')
    return result


def before_stops_report(contexts,cfg,output,options):
    import matplotlib.pyplot as plt
    result=[]
    for unit,c in contexts.items():
        folder=Path(output)/unit/'before_stops';z,_=reference_residuals(c,cfg)
        events=c['events'].loc[c['events'].observed_stop.eq(True)&c['events'].stop.notna()]
        body='<p>Окна привязаны к физическому останову, а не дате открытия заявки. '
        body+='Остановы без заявки показаны с неизвестной причиной. '
        body+='Приближённое отклонение EDA учитывает только диапазон нагрузки; это не предсказание обученной модели. '
        body+='Рейтинг помогает выбрать каналы для разбора и не доказывает причину аварии.</p>'
        plotted=0;rare_rows=[]
        for event in events.itertuples():
            left=event.stop-pd.Timedelta(hours=options['pre_stop_hours'])
            part=z.loc[(z.index>=left)&(z.index<event.stop)]
            rank=part.abs().quantile(.9).dropna().sort_values(ascending=False) if not part.empty else pd.Series(dtype=float)
            for sensor,value in rank.head(10).items():
                result.append(dict(unit=unit,event_id=event.event_id,type=event.type,stop=event.stop,sensor=sensor,
                                   absolute_q90=float(value),observed_rows=int(part[sensor].notna().sum())))
            for sensor in c['targets']:
                # This audit must also reveal an important change after a
                # plateau that an explicit cleaning policy happened to mask.
                values=pd.to_numeric(c['frame'][sensor],errors='coerce').loc[left:event.stop].iloc[:-1]
                changed=values.notna()&values.shift().notna()&values.ne(values.shift())
                held=c['changes'].get(sensor+'::held_hours')
                if held is not None:
                    previous=held.shift().reindex(values.index)
                    for t in values.index[changed&previous.ge(24)]:
                        rare_rows.append(dict(event_id=event.event_id,type=event.type,stop=event.stop,sensor=sensor,
                            changed_at=t,previous_plateau_hours=float(previous.loc[t]),
                            old_value=float(values.shift().loc[t]),new_value=float(values.loc[t])))
            if plotted>=options['max_event_plots'] or rank.empty:continue
            chosen=list(rank.head(4).index);fig,axes=plt.subplots(len(chosen),1,figsize=(14,2.4*len(chosen)),sharex=True,squeeze=False)
            for ax,sensor in zip(axes[:,0],chosen):
                hours=(part.index-event.stop)/pd.Timedelta('1h')
                ax.plot(hours,part[sensor],lw=.8);ax.axvline(0,color='red',ls='--');ax.set_title(sensor,fontsize=9);ax.grid(alpha=.2)
            axes[-1,0].set_xlabel('Часы до физического останова');fig.suptitle(unit+' · '+event.type+' · '+str(event.stop))
            fig.tight_layout();body+=details(str(event.stop)+' · '+event.type,figure_html(fig,folder/f'event_{plotted}.png'));plotted+=1
        ranked=pd.DataFrame([r for r in result if r['unit']==unit]);rare=pd.DataFrame(rare_rows)
        save_table(ranked,folder/'pre_stop_channels.csv');save_table(rare,folder/'plateau_changes_before_stops.csv')
        body=table('Сильные отклонения перед каждым остановом: первые 10 каналов',ranked)+table('Изменения после плато не короче 24 часов',rare)+body
        body+='<p>Таблица плато использует исходное правило подготовки held_hours. Такие изменения могут быть полезным сигналом; '
        body+='автоматического удаления датчиков из-за долгого постоянного значения нет.</p>'
        emit(unit+' · поведение перед остановами',body,folder)
    return pd.DataFrame(result)


def balance_report(contexts,cfg,output,options):
    import matplotlib.pyplot as plt
    rows=[]
    for unit,c in contexts.items():
        chosen=options.get('balance_sensors',{}).get(unit,{})
        water,steam=chosen.get('water'),chosen.get('steam')
        if not water or not steam:
            show_html(details(unit+' · баланс вода–пар','<p>Для этой необязательной проверки задайте два проверенных канала в EDA.balance_sensors.</p>'));continue
        if water not in c['cleaned'] or steam not in c['cleaned']:raise ValueError(unit+': не найдены каналы баланса')
        units=c.get('declared_units',{})
        if not units.get(water) or units.get(water)!=units.get(steam):
            show_html(details(unit+' · баланс вода–пар','<p>Единицы потоков не заданы или различаются: вычитание не выполняется.</p>'));continue
        from monitoring_v3.alarm_scores import alarm_stream
        w,s=c['cleaned'][water],c['cleaned'][steam]
        value=((w-s)/s.where(s.gt(0))*100).where(c['eligible'])
        policy={**cfg,'persist_hours':options['balance_persist_hours']}
        stream,episodes=alarm_stream(value.abs()/options['balance_threshold_percent'],policy)
        if len(episodes):
            episodes=episodes.assign(unit=unit,water_sensor=water,steam_sensor=steam);rows.append(episodes)
        folder=Path(output)/unit/'balance';save_table(episodes,folder/'episodes.csv')
        fig,ax=plt.subplots(figsize=(14,4));ax.plot(value.index,value,lw=.7)
        for sign in (-1,1):ax.axhline(sign*options['balance_threshold_percent'],color='red',ls='--')
        ax.set_ylabel('(вода − пар) / пар, %');ax.set_title(unit+' · наблюдаемый баланс');ax.grid(alpha=.2);fig.tight_layout()
        emit(unit+' · баланс вода–пар',table('Подтверждённые эпизоды расхождения',episodes)+figure_html(fig,folder/'balance.png')+
            '<p>Разность потоков включает продувку, отборы, погрешности и накопление среды. Эпизод не является готовым диагнозом утечки.</p>',folder)
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()
