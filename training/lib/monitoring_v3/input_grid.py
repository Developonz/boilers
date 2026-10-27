"""One hourly-source contract for training, saved-model checks and the dashboard."""
import numpy as np
import pandas as pd


def normalize_grid(frame,step='1h',numeric_columns=(),state_columns=(),time_col='TimeValue'):
    raw=frame.copy()
    raw.columns=[str(c).strip() for c in raw.columns]
    if raw.columns.has_duplicates:raise ValueError('Duplicate source channel names after trimming whitespace')
    values=raw[time_col] if time_col in raw else pd.Series(raw.index)
    if values.astype(str).str.match(r'^\d{1,2}[./]\d{1,2}[./]\d{2,4}(?:\s|$)').any():
        raise ValueError('Ambiguous day/month timestamps: use datetime values or ISO YYYY-MM-DD timestamps')
    if time_col in raw:
        time=pd.to_datetime(raw.pop(time_col),errors='coerce',format='mixed')
    else:time=pd.to_datetime(raw.index,errors='coerce',format='mixed')
    raw.index=pd.DatetimeIndex(time,name=time_col).as_unit('ns')
    if raw.index.tz is not None:raise ValueError('Use one agreed timezone and convert source timestamps to local naive time before ML')
    invalid=int(raw.index.isna().sum());raw=raw.loc[raw.index.notna()]
    if raw.empty:raise ValueError('No valid timestamps')
    if not raw.index.equals(raw.index.floor(step)):
        raise ValueError('Source timestamps must be aligned with '+step+'; prepare matching intervals before ML')
    duplicates=int(raw.index.duplicated().sum())
    states=set(state_columns)
    for c in set(numeric_columns)&set(raw):
        if pd.api.types.is_bool_dtype(raw[c]):raw[c]=raw[c].to_numpy(dtype=float,na_value=np.nan)
        else:raw[c]=pd.to_numeric(raw[c].astype(str).str.replace('\u00a0','',regex=False)
            .str.replace(' ','',regex=False).str.replace(',','.',regex=False),errors='coerce')
    raw=raw.sort_index(kind='stable')
    numeric=[c for c in raw if c not in states and pd.api.types.is_numeric_dtype(raw[c])]
    categorical=[c for c in raw if c not in numeric]
    parts=[]
    if numeric:parts.append(raw[numeric].replace([np.inf,-np.inf],np.nan).resample(step).mean())
    if categorical:parts.append(raw.loc[~raw.index.duplicated(keep='last'),categorical].asfreq(step))
    return pd.concat(parts,axis=1).reindex(columns=raw.columns),dict(
        invalid_timestamps=invalid,duplicate_timestamps=duplicates,raw_rows=len(raw))
