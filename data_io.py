from pathlib import Path
import pandas as pd

DATE_NAMES = {'date', 'month', 'datetime', 'timestamp'}

def _normalise_headers(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        if str(c).lower() in DATE_NAMES or str(c).lower().endswith('_date'):
            df[c] = pd.to_datetime(df[c], errors='coerce')
            continue
        if df[c].dtype == 'object':
            parsed = pd.to_numeric(df[c].astype(str).str.replace(',', '', regex=False), errors='coerce')
            nonmissing = df[c].notna().sum()
            if nonmissing and parsed.notna().sum() / nonmissing >= 0.70:
                df[c] = parsed
    return df

def read_tabular_upload(file, sheet_name=None):
    name = getattr(file, 'name', str(file)).lower()
    if name.endswith(('.xlsx', '.xls')):
        raw = pd.read_excel(file, sheet_name=sheet_name or 0)
        if isinstance(raw, dict):
            return {k: _normalise_headers(v) for k, v in raw.items()}
        # Support workbook sheets with explanatory preambles above the header row.
        if raw.columns.astype(str).str.contains('Unnamed').all() or raw.iloc[:, 0].astype(str).str.lower().isin(DATE_NAMES).sum() == 0:
            file.seek(0) if hasattr(file, 'seek') else None
            raw0 = pd.read_excel(file, sheet_name=sheet_name or 0, header=None)
            idx = next((i for i, v in enumerate(raw0.iloc[:, 0].astype(str)) if v.strip().lower() in DATE_NAMES), None)
            if idx is not None:
                raw = raw0.iloc[idx+1:].copy(); raw.columns = raw0.iloc[idx].tolist()
                raw = raw.dropna(how='all')
        return _normalise_headers(raw)
    return _normalise_headers(pd.read_csv(file))

def load_workbook(path):
    xls = pd.ExcelFile(path)
    out = {}
    for sheet in xls.sheet_names:
        raw = pd.read_excel(path, sheet_name=sheet, header=None)
        idx = next((i for i, v in enumerate(raw.iloc[:, 0].astype(str)) if v.strip().lower() in DATE_NAMES), None)
        if idx is not None:
            df = raw.iloc[idx+1:].copy(); df.columns = raw.iloc[idx].tolist(); df = df.dropna(how='all').reset_index(drop=True)
            out[sheet] = _normalise_headers(df)
        else:
            # Parse supporting audit/register/plan tables whose headers are not date fields.
            header_idx = next((i for i in range(min(8, len(raw))) if raw.iloc[i].notna().sum() >= 3), None)
            if header_idx is not None:
                df = raw.iloc[header_idx+1:].copy(); df.columns = raw.iloc[header_idx].tolist(); df = df.dropna(how='all').reset_index(drop=True)
                out[sheet] = _normalise_headers(df)
            else:
                out[sheet] = pd.read_excel(path, sheet_name=sheet)
    return out

def data_audit(df, date_col=None, label='dataset'):
    d = df.copy()
    date_col = date_col or next((c for c in d.columns if str(c).lower() in DATE_NAMES), None)
    if date_col and date_col in d:
        d[date_col] = pd.to_datetime(d[date_col], errors='coerce')
        dates = d[date_col].dropna().sort_values()
        start = dates.min() if len(dates) else None; end = dates.max() if len(dates) else None
        duplicate_dates = int(d[date_col].duplicated().sum())
        freq = 'monthly' if date_col.lower() == 'month' else 'daily/irregular'
    else:
        start=end=None; duplicate_dates=0; freq='unknown'
    missing = d.isna().sum()
    numeric = d.select_dtypes(include='number')
    outliers = {}
    for c in numeric:
        q1,q3=numeric[c].quantile([.25,.75]); iqr=q3-q1
        outliers[c]=int(((numeric[c] < q1-1.5*iqr)|(numeric[c] > q3+1.5*iqr)).sum()) if pd.notna(iqr) else 0
    return {'dataset':label,'rows':len(d),'columns':len(d.columns),'date_column':date_col,'frequency':freq,'start':start,'end':end,'duplicate_rows':int(d.duplicated().sum()),'duplicate_dates':duplicate_dates,'missing_cells':int(missing.sum()),'missing_by_column':missing[missing>0].to_dict(),'numeric_outlier_iqr_counts':outliers,'dtypes':d.dtypes.astype(str).to_dict()}
