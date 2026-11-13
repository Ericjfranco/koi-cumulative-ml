"""Shared feature engineering and target definitions."""
import hashlib
import pandas as pd

TARGET = 'koi_disposition'
# Variables measured from the transit/host star. Explicit allow-list prevents
# accidental inclusion of identifiers and disposition-derived leakage fields.
FEATURE_CANDIDATES = [
    'koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration', 'koi_depth',
    'koi_prad', 'koi_teq', 'koi_insol', 'koi_model_snr', 'koi_steff',
    'koi_slogg', 'koi_smet', 'koi_srad', 'ra', 'dec', 'koi_kepmag',
]
LEAKAGE_COLUMNS = {
    'koi_pdisposition', 'koi_score', 'koi_fpflag_nt', 'koi_fpflag_ss',
    'koi_fpflag_co', 'koi_fpflag_ec', 'koi_disposition',
}


def prepare_dataset(frame: pd.DataFrame):
    if TARGET not in frame.columns:
        raise ValueError(f"Coluna alvo obrigatória não encontrada: {TARGET}")
    available = [c for c in FEATURE_CANDIDATES if c in frame.columns and c not in LEAKAGE_COLUMNS]
    if len(available) < 3:
        raise ValueError(f"Poucas features encontradas. Disponíveis: {available}")

    data = frame[[TARGET] + available].copy()
    data[TARGET] = data[TARGET].astype('string').str.strip().str.upper()
    data = data[data[TARGET].isin(['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'])]
    for col in available:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.replace([float('inf'), float('-inf')], pd.NA)
    # Remove features sem nenhum valor numérico e linhas sem classe.
    available = [c for c in available if data[c].notna().any()]
    data = data.dropna(subset=[TARGET])
    data = data.drop_duplicates()
    if not available:
        raise ValueError('Nenhuma feature numérica válida permaneceu após a limpeza.')
    X = data[available].astype('float64')
    y = data[TARGET].astype(str)
    return X, y


def dataframe_fingerprint(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()
