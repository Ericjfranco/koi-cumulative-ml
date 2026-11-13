"""Build Gold analytical tables from public.koi_cumulative."""
import logging
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.types import Text, Float
from src.config import DATABASE_URL, GOLD_SCHEMA, TABLE_NAME

logger = logging.getLogger(__name__)
NUMERIC_COLUMNS = [
    'koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration', 'koi_depth',
    'koi_prad', 'koi_teq', 'koi_insol', 'koi_model_snr', 'koi_steff',
    'koi_slogg', 'koi_smet', 'koi_srad', 'ra', 'dec', 'koi_kepmag',
]


def build_gold():
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            exists = conn.execute(text("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema='public' AND table_name=:table
                )
            """), {'table': TABLE_NAME}).scalar()
            if not exists:
                raise RuntimeError('Tabela public.koi_cumulative não existe. Execute primeiro o ETL de ingestão.')
            raw = pd.read_sql_query(text(f'SELECT * FROM public."{TABLE_NAME}"'), conn)

        if 'koi_disposition' not in raw.columns:
            raise ValueError('A coluna koi_disposition não existe na tabela Silver.')
        # Mantém apenas colunas utilizadas.
        id_cols = [c for c in ['kepoi_name', 'kepid', 'kepoi_name'] if c in raw.columns]
        selected = list(dict.fromkeys(id_cols + ['koi_disposition'] + [c for c in NUMERIC_COLUMNS if c in raw.columns]))
        gold = raw[selected].copy()
        gold['koi_disposition'] = gold['koi_disposition'].astype('string').str.strip().str.upper()
        for col in NUMERIC_COLUMNS:
            if col in gold.columns:
                gold[col] = pd.to_numeric(gold[col], errors='coerce')
        gold = gold.replace([float('inf'), float('-inf')], pd.NA)
        gold = gold.dropna(subset=['koi_disposition'])
        gold = gold.drop_duplicates()

        summary = (gold.groupby('koi_disposition', dropna=False)
                   .size().rename('records').reset_index())
        summary['percentage'] = summary['records'] / summary['records'].sum() * 100

        with engine.begin() as conn:
            conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{GOLD_SCHEMA}"'))
            dtype = {c: Text() for c in gold.columns if c == 'koi_disposition' or c in id_cols}
            dtype.update({c: Float() for c in gold.columns if c in NUMERIC_COLUMNS})
            gold.to_sql('koi_analytics', conn, schema=GOLD_SCHEMA, if_exists='replace', index=False, dtype=dtype, chunksize=2000)
            summary.to_sql('koi_disposition_summary', conn, schema=GOLD_SCHEMA, if_exists='replace', index=False, chunksize=500)
        logger.info('Gold atualizada: %s registros; %s colunas.', len(gold), len(gold.columns))
        logger.info('Distribuição: %s', summary.to_dict(orient='records'))
        return len(gold)
    finally:
        engine.dispose()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    build_gold()
