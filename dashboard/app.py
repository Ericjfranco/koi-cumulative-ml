import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine, text
from src.config import DATABASE_URL, GOLD_SCHEMA

st.set_page_config(page_title='KOI Exoplanet Analytics', page_icon='🪐', layout='wide')
st.title('KOI Cumulative — Analytics & Machine Learning')
st.caption('NASA Exoplanet Archive | Classificação de candidatos a exoplanetas')

@st.cache_data(ttl=300)
def load_gold():
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            data = pd.read_sql_query(text(f'SELECT * FROM "{GOLD_SCHEMA}".koi_analytics'), conn)
            summary = pd.read_sql_query(text(f'SELECT * FROM "{GOLD_SCHEMA}".koi_disposition_summary'), conn)
        return data, summary
    finally:
        engine.dispose()

try:
    df, summary = load_gold()
except Exception as exc:
    st.error(f'Não foi possível consultar a Camada Gold: {exc}')
    st.info('Execute primeiro a ingestão e depois `python -m src.gold`.')
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric('Registros analíticos', f'{len(df):,}'.replace(',', '.'))
c2.metric('Classes de disposition', str(df['koi_disposition'].nunique()))
c3.metric('Features numéricas', str(len(df.select_dtypes(include='number').columns)))

left, right = st.columns(2)
with left:
    st.subheader('Distribuição das classificações')
    st.plotly_chart(px.pie(summary, names='koi_disposition', values='records', hole=.45), use_container_width=True)
with right:
    st.subheader('Quantidade por classificação')
    st.plotly_chart(px.bar(summary, x='koi_disposition', y='records', color='koi_disposition'), use_container_width=True)

numeric = [c for c in ['koi_period', 'koi_prad', 'koi_teq', 'koi_depth', 'koi_model_snr', 'koi_steff'] if c in df.columns]
if numeric:
    st.subheader('Distribuição das variáveis físicas')
    selected = st.selectbox('Selecione uma variável', numeric)
    plot_df = df[[selected, 'koi_disposition']].copy()
    plot_df[selected] = pd.to_numeric(plot_df[selected], errors='coerce')
    plot_df = plot_df.dropna()
    st.plotly_chart(px.histogram(plot_df, x=selected, color='koi_disposition', marginal='box', barmode='overlay'), use_container_width=True)

st.subheader('Qualidade dos dados')
missing = (
    df.isna()
    .mean()
    .mul(100)
    .sort_values(ascending=False)
    .rename('missing_pct')
    .reset_index()
    .rename(columns={'index': 'column'})
)
st.plotly_chart(px.bar(missing, x='column', y='missing_pct', title='Percentual de valores ausentes por coluna'), use_container_width=True)

st.subheader('Comparação dos modelos')
comparison_path = Path(os.getenv('REPORTS_DIR', 'reports')) / 'model_comparison.csv'
if comparison_path.exists():
    comparison = pd.read_csv(comparison_path)
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(comparison, x='model', y=['accuracy', 'balanced_accuracy', 'f1_macro'], barmode='group'), use_container_width=True)
else:
    st.info('Ainda não há resultados. Execute `docker compose run --rm trainer` para treinar e comparar os modelos.')

st.subheader('Amostra dos dados')
st.dataframe(df.head(100), use_container_width=True)
