import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy.engine import URL

load_dotenv()
BASE_DIR = Path(__file__).resolve().parents[1]
POSTGRES_DB = os.getenv('POSTGRES_DB', 'exoplanets')
POSTGRES_USER = os.getenv('POSTGRES_USER', 'postgres')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'postgres')
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
POSTGRES_PORT = int(os.getenv('POSTGRES_PORT', '5432'))
DATABASE_URL = URL.create(
    'postgresql+psycopg2', username=POSTGRES_USER,
    password=POSTGRES_PASSWORD, host=POSTGRES_HOST,
    port=POSTGRES_PORT, database=POSTGRES_DB,
)
TAP_URL = os.getenv('TAP_URL', 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync')
TAP_QUERY = os.getenv('TAP_QUERY', 'select * from cumulative')
RAW_DATA_DIR = Path(os.getenv('RAW_DATA_DIR', str(BASE_DIR / 'data/raw')))
LOG_DIR = Path(os.getenv('LOG_DIR', str(BASE_DIR / 'logs')))
REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '300'))
TABLE_NAME = 'koi_cumulative'
GOLD_SCHEMA = os.getenv('GOLD_SCHEMA', 'gold')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
MLFLOW_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'koi-disposition-classification')
