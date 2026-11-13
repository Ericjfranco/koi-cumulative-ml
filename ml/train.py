"""Train, evaluate, compare and track classification models with MLflow."""
import json
import logging
import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sqlalchemy import create_engine, text
from xgboost import XGBClassifier

from src.config import DATABASE_URL, GOLD_SCHEMA, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI
from ml.features import prepare_dataset, dataframe_fingerprint

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'
logger = logging.getLogger(__name__)
RANDOM_STATE = 666


def load_gold_data():
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(text(f'SELECT * FROM "{GOLD_SCHEMA}".koi_analytics'), conn)
        if df.empty:
            raise ValueError('A tabela Gold está vazia. Execute python -m src.gold primeiro.')
        return df
    finally:
        engine.dispose()


def get_git_revision():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return 'working-tree-or-not-a-git-repository'


def make_models(n_classes):
    return {
        'logistic_regression': LogisticRegression(max_iter=2000, class_weight='balanced'),
        'random_forest': RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=2,
            class_weight='balanced_subsample', n_jobs=-1, random_state=RANDOM_STATE,
        ),
        'xgboost': XGBClassifier(
            n_estimators=450, max_depth=5, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85,
            objective='multi:softprob', num_class=n_classes,
            eval_metric='mlogloss', tree_method='hist',
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }


def run_training():
    REPORTS.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    source = load_gold_data()
    X, y = prepare_dataset(source)
    counts = y.value_counts()
    if len(counts) < 2:
        raise ValueError(f'Classificação requer pelo menos duas classes. Encontrado: {counts.to_dict()}')
    if counts.min() < 5:
        raise ValueError(f'Poucos exemplos na classe minoritária para split estratificado: {counts.to_dict()}')

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=RANDOM_STATE, stratify=y_encoded
    )
    labels = list(range(len(encoder.classes_)))
    fingerprint = dataframe_fingerprint(pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1))
    model_results = []

    for model_name, estimator in make_models(len(encoder.classes_)).items():
        preprocessor = ColumnTransformer([
            ('numeric', Pipeline([
                ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
                ('scaler', StandardScaler()),
            ]), list(X.columns)),
        ], remainder='drop')
        pipeline = Pipeline([('preprocessing', preprocessor), ('model', estimator)])
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

        with mlflow.start_run(run_name=model_name) as run:
            params = estimator.get_params(deep=True)
            mlflow.log_params({f'model__{k}': v for k, v in params.items() if isinstance(v, (str, int, float, bool, type(None)))})
            mlflow.log_params({
                'model_name': model_name,
                'random_state': RANDOM_STATE,
                'test_size': 0.20,
                'feature_columns': ','.join(X.columns),
                'target': 'koi_disposition',
                'dataset_sha256': fingerprint,
                'git_revision': get_git_revision(),
                'train_rows': len(X_train),
                'test_rows': len(X_test),
                'class_names': ','.join(encoder.classes_),
            })

            if model_name == 'xgboost':
                pipeline.fit(X_train, y_train, model__sample_weight=sample_weights)
            else:
                pipeline.fit(X_train, y_train)

            predictions = pipeline.predict(X_test)
            probabilities = pipeline.predict_proba(X_test)
            metrics = {
                'accuracy': accuracy_score(y_test, predictions),
                'balanced_accuracy': balanced_accuracy_score(y_test, predictions),
                'precision_macro': precision_score(y_test, predictions, average='macro', zero_division=0),
                'recall_macro': recall_score(y_test, predictions, average='macro', zero_division=0),
                'f1_macro': f1_score(y_test, predictions, average='macro', zero_division=0),
                'f1_weighted': f1_score(y_test, predictions, average='weighted', zero_division=0),
            }
            mlflow.log_metrics(metrics)

            report = classification_report(y_test, predictions, labels=labels,
                target_names=encoder.classes_, output_dict=True, zero_division=0)
            report_path = REPORTS / f'{model_name}_classification_report.json'
            report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
            mlflow.log_artifact(str(report_path), artifact_path='evaluation')

            fig, ax = plt.subplots(figsize=(7, 6))
            ConfusionMatrixDisplay.from_predictions(
                y_test, predictions, labels=labels, display_labels=encoder.classes_,
                xticks_rotation=35, cmap='Blues', ax=ax, colorbar=False,
            )
            ax.set_title(f'Matriz de confusão — {model_name}')
            fig.tight_layout()
            matrix_path = REPORTS / f'{model_name}_confusion_matrix.png'
            fig.savefig(matrix_path, dpi=140)
            plt.close(fig)
            mlflow.log_artifact(str(matrix_path), artifact_path='evaluation')

            signature = infer_signature(X_test, pipeline.predict(X_test))
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                signature=signature,
                skops_trusted_types=['numpy.dtype','sklearn.tree._tree.Tree','xgboost.core.Booster', 'xgboost.sklearn.XGBClassifier'])
            mlflow.log_dict({'classes': encoder.classes_.tolist(), 'features': X.columns.tolist()}, 'model_metadata.json')

            result = {'model': model_name, 'run_id': run.info.run_id, **metrics}
            model_results.append(result)
            logger.info('%s -> %s', model_name, metrics)

    comparison = pd.DataFrame(model_results).sort_values('f1_macro', ascending=False)
    comparison.to_csv(REPORTS / 'model_comparison.csv', index=False)
    best = comparison.iloc[0].to_dict()
    (REPORTS / 'best_model.json').write_text(json.dumps(best, indent=2), encoding='utf-8')
    logger.info('Melhor modelo por F1 macro: %s', best)
    return comparison


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    print(run_training().to_string(index=False))
