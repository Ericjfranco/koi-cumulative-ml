import pandas as pd
from ml.features import prepare_dataset, LEAKAGE_COLUMNS


def test_prepare_dataset_excludes_target_leakage_and_keeps_classes():
    frame = pd.DataFrame({
        'koi_disposition': ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE', 'CONFIRMED'],
        'koi_period': ['10.1', '20.2', '30.3', '40.4'],
        'koi_prad': ['1.1', '2.2', '3.3', '4.4'],
        'koi_teq': ['500', '600', '700', '800'],
        'koi_pdisposition': ['FALSE POSITIVE'] * 4,
        'koi_score': ['1'] * 4,
    })
    X, y = prepare_dataset(frame)
    assert set(y) == {'CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'}
    assert not (set(X.columns) & LEAKAGE_COLUMNS)
    assert X.shape == (4, 3)
