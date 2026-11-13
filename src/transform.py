import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)


def transform_csv(csv_path):
    logger.info("Lendo e validando o arquivo CSV: %s", csv_path)

    df = pd.read_csv(
        csv_path,
        dtype="string",
        keep_default_na=False,
        na_filter=False,
    )

    if df.empty:
        raise ValueError("O CSV não contém registros.")

    if len(df.columns) == 0:
        raise ValueError("O CSV não contém colunas.")

    # Padroniza nomes.
    normalized = []
    for column in df.columns:
        name = re.sub(r"[^a-zA-Z0-9_]", "_", str(column).strip().lower())
        name = re.sub(r"_+", "_", name).strip("_")
        if not name:
            raise ValueError(f"Nome de coluna inválido: {column!r}")
        if name[0].isdigit():
            name = f"col_{name}"
        normalized.append(name)

    if len(set(normalized)) != len(normalized):
        raise ValueError("Existem nomes de colunas duplicados após padronização.")

    df.columns = normalized
    logger.info(
        "Transformação concluída: %s registros e %s colunas.",
        len(df),
        len(df.columns),
    )
    return df
