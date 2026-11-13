import logging
import uuid

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.types import Text

from src.config import DATABASE_URL, TABLE_NAME

logger = logging.getLogger(__name__)


def load_to_postgres(df):
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    staging_table = f"_stg_koi_{uuid.uuid4().hex[:12]}"

    try:
        dtype = {column: Text() for column in df.columns}

        logger.info("Carregando %s registros na tabela de staging.", len(df))
        df.to_sql(
            staging_table,
            engine,
            schema="public",
            if_exists="fail",
            index=False,
            dtype=dtype,
            chunksize=2000,
            method="multi",
        )

        inspector = inspect(engine)
        staging_columns = [
            col["name"] for col in inspector.get_columns(
                staging_table, schema="public"
            )
        ]

        quoted_columns = ", ".join(
            '"' + col.replace('"', '""') + '"' for col in staging_columns
        )

        with engine.begin() as connection:
            target_exists = inspector.has_table(TABLE_NAME, schema="public")

            if not target_exists:
                connection.execute(text(
                    f'CREATE TABLE public."{TABLE_NAME}" '
                    f'(LIKE public."{staging_table}" INCLUDING ALL)'
                ))
            else:
                # Adiciona colunas novas.
                existing_columns = {
                    col["name"] for col in inspector.get_columns(
                        TABLE_NAME, schema="public"
                    )
                }
                for column in staging_columns:
                    if column not in existing_columns:
                        safe_column = column.replace('"', '""')
                        connection.execute(text(
                            f'ALTER TABLE public."{TABLE_NAME}" '
                            f'ADD COLUMN "{safe_column}" TEXT'
                        ))

            # Snapshot substituído.
            connection.execute(text(f'TRUNCATE TABLE public."{TABLE_NAME}"'))
            connection.execute(text(
                f'INSERT INTO public."{TABLE_NAME}" ({quoted_columns}) '
                f'SELECT {quoted_columns} FROM public."{staging_table}"'
            ))
            connection.execute(text(
                f'DROP TABLE public."{staging_table}"'
            ))

        logger.info(
            "Carga concluída com sucesso: %s registros em public.%s.",
            len(df),
            TABLE_NAME,
        )
    except Exception:
        # Remove staging caso ocorra falha.
        with engine.begin() as connection:
            connection.execute(text(
                f'DROP TABLE IF EXISTS public."{staging_table}"'
            ))
        logger.exception("Falha durante a carga no PostgreSQL.")
        raise
    finally:
        engine.dispose()
