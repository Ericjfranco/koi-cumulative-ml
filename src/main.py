import argparse
import logging
import time
from datetime import datetime, timezone

from src.config import LOG_DIR, RAW_DATA_DIR
from src.extract import download_koi_csv
from src.load import load_to_postgres
from src.transform import transform_csv


def configure_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "koi_etl.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,
    )


def run_pipeline():
    logger = logging.getLogger(__name__)
    started_at = datetime.now(timezone.utc)
    logger.info("========== INÍCIO DO PIPELINE KOI ==========")

    try:
        csv_path = download_koi_csv(RAW_DATA_DIR)
        dataframe = transform_csv(csv_path)
        load_to_postgres(dataframe)

        elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
        logger.info(
            "Pipeline finalizado. Linhas: %s | Colunas: %s | Duração: %.1fs",
            len(dataframe),
            len(dataframe.columns),
            elapsed,
        )
    except Exception:
        logger.exception("Pipeline encerrado com erro.")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Download e carga da tabela KOI Cumulative."
    )
    parser.add_argument(
        "--interval-hours",
        type=float,
        default=0,
        help="Intervalo em horas entre execuções. Zero executa apenas uma vez.",
    )
    args = parser.parse_args()

    if args.interval_hours < 0:
        parser.error("--interval-hours não pode ser negativo.")

    configure_logging()

    if args.interval_hours == 0:
        run_pipeline()
        return

    logging.getLogger(__name__).info(
        "Modo recorrente ativado: intervalo de %s horas.",
        args.interval_hours,
    )
    while True:
        try:
            run_pipeline()
        except Exception:
            logging.getLogger(__name__).exception(
                "A execução falhou; uma nova tentativa ocorrerá no próximo ciclo."
            )
        time.sleep(args.interval_hours * 3600)


if __name__ == "__main__":
    main()
