import logging
from datetime import datetime, timezone
from pathlib import Path

import requests

from src.config import TAP_QUERY, TAP_URL, REQUEST_TIMEOUT

logger = logging.getLogger(__name__)


def download_koi_csv(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "query": TAP_QUERY,
        "format": "csv",
    }

    logger.info("Iniciando download da tabela KOI Cumulative via TAP.")
    logger.info("Consulta ADQL: %s", TAP_QUERY)

    response = requests.get(
        TAP_URL,
        params=params,
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": "koi-cumulative-etl/1.0"},
    )
    response.raise_for_status()

    content = response.content
    if not content or len(content) < 20:
        raise ValueError("A resposta do serviço TAP está vazia ou incompleta.")

    # Verificação para evitar salvar mensagens de erro.
    content_start = content.lstrip()[:100].lower()
    if content_start.startswith(b"<!doctype html") or content_start.startswith(b"<html"):
        raise ValueError("O serviço TAP retornou HTML em vez de CSV.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"koi_cumulative_{timestamp}.csv"
    output_path.write_bytes(content)

    logger.info(
        "CSV salvo em %s (%.2f KB).",
        output_path,
        len(content) / 1024,
    )
    return output_path
