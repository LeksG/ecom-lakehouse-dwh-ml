"""Етап E (Extract): інгестія сирих файлів у шар Bronze озера даних (Delta Lake).

Дані зберігаються «як є» (усі поля - рядки) з технічними метаданими:
_ingest_ts - час завантаження, _source_file - назва файла-джерела, _batch_id - ідентифікатор партії.
"""
import uuid
from datetime import datetime, timezone

import pandas as pd
import pyarrow as pa
from deltalake import write_deltalake

from config import RAW_DIR, BRONZE, SOURCES


def ingest_to_bronze(batch_id: str | None = None) -> dict:
    batch_id = batch_id or uuid.uuid4().hex[:12]
    ts = datetime.now(timezone.utc).isoformat()
    stats = {}
    for name, fname in SOURCES.items():
        df = pd.read_csv(RAW_DIR / fname, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        df["_ingest_ts"] = ts
        df["_source_file"] = fname
        df["_batch_id"] = batch_id
        table = pa.Table.from_pandas(df, preserve_index=False)
        write_deltalake(str(BRONZE / name), table, mode="overwrite", schema_mode="overwrite")
        stats[name] = len(df)
    return {"batch_id": batch_id, "rows": stats}


if __name__ == "__main__":
    print(ingest_to_bronze())
