"""Optional Postgres/Neon persistence for run history.

The filesystem stays the source of truth — this layer *mirrors* run
summaries, metrics, and report cards into a Postgres database so history
survives ephemeral machines (a Colab VM, a rented GPU box) and multiple
machines can share one dashboard.

The app owns its schema: connecting migrates automatically. Users never
run SQL by hand.
"""

from distillanything.db.store import (  # noqa: F401
    MemoryStore,
    PgStore,
    clear_db_url,
    redact_dsn,
    resolve_db_url,
    save_db_url,
)
from distillanything.db.sync import local_host_label, sync_all, sync_run  # noqa: F401
