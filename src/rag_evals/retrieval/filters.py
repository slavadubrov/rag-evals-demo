"""Helpers for filter predicates with audit logging.

The article's centrepiece metric depends on knowing which predicates were
applied. ``log_predicates`` emits a structured trace event so the eval
harness can replay them — this is what makes filter false-exclusion
detectable in production.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def log_predicates(qid: str, predicates: dict[str, Any]) -> None:
    logger.info("filter_predicates", extra={"qid": qid, "predicates": predicates})


def predicates_to_str(predicates: dict[str, Any]) -> str:
    if not predicates:
        return "(none)"
    return json.dumps(predicates, sort_keys=True)
