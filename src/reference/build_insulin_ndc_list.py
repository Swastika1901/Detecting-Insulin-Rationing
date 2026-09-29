"""
build_insulin_ndc_list.py
=========================
Fetches NDC codes for common insulin products from the **openFDA Drug NDC**
endpoint and writes them to ``data/reference/insulin_ndc_list.csv``.

No NDC codes are hard-coded; all data originates from the FDA's live API:
    https://open.fda.gov/apis/drug/ndc/

Output columns
--------------
ndc_code   : Full 11-digit package-level NDC  (e.g. ``0088-2220-33``)
drug_name  : Human-readable label built from brand name, generic name,
             strength, dosage form, and package description sourced directly
             from the FDA response.
brand_name : Normalised brand name used in the query.
labeler    : Labeler/manufacturer name from the FDA record.
source_url : openFDA query URL that produced this row (for auditability).

How to refresh this list
------------------------
Re-run this script at any time::

    python src/reference/build_insulin_ndc_list.py

To add or remove products, edit the ``BRAND_SEARCH_TERMS`` list near the top
of this file.  Each entry is passed verbatim as the ``brand_name`` search
value to the openFDA API.

For broader coverage (e.g. generic / biosimilar names), you can also add
entries to ``GENERIC_SEARCH_TERMS``; those are searched against the
``generic_name`` field instead.

Notes
-----
* The openFDA API is rate-limited to ~240 requests/minute without an API key.
  Set the ``OPENFDA_API_KEY`` environment variable to raise this limit.
  Request a free key at https://open.fda.gov/apis/authentication/
* Only *finished* drug products (``finished:true``) that are prescription or
  OTC human drugs are included.  Sample packs are excluded.
* The script fetches up to ``MAX_RESULTS_PER_BRAND`` results per search term
  (default 100). Increase this if a brand has more registered NDC products.
"""

from __future__ import annotations

import csv
import os
import sys
import time
import urllib.parse
import urllib.request
import json
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# Configuration – edit here to add / remove products.  No NDC codes appear
# anywhere in this file; all codes come from the FDA API at runtime.
# ---------------------------------------------------------------------------

# Brand-name search terms (searched against the ``brand_name`` field).
BRAND_SEARCH_TERMS: list[str] = [
    "Lantus",
    "Lantus SoloStar",
    "Humalog",
    "NovoLog",
    "Levemir",
    "Tresiba",
    "Basaglar",
    "Humulin",
    "Fiasp",
    "Toujeo",
]

# Generic-name search terms (searched against the ``generic_name`` field).
# These complement brand searches and capture biosimilars / generics.
GENERIC_SEARCH_TERMS: list[str] = [
    "insulin glargine",
    "insulin lispro",
    "insulin aspart",
    "insulin detemir",
    "insulin degludec",
]

# Maximum records fetched per search term (openFDA hard-caps at 1000).
MAX_RESULTS_PER_BRAND: int = 100

# Seconds to wait between successive API calls to respect rate limits.
REQUEST_DELAY_SECONDS: float = 0.3

# Retry configuration for transient HTTP errors (500, 502, 503, 429).
MAX_RETRIES: int = 4          # total attempts per query (1 original + 3 retries)
RETRY_BACKOFF_BASE: float = 2.0  # seconds; doubles each attempt: 2, 4, 8 …

# Base openFDA endpoint.
OPENFDA_NDC_BASE = "https://api.fda.gov/drug/ndc.json"

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def _build_url(search: str, limit: int, api_key: str | None) -> str:
    """Return a fully-encoded openFDA query URL."""
    params: dict[str, str] = {"search": search, "limit": str(limit)}
    if api_key:
        params["api_key"] = api_key
    return f"{OPENFDA_NDC_BASE}?{urllib.parse.urlencode(params)}"


# HTTP status codes that are safe to retry (transient server / rate-limit errors).
_RETRYABLE_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})


def _get_json(url: str, timeout: int = 30) -> dict:
    """Perform a GET request and return the parsed JSON body."""
    req = urllib.request.Request(url, headers={"User-Agent": "insulin-rationing-research/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _query_openfda(
    field: str,
    value: str,
    limit: int,
    api_key: str | None,
) -> tuple[list[dict], str]:
    """Query the openFDA NDC endpoint with automatic retry on transient errors.

    Retries up to ``MAX_RETRIES`` times on HTTP 429 / 5xx responses and
    network timeouts, using exponential back-off (``RETRY_BACKOFF_BASE ** attempt``
    seconds between attempts).  HTTP 404 (no results) is returned as an empty
    list immediately without retrying.

    Parameters
    ----------
    field:
        The openFDA field to search (e.g. ``"brand_name"`` or
        ``"generic_name"``).
    value:
        The value to search for; will be double-quoted for exact matching.
    limit:
        Maximum number of results to request.
    api_key:
        Optional openFDA API key.

    Returns
    -------
    tuple[list[dict], str]
        A tuple of ``(results_list, query_url)`` where ``results_list`` is the
        list of product records returned by the API (may be empty) and
        ``query_url`` is the URL that was called.

    Raises
    ------
    urllib.error.HTTPError
        Re-raised after all retries are exhausted for non-404 errors.
    """
    search_expr = f'{field}:"{value}"'
    url = _build_url(search_expr, limit, api_key)

    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            data = _get_json(url)
            return data.get("results", []), url

        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                # No results – not an error, don't retry.
                return [], url
            if exc.code in _RETRYABLE_CODES:
                wait = RETRY_BACKOFF_BASE ** attempt
                print(
                    f"[retry {attempt + 1}/{MAX_RETRIES}] HTTP {exc.code} – "
                    f"waiting {wait:.0f}s before retrying …",
                    flush=True,
                )
                time.sleep(wait)
                last_exc = exc
                continue
            raise  # non-retryable HTTP error (e.g. 400 bad request)

        except (urllib.error.URLError, TimeoutError) as exc:
            # Network-level error or timeout – safe to retry.
            wait = RETRY_BACKOFF_BASE ** attempt
            print(
                f"[retry {attempt + 1}/{MAX_RETRIES}] Network error ({exc}) – "
                f"waiting {wait:.0f}s before retrying …",
                flush=True,
            )
            time.sleep(wait)
            last_exc = exc
            continue

    # All retries exhausted.
    raise RuntimeError(
        f"openFDA query failed after {MAX_RETRIES} attempts for "
        f'{field}="{value}". Last error: {last_exc}'
    ) from last_exc


# ---------------------------------------------------------------------------
# Record parsing
# ---------------------------------------------------------------------------


def _format_strength(active_ingredients: list[dict]) -> str:
    """Return a comma-separated strength string from the active_ingredients list."""
    parts = [
        f"{ing.get('name', '').title()} {ing.get('strength', '')}".strip()
        for ing in active_ingredients
    ]
    return ", ".join(parts)


def _iter_rows_from_result(result: dict, query_url: str) -> Iterator[dict[str, str]]:
    """Yield one CSV row per non-sample package in an openFDA product record.

    Parameters
    ----------
    result:
        A single product record from the openFDA ``results`` array.
    query_url:
        The API URL that produced this record (stored for auditability).

    Yields
    ------
    dict[str, str]
        A row with keys: ``ndc_code``, ``drug_name``, ``brand_name``,
        ``labeler``, ``source_url``.
    """
    brand = result.get("brand_name", "").strip()
    generic = result.get("generic_name", "").strip()
    dosage_form = result.get("dosage_form", "").strip().title()
    labeler = result.get("labeler_name", "").strip()
    strength_str = _format_strength(result.get("active_ingredients", []))

    for pkg in result.get("packaging", []):
        # Skip sample packs – they are not commercially dispensed.
        if pkg.get("sample", False):
            continue

        ndc_code = pkg.get("package_ndc", "").strip()
        pkg_desc = pkg.get("description", "").strip()

        if not ndc_code:
            continue

        # Build a human-readable drug_name from FDA-sourced fields.
        name_parts = [brand or generic]
        if generic and generic.lower() not in brand.lower():
            name_parts.append(f"({generic})")
        if strength_str:
            name_parts.append(strength_str)
        if dosage_form:
            name_parts.append(f"[{dosage_form}]")
        if pkg_desc:
            name_parts.append(f"– {pkg_desc}")

        drug_name = " ".join(name_parts)

        yield {
            "ndc_code": ndc_code,
            "drug_name": drug_name,
            "brand_name": brand,
            "labeler": labeler,
            "source_url": query_url,
        }


# ---------------------------------------------------------------------------
# Main fetch logic
# ---------------------------------------------------------------------------


def fetch_all_rows(
    brand_terms: list[str],
    generic_terms: list[str],
    limit: int,
    api_key: str | None,
    delay: float,
) -> list[dict[str, str]]:
    """Fetch NDC rows from openFDA for all configured search terms.

    Parameters
    ----------
    brand_terms:
        List of brand-name strings to query.
    generic_terms:
        List of generic-name strings to query.
    limit:
        Max results per query.
    api_key:
        Optional openFDA API key.
    delay:
        Seconds to wait between API requests.

    Returns
    -------
    list[dict[str, str]]
        Deduplicated, sorted rows ready for CSV output.
    """
    seen_ndcs: set[str] = set()
    rows: list[dict[str, str]] = []
    failed_queries: list[tuple[str, str, str]] = []  # (field, value, error)

    queries: list[tuple[str, str]] = (
        [("brand_name", t) for t in brand_terms]
        + [("generic_name", t) for t in generic_terms]
    )

    for i, (field, value) in enumerate(queries):
        print(f"  Querying {field}=\"{value}\" …", end=" ", flush=True)
        try:
            results, url = _query_openfda(field, value, limit, api_key)
        except Exception as exc:  # noqa: BLE001
            # Non-fatal: log the failure and continue with remaining queries.
            print(f"SKIPPED (all retries failed: {exc})")
            failed_queries.append((field, value, str(exc)))
            if i < len(queries) - 1:
                time.sleep(delay)
            continue

        print(f"{len(results)} product(s) found.")

        for result in results:
            for row in _iter_rows_from_result(result, url):
                if row["ndc_code"] not in seen_ndcs:
                    seen_ndcs.add(row["ndc_code"])
                    rows.append(row)

        # Respect the API rate limit between requests (skip after last).
        if i < len(queries) - 1:
            time.sleep(delay)

    # Sort deterministically: brand name then NDC code.
    rows.sort(key=lambda r: (r["brand_name"].lower(), r["ndc_code"]))
    return rows, failed_queries


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

FIELDNAMES = ["ndc_code", "drug_name", "brand_name", "labeler", "source_url"]


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    """Write ``rows`` to a CSV at ``output_path``, creating parent dirs if needed.

    Parameters
    ----------
    rows:
        List of row dicts with keys matching :data:`FIELDNAMES`.
    output_path:
        Destination file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} NDC rows to {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Fetch insulin NDC codes from openFDA and write the reference CSV."""
    api_key: str | None = os.environ.get("OPENFDA_API_KEY") or None
    if api_key:
        print("Using openFDA API key from OPENFDA_API_KEY environment variable.")
    else:
        print(
            "No OPENFDA_API_KEY set – using anonymous access (rate-limited to "
            "~240 req/min). Set the env-var for a higher quota."
        )

    print(
        f"\nFetching from openFDA ({len(BRAND_SEARCH_TERMS)} brand + "
        f"{len(GENERIC_SEARCH_TERMS)} generic search terms) …\n"
    )

    rows, failed_queries = fetch_all_rows(
        brand_terms=BRAND_SEARCH_TERMS,
        generic_terms=GENERIC_SEARCH_TERMS,
        limit=MAX_RESULTS_PER_BRAND,
        api_key=api_key,
        delay=REQUEST_DELAY_SECONDS,
    )

    if not rows:
        print("ERROR: No rows fetched. Check network access and API status.", file=sys.stderr)
        sys.exit(1)

    project_root = Path(__file__).resolve().parent.parent.parent
    output_path = project_root / "data" / "reference" / "insulin_ndc_list.csv"
    write_csv(rows, output_path)

    if failed_queries:
        print(
            f"\nWARNING: {len(failed_queries)} search term(s) could not be fetched "
            "(FDA API error) and were skipped:",
            file=sys.stderr,
        )
        for field, value, err in failed_queries:
            print(f"  {field}=\"{value}\" → {err}", file=sys.stderr)
        print(
            "Re-run the script later to pick up the missing terms, "
            "or add them to GENERIC_SEARCH_TERMS as a fallback.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
