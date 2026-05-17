"""Tournament resolution helpers for data importers."""

from __future__ import annotations

import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from algobet.models import Tournament


def get_or_create_tournament_by_country(
    session: Session,
    *,
    name: str,
    country: str,
    url_slug: str,
) -> Tournament:
    """Resolve tournaments by country while respecting the legacy slug index.

    ``Tournament.url_slug`` is still globally unique in the database. Some
    sources expose same-name leagues with the same slug across countries
    (German Bundesliga and Austrian Bundesliga). When a bare slug is already
    owned by a different country, create a country-qualified slug instead of
    attaching imported matches to the wrong tournament.
    """
    # Prefer the exact source slug when it already belongs to this country.
    # This keeps a repaired legacy row such as ``bundesliga`` -> Germany
    # authoritative even if an earlier collision-safe row also exists.
    existing_by_slug_country = session.execute(
        select(Tournament).where(
            Tournament.url_slug == url_slug,
            Tournament.country == country,
        )
    ).scalar_one_or_none()
    if existing_by_slug_country:
        return existing_by_slug_country

    existing_by_identity = session.execute(
        select(Tournament)
        .where(
            Tournament.name == name,
            Tournament.country == country,
        )
        .order_by(Tournament.id)
    ).scalar()
    if existing_by_identity:
        return existing_by_identity

    tournament = Tournament(
        name=name,
        country=country,
        url_slug=_available_slug(session, url_slug, country),
    )
    session.add(tournament)
    session.flush()
    return tournament


def _available_slug(session: Session, desired_slug: str, country: str) -> str:
    existing = session.execute(
        select(Tournament).where(Tournament.url_slug == desired_slug)
    ).scalar_one_or_none()
    if existing is None or existing.country == country:
        return desired_slug

    base = f"{_slugify(country)}-{desired_slug}"
    candidate = base
    suffix = 2
    while session.execute(
        select(Tournament).where(Tournament.url_slug == candidate)
    ).scalar_one_or_none():
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    return slug.strip("-") or "league"
