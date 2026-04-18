"""Team-related database models."""

from sqlalchemy import ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from algobet.infrastructure.models import Base


class Tournament(Base):
    """Football tournament/league (e.g., Premier League, La Liga)."""

    __tablename__ = "tournaments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    country: Mapped[str] = mapped_column(String(100), nullable=False)
    url_slug: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)

    # Relationships
    seasons: Mapped[list["Season"]] = relationship(
        back_populates="tournament", cascade="all, delete-orphan"
    )
    matches: Mapped[list["Match"]] = relationship(
        back_populates="tournament", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<Tournament(id={self.id}, name='{self.name}', country='{self.country}')>"
        )


class Season(Base):
    """A season within a tournament (e.g., 2023/2024)."""

    __tablename__ = "seasons"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tournament_id: Mapped[int] = mapped_column(
        ForeignKey("tournaments.id"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(20), nullable=False)  # e.g., "2023/2024"
    start_year: Mapped[int] = mapped_column(Integer, nullable=False)
    end_year: Mapped[int] = mapped_column(Integer, nullable=False)
    url_suffix: Mapped[str | None] = mapped_column(
        String(50), nullable=True
    )  # e.g., "2023-2024" or None for current

    # Relationships
    tournament: Mapped["Tournament"] = relationship(back_populates="seasons")
    matches: Mapped[list["Match"]] = relationship(
        back_populates="season", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("tournament_id", "name", name="uq_tournament_season"),
    )

    def __repr__(self) -> str:
        return f"<Season(id={self.id}, name='{self.name}')>"


class Team(Base):
    """A football team."""

    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)

    # Relationships
    home_matches: Mapped[list["Match"]] = relationship(
        back_populates="home_team",
        foreign_keys="Match.home_team_id",
        cascade="all, delete-orphan",
    )
    away_matches: Mapped[list["Match"]] = relationship(
        back_populates="away_team",
        foreign_keys="Match.away_team_id",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Team(id={self.id}, name='{self.name}')>"


# Forward references for type hints
from algobet.matches.models import Match

Tournament.matches = relationship("Match", back_populates="tournament")
Tournament.seasons = relationship("Season", back_populates="tournament")
Season.matches = relationship("Match", back_populates="season")
Team.home_matches = relationship(
    "Match",
    back_populates="home_team",
    foreign_keys="Match.home_team_id",
)
Team.away_matches = relationship(
    "Match",
    back_populates="away_team",
    foreign_keys="Match.away_team_id",
)
