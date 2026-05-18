"""Generate an enriched household energy dataset with synthetic housing features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATASET_FILE = DATA_DIR / "household_energy_consumption.csv"
ENRICHED_DATASET_FILE = DATA_DIR / "household_energy_consumption_enriched.csv"
RANDOM_STATE = 42

HEATING_TYPES = ("Electric", "Gas", "Heat Pump", "District Heating")


def _choose_heating_type(
    rng: np.random.Generator,
    household_size: int,
    surface_m2: float,
    has_ac: bool,
) -> str:
    """Pick a plausible heating type from household profile attributes."""

    heat_pump_score = 0.08 + (0.18 if has_ac else 0.0) + max(surface_m2 - 70, 0) * 0.002
    gas_score = 0.22 + household_size * 0.035 + max(surface_m2 - 55, 0) * 0.0015
    district_score = 0.20 if household_size <= 3 else 0.11
    electric_score = 0.34 if surface_m2 <= 75 else 0.22

    weights = np.array(
        [electric_score, gas_score, heat_pump_score, district_score],
        dtype=float,
    )
    weights = weights / weights.sum()
    return str(rng.choice(HEATING_TYPES, p=weights))


def build_enriched_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Return the original observations with generated housing and presence features."""

    required_columns = {
        "Household_ID",
        "Date",
        "Household_Size",
        "Avg_Temperature_C",
        "Has_AC",
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    rng = np.random.default_rng(RANDOM_STATE)
    enriched = df.copy()
    enriched["Date"] = pd.to_datetime(enriched["Date"], errors="coerce")
    enriched["Has_AC"] = enriched["Has_AC"].astype(str).str.strip().str.title()

    household_profile = (
        enriched.sort_values(["Household_ID", "Date"])
        .groupby("Household_ID", as_index=False)
        .agg(
            Household_Size=("Household_Size", "first"),
            Has_AC=("Has_AC", "first"),
            Avg_Temperature_C=("Avg_Temperature_C", "mean"),
        )
    )

    n_households = len(household_profile)
    has_ac_binary = household_profile["Has_AC"].eq("Yes").astype(int).to_numpy()
    household_size = household_profile["Household_Size"].to_numpy()

    surface_noise = rng.normal(0, 8.5, size=n_households)
    surface_m2 = (
        30
        + household_size * 17.5
        + has_ac_binary * 8.0
        + surface_noise
        + rng.gamma(shape=2.0, scale=2.8, size=n_households)
    )
    household_profile["surface_m2"] = np.clip(surface_m2, 24, 180).round(1)

    household_profile["heating_type"] = [
        _choose_heating_type(
            rng=rng,
            household_size=int(row.Household_Size),
            surface_m2=float(row.surface_m2),
            has_ac=row.Has_AC == "Yes",
        )
        for row in household_profile.itertuples(index=False)
    ]

    household_profile["home_presence_base"] = (
        7.4
        + household_profile["Household_Size"] * 0.48
        + rng.normal(0, 1.0, size=n_households)
    )

    enriched = enriched.merge(
        household_profile[
            ["Household_ID", "surface_m2", "heating_type", "home_presence_base"]
        ],
        on="Household_ID",
        how="left",
        validate="many_to_one",
    )

    weekday = enriched["Date"].dt.dayofweek
    weekend_bonus = np.where(weekday >= 5, 2.7, 0.0)
    cold_or_hot_bonus = np.where(
        (enriched["Avg_Temperature_C"] <= 14.0) | (enriched["Avg_Temperature_C"] >= 21.0),
        0.7,
        0.0,
    )
    ac_hot_bonus = np.where(
        enriched["Has_AC"].eq("Yes") & (enriched["Avg_Temperature_C"] >= 20.0),
        0.4,
        0.0,
    )
    daily_noise = rng.normal(0, 0.75, size=len(enriched))

    enriched["hours_at_home"] = np.clip(
        enriched["home_presence_base"]
        + weekend_bonus
        + cold_or_hot_bonus
        + ac_hot_bonus
        + daily_noise,
        4.0,
        22.0,
    ).round(1)

    enriched = enriched.drop(columns=["home_presence_base"])
    enriched["Date"] = enriched["Date"].dt.strftime("%Y-%m-%d")

    return enriched


def main() -> None:
    """Generate and save the enriched household energy dataset."""

    df = pd.read_csv(RAW_DATASET_FILE)
    enriched = build_enriched_dataset(df)
    enriched.to_csv(ENRICHED_DATASET_FILE, index=False)
    print(f"Saved {len(enriched):,} rows to {ENRICHED_DATASET_FILE}")


if __name__ == "__main__":
    main()
