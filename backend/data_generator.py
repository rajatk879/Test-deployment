"""
MC4 Synthetic Data Generator v4 — 5-Layer Star Schema (Validated)
=================================================================
Generates a comprehensive flour-milling planning dataset following the
updated dimensional model with REALISTIC values validated against:

  • MC4 (Fourth Milling Company) actual capacities:
    – Dammam: 1,350 TPD wheat grinding, 80K ton silos
    – Medina: 1,200 TPD wheat grinding, 60K ton silos
    – Al-Kharj: 600 TPD wheat grinding, 10K ton silos
    – Total: 3,150 TPD wheat → ~2,400 TPD flour output

  • Real wheat import prices (USD/ton × 3.75 SAR/USD + freight)
  • Real import source countries (AU, CA, US, AR, DE, RO)
  • Industry-standard extraction rates, energy, water, waste
  • Saudi Arabia weekend change (Sat-Sun from Jan 2022)

  Layer 1  – Master / Dimension tables   (dim_*)
  Layer 2  – Mapping tables              (map_*)
  Layer 3  – Fact / Transactional tables  (fact_*)
  Layer 4  – KPI Snapshot                 (fact_kpi_snapshot)
  Layer 5  – KPI Catalog                  (consumed by UI via API)

Core truth: One mill can run only ONE recipe at a time.
"""

import numpy as np
import pandas as pd
from datetime import timedelta, datetime
import os

# Reproducibility
np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL CONSTANTS — Validated against real-world data
# ═══════════════════════════════════════════════════════════════════════════════

# Wheat base price (SAR/ton) — long-term average CIF Saudi ports
# Global wheat avg ~$270/ton × 3.75 SAR/USD + $30 freight = ~$300 × 3.75 ≈ 1,125
WHEAT_BASE_PRICE_SAR = 1050

# Yearly price multipliers reflecting real global wheat market
# 2020: COVID disruptions, 2022: Ukraine war spike, 2023-24: normalization
YEARLY_WHEAT_PRICE_FACTOR = {
    2020: 0.90,   # ~945 SAR/ton — COVID, lower global prices (~$215/ton FOB)
    2021: 1.10,   # ~1,155 SAR/ton — Rising demand, supply chain issues (~$270/ton)
    2022: 1.40,   # ~1,470 SAR/ton — Ukraine war spike (~$350/ton FOB)
    2023: 1.07,   # ~1,124 SAR/ton — Prices cooling (~$265/ton)
    2024: 0.96,   # ~1,008 SAR/ton — Normalized (~$230/ton)
    2025: 1.00,   # ~1,050 SAR/ton — Stable baseline (~$245/ton)
    2026: 1.02,   # ~1,071 SAR/ton — Slight uptick (~$250/ton)
}

# Waste target for Vision 2030 sustainability goals
WASTE_TARGET_PCT = 2.5

# Saudi Arabia imports ~100% of wheat since phasing out domestic farming (2016)
# Import dependency should be ~100%

# ═══════════════════════════════════════════════════════════════════════════════
# TIME DIMENSION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_time_dimension():
    start_date = "2020-01-01"
    end_date = "2026-02-10"
    dates = pd.date_range(start_date, end_date, freq="D")

    df = pd.DataFrame({"date": dates})
    df["week"] = (
        df["date"].dt.isocalendar().year.astype(str)
        + "-W"
        + df["date"].dt.isocalendar().week.astype(str).str.zfill(2)
    )
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter
    df["day_of_week"] = df["date"].dt.dayofweek

    # Saudi Arabia weekend:
    #   Before Jan 2022: Friday-Saturday (dayofweek 4, 5)
    #   From Jan 2022:   Saturday-Sunday (dayofweek 5, 6) — aligned with global markets
    df["is_weekend"] = np.where(
        df["date"] < pd.Timestamp("2022-01-01"),
        df["day_of_week"].isin([4, 5]),   # Fri, Sat
        df["day_of_week"].isin([5, 6]),   # Sat, Sun
    )

    # Ramadan dates (approximate Gregorian starts)
    ramadan_starts = {
        2020: (4, 24), 2021: (4, 13), 2022: (4, 2), 2023: (3, 23),
        2024: (3, 11), 2025: (3, 1), 2026: (2, 18), 2027: (2, 7),
    }

    def _ramadan(date):
        if date.year in ramadan_starts:
            m, d = ramadan_starts[date.year]
            s = datetime(date.year, m, d)
            return s <= date <= s + timedelta(days=29)
        return False

    hajj_months = {
        2020: 7, 2021: 7, 2022: 7, 2023: 6,
        2024: 6, 2025: 6, 2026: 5, 2027: 5,
    }

    def _hajj(date):
        return date.year in hajj_months and date.month == hajj_months[date.year]

    df["is_ramadan"] = df["date"].apply(_ramadan)
    df["is_hajj"] = df["date"].apply(_hajj)
    df["is_eid"] = df["is_ramadan"] | df["is_hajj"]
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — MASTER / DIMENSION TABLES
# ═══════════════════════════════════════════════════════════════════════════════

def generate_dim_mill():
    """
    MC4 mills from https://www.mc4.com.sa/:
      Dammam  — 1,350 TPD wheat, 80K ton silo, 450 TPD feed capacity
      Medina  — 1,200 TPD wheat, 60K ton silo
      Al-Kharj — 600 TPD wheat, 10K ton silo (upgraded 2016)
    Operating hours: Modern mills run 22h/day with 2h maintenance/cleaning.
    Al-Kharj (smaller/older): 20h/day.
    """
    return pd.DataFrame([
        {"mill_id": "M1", "mill_name": "Dammam Mill", "location": "Eastern Region",
         "wheat_capacity_tpd": 1350,
         "max_hours_per_day": 22, "planned_days_per_month": 28,
         "max_monthly_hours": 616, "is_active": True,
         "energy_efficiency_index": 85, "water_efficiency_index": 82,
         "silo_capacity_tons": 80000, "feed_capacity_tpd": 450},
        {"mill_id": "M2", "mill_name": "Medina Mill", "location": "Medina",
         "wheat_capacity_tpd": 1200,
         "max_hours_per_day": 22, "planned_days_per_month": 28,
         "max_monthly_hours": 616, "is_active": True,
         "energy_efficiency_index": 80, "water_efficiency_index": 76,
         "silo_capacity_tons": 60000, "feed_capacity_tpd": 0},
        {"mill_id": "M3", "mill_name": "Al-Kharj Mill", "location": "Central Region",
         "wheat_capacity_tpd": 600,
         "max_hours_per_day": 20, "planned_days_per_month": 26,
         "max_monthly_hours": 520, "is_active": True,
         "energy_efficiency_index": 72, "water_efficiency_index": 70,
         "silo_capacity_tons": 10000, "feed_capacity_tpd": 0},
    ])


def generate_dim_recipe():
    """
    Milling recipes with realistic throughput rates (FLOUR output TPH).
    Rates are fleet-average; mill-specific rates in map_recipe_mill.

    Extraction rates (industry standard):
      Patent (fine white):  70-76%
      Standard/Bakery:      76-80%
      Brown/Whole wheat:    85-92%

    Waste = actual material loss (dust, cleaning, spoilage).
    The remainder (100% - extraction - waste) goes to saleable
    byproducts: bran, germ, animal feed — NOT counted as waste.

    Changeover times: Real mill recipe switches take 1-4 hours
    (roll adjustment, cleaning, quality verification).
    """
    return pd.DataFrame([
        {"recipe_id": "R1", "recipe_name": "Patent Blend",
         "description": "High-quality patent flour blend",
         "milling_rate_tph": 38, "avg_yield": 76.0, "avg_waste_pct": 2.0,
         "changeover_time_hrs": 1.5, "allowed_split_day": True, "is_active": True},
        {"recipe_id": "R2", "recipe_name": "Bakery Standard",
         "description": "Fast throughput, standard bakery quality",
         "milling_rate_tph": 43, "avg_yield": 78.0, "avg_waste_pct": 1.8,
         "changeover_time_hrs": 1.0, "allowed_split_day": True, "is_active": True},
        {"recipe_id": "R3", "recipe_name": "Brown Flour",
         "description": "Whole grain, higher extraction rate",
         "milling_rate_tph": 35, "avg_yield": 88.0, "avg_waste_pct": 2.8,
         "changeover_time_hrs": 2.5, "allowed_split_day": True, "is_active": True},
        {"recipe_id": "R4", "recipe_name": "Standard Blend",
         "description": "Balanced bakery-grade blend",
         "milling_rate_tph": 40, "avg_yield": 75.0, "avg_waste_pct": 2.2,
         "changeover_time_hrs": 1.5, "allowed_split_day": True, "is_active": True},
        {"recipe_id": "R5", "recipe_name": "Premium Patent",
         "description": "Premium low-extraction patent flour",
         "milling_rate_tph": 35, "avg_yield": 72.0, "avg_waste_pct": 1.5,
         "changeover_time_hrs": 2.0, "allowed_split_day": True, "is_active": True},
    ])


def generate_dim_flour_type():
    """
    Flour types aligned with MC4 actual products (https://www.mc4.com.sa/):
      - Fortified Patent Flour  (premium white, retail)
      - Fortified Bakery Flour  (standard bakery grade)
      - Fortified Brown Flour   (whole wheat)
      - Superior Brown Flour    (premium whole wheat)
      - Premium Artisan Flour   (specialty high-gluten, B2B/HoReCa)
    """
    return pd.DataFrame([
        {"flour_type_id": "FT001", "flour_name": "Fortified Patent",
         "protein_min": 12.0, "protein_max": 14.0, "ash_max": 0.45,
         "quality_spec": "Premium white flour for retail — MC4 flagship", "is_active": True},
        {"flour_type_id": "FT002", "flour_name": "Fortified Bakery",
         "protein_min": 10.0, "protein_max": 12.0, "ash_max": 0.65,
         "quality_spec": "Standard bakery grade for commercial bakers", "is_active": True},
        {"flour_type_id": "FT003", "flour_name": "Premium Artisan",
         "protein_min": 13.0, "protein_max": 15.0, "ash_max": 0.50,
         "quality_spec": "High-gluten specialty flour for artisan bakers", "is_active": True},
        {"flour_type_id": "FT004", "flour_name": "Fortified Brown",
         "protein_min": 10.0, "protein_max": 12.0, "ash_max": 1.00,
         "quality_spec": "Whole wheat brown flour — MC4 brown line", "is_active": True},
        {"flour_type_id": "FT005", "flour_name": "Superior Brown",
         "protein_min": 11.0, "protein_max": 13.0, "ash_max": 0.85,
         "quality_spec": "Premium brown flour — MC4 superior brown line", "is_active": True},
    ])


def generate_dim_sku():
    """
    14 SKUs across MC4's two brands (FOOM retail, Miller professional).
    Pack sizes match Saudi market norms: 45kg wholesale, 10kg retail, 1kg retail.

    Base daily demand targets ~2,000 tons/day flour (stored in forecast generator).
    MC4 total flour output capacity: ~2,400 tons/day → ~83% utilization baseline.
    """
    return pd.DataFrame([
        # FOOM brand — Fortified Patent (FT001)
        {"sku_id": "SKU001", "sku_name": "FOOM Patent Flour 45kg",
         "flour_type_id": "FT001", "pack_size_kg": 45,
         "region": "Central", "channel": "wholesale", "is_active": True},
        {"sku_id": "SKU002", "sku_name": "FOOM Patent Flour 10kg",
         "flour_type_id": "FT001", "pack_size_kg": 10,
         "region": "Central", "channel": "retail", "is_active": True},
        {"sku_id": "SKU003", "sku_name": "FOOM Patent Flour 1kg",
         "flour_type_id": "FT001", "pack_size_kg": 1,
         "region": "Central", "channel": "retail", "is_active": True},
        # FOOM brand — Fortified Bakery (FT002)
        {"sku_id": "SKU004", "sku_name": "FOOM Bakery Flour 45kg",
         "flour_type_id": "FT002", "pack_size_kg": 45,
         "region": "Eastern", "channel": "wholesale", "is_active": True},
        {"sku_id": "SKU005", "sku_name": "FOOM Bakery Flour 10kg",
         "flour_type_id": "FT002", "pack_size_kg": 10,
         "region": "Eastern", "channel": "retail", "is_active": True},
        # FOOM brand — Premium Artisan (FT003)
        {"sku_id": "SKU006", "sku_name": "FOOM Artisan Flour 45kg",
         "flour_type_id": "FT003", "pack_size_kg": 45,
         "region": "Western", "channel": "wholesale", "is_active": True},
        {"sku_id": "SKU007", "sku_name": "FOOM Artisan Flour 10kg",
         "flour_type_id": "FT003", "pack_size_kg": 10,
         "region": "Western", "channel": "retail", "is_active": True},
        # FOOM brand — Fortified Brown (FT004)
        {"sku_id": "SKU008", "sku_name": "FOOM Brown Flour 45kg",
         "flour_type_id": "FT004", "pack_size_kg": 45,
         "region": "Central", "channel": "wholesale", "is_active": True},
        {"sku_id": "SKU009", "sku_name": "FOOM Brown Flour 10kg",
         "flour_type_id": "FT004", "pack_size_kg": 10,
         "region": "Central", "channel": "retail", "is_active": True},
        # FOOM brand — Superior Brown (FT005)
        {"sku_id": "SKU010", "sku_name": "FOOM Superior Brown 45kg",
         "flour_type_id": "FT005", "pack_size_kg": 45,
         "region": "Southern", "channel": "wholesale", "is_active": True},
        # Miller brand — Fortified Patent (FT001)
        {"sku_id": "SKU011", "sku_name": "Miller Patent Flour 45kg",
         "flour_type_id": "FT001", "pack_size_kg": 45,
         "region": "Eastern", "channel": "wholesale", "is_active": True},
        {"sku_id": "SKU012", "sku_name": "Miller Patent Flour 10kg",
         "flour_type_id": "FT001", "pack_size_kg": 10,
         "region": "Eastern", "channel": "retail", "is_active": True},
        # Miller brand — Fortified Bakery (FT002)
        {"sku_id": "SKU013", "sku_name": "Miller Bakery Flour 45kg",
         "flour_type_id": "FT002", "pack_size_kg": 45,
         "region": "Northern", "channel": "wholesale", "is_active": True},
        {"sku_id": "SKU014", "sku_name": "Miller Bakery Flour 10kg",
         "flour_type_id": "FT002", "pack_size_kg": 10,
         "region": "Northern", "channel": "retail", "is_active": True},
    ])


def generate_dim_wheat_type():
    """
    Wheat classes that Saudi Arabia actually imports.
    Saudi Arabia imports 100% of its wheat since phasing out domestic
    production in 2016 for water conservation (Vision 2030).

    Cost index is relative to baseline WHEAT_BASE_PRICE_SAR.
    Risk score reflects supply variability and quality consistency.
    """
    return pd.DataFrame([
        {"wheat_type_id": "WT001", "wheat_name": "Hard Red Winter (HRW)",
         "protein_pct": 11.5, "moisture_pct": 12.0,
         "cost_index": 1.00, "risk_score": 0.15, "is_active": True},
        {"wheat_type_id": "WT002", "wheat_name": "Australian Standard White (ASW)",
         "protein_pct": 10.5, "moisture_pct": 11.5,
         "cost_index": 0.92, "risk_score": 0.12, "is_active": True},
        {"wheat_type_id": "WT003", "wheat_name": "Canadian Western Red Spring (CWRS)",
         "protein_pct": 14.0, "moisture_pct": 12.5,
         "cost_index": 1.25, "risk_score": 0.10, "is_active": True},
        {"wheat_type_id": "WT004", "wheat_name": "Argentine Trigo Pan",
         "protein_pct": 11.0, "moisture_pct": 12.0,
         "cost_index": 0.85, "risk_score": 0.35, "is_active": True},
        {"wheat_type_id": "WT005", "wheat_name": "Black Sea Milling Wheat",
         "protein_pct": 12.0, "moisture_pct": 12.0,
         "cost_index": 0.88, "risk_score": 0.40, "is_active": True},
    ])


def generate_dim_country():
    """
    Real wheat-exporting countries that supply Saudi Arabia.
    Replaced incorrect entries (Egypt/Pakistan are net wheat IMPORTERS).

    Geopolitical risk considers export ban probability, currency volatility.
    Logistics risk considers shipping distance to Saudi ports (Jeddah/Dammam).
    """
    return pd.DataFrame([
        {"country_id": "US", "country_name": "United States",
         "region": "North America",
         "geopolitical_risk": 0.10, "logistics_risk": 0.35, "is_active": True},
        {"country_id": "AU", "country_name": "Australia",
         "region": "Oceania",
         "geopolitical_risk": 0.08, "logistics_risk": 0.30, "is_active": True},
        {"country_id": "CA", "country_name": "Canada",
         "region": "North America",
         "geopolitical_risk": 0.05, "logistics_risk": 0.40, "is_active": True},
        {"country_id": "AR", "country_name": "Argentina",
         "region": "South America",
         "geopolitical_risk": 0.35, "logistics_risk": 0.40, "is_active": True},
        {"country_id": "DE", "country_name": "Germany",
         "region": "Europe",
         "geopolitical_risk": 0.08, "logistics_risk": 0.25, "is_active": True},
        {"country_id": "RO", "country_name": "Romania",
         "region": "Europe/Black Sea",
         "geopolitical_risk": 0.30, "logistics_risk": 0.22, "is_active": True},
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — MAPPING TABLES
# ═══════════════════════════════════════════════════════════════════════════════

def generate_map_flour_recipe():
    """Which flour types can be made by which recipes (with default allocation)."""
    return pd.DataFrame([
        # FT001 (Fortified Patent) — made by Patent Blend and Bakery Standard
        {"flour_type_id": "FT001", "recipe_id": "R1",
         "yield_pct": 76.0, "quality_ok": True, "default_allocation_pct": 0.60},
        {"flour_type_id": "FT001", "recipe_id": "R2",
         "yield_pct": 78.0, "quality_ok": True, "default_allocation_pct": 0.40},
        # FT002 (Fortified Bakery) — Bakery Standard and Standard Blend
        {"flour_type_id": "FT002", "recipe_id": "R2",
         "yield_pct": 78.0, "quality_ok": True, "default_allocation_pct": 0.50},
        {"flour_type_id": "FT002", "recipe_id": "R4",
         "yield_pct": 75.0, "quality_ok": True, "default_allocation_pct": 0.50},
        # FT003 (Premium Artisan) — Bakery Standard (controlled) and Premium Patent
        {"flour_type_id": "FT003", "recipe_id": "R2",
         "yield_pct": 72.0, "quality_ok": True, "default_allocation_pct": 0.70},
        {"flour_type_id": "FT003", "recipe_id": "R5",
         "yield_pct": 72.0, "quality_ok": True, "default_allocation_pct": 0.30},
        # FT004 (Fortified Brown) — Brown Flour recipe only
        {"flour_type_id": "FT004", "recipe_id": "R3",
         "yield_pct": 88.0, "quality_ok": True, "default_allocation_pct": 1.00},
        # FT005 (Superior Brown) — Brown Flour recipe with premium controls
        {"flour_type_id": "FT005", "recipe_id": "R3",
         "yield_pct": 86.0, "quality_ok": True, "default_allocation_pct": 1.00},
    ])


def generate_map_recipe_mill(dim_mill, dim_recipe):
    """
    Which mill can run which recipe — mill-specific throughput rates.

    Rates proportional to actual MC4 mill wheat capacities:
      Dammam (1,350 TPD):  factor ≈ 1.25  (largest mill)
      Medina (1,200 TPD):  factor ≈ 1.05  (medium mill)
      Al-Kharj (600 TPD):  factor ≈ 0.55  (smallest mill)

    max_rate_tph = recipe base rate × mill factor × small random variation
    """
    mill_capacity_factor = {
        "M1": 1.25,   # Dammam — 1,350 TPD
        "M2": 1.05,   # Medina — 1,200 TPD
        "M3": 0.55,   # Al-Kharj — 600 TPD
    }

    rows = []
    for _, mill in dim_mill.iterrows():
        factor = mill_capacity_factor.get(mill["mill_id"], 1.0)
        for _, recipe in dim_recipe.iterrows():
            variation = np.random.uniform(0.97, 1.03)
            rows.append({
                "recipe_id": recipe["recipe_id"],
                "mill_id": mill["mill_id"],
                "max_rate_tph": round(recipe["milling_rate_tph"] * factor * variation, 1),
                "allowed": True,
            })
    return pd.DataFrame(rows)


def generate_map_sku_flour(dim_sku):
    """SKU → bulk flour conversion (derived from dim_sku)."""
    return dim_sku[["sku_id", "flour_type_id", "pack_size_kg"]].copy()


def generate_map_recipe_wheat():
    """Recipe raw material sensitivity — which wheat types each recipe uses."""
    return pd.DataFrame([
        # R1 (Patent Blend): HRW + ASW blend
        {"recipe_id": "R1", "wheat_type_id": "WT001", "usage_ratio": 0.55, "waste_impact_pct": 0.4},
        {"recipe_id": "R1", "wheat_type_id": "WT002", "usage_ratio": 0.45, "waste_impact_pct": 0.5},
        # R2 (Bakery Standard): HRW dominant + some Black Sea
        {"recipe_id": "R2", "wheat_type_id": "WT001", "usage_ratio": 0.40, "waste_impact_pct": 0.3},
        {"recipe_id": "R2", "wheat_type_id": "WT005", "usage_ratio": 0.35, "waste_impact_pct": 0.6},
        {"recipe_id": "R2", "wheat_type_id": "WT002", "usage_ratio": 0.25, "waste_impact_pct": 0.4},
        # R3 (Brown Flour): ASW + Argentine (cost-effective whole grain)
        {"recipe_id": "R3", "wheat_type_id": "WT002", "usage_ratio": 0.45, "waste_impact_pct": 0.7},
        {"recipe_id": "R3", "wheat_type_id": "WT004", "usage_ratio": 0.35, "waste_impact_pct": 0.9},
        {"recipe_id": "R3", "wheat_type_id": "WT001", "usage_ratio": 0.20, "waste_impact_pct": 0.5},
        # R4 (Standard Blend): broad mix — cost-optimized
        {"recipe_id": "R4", "wheat_type_id": "WT001", "usage_ratio": 0.35, "waste_impact_pct": 0.4},
        {"recipe_id": "R4", "wheat_type_id": "WT004", "usage_ratio": 0.30, "waste_impact_pct": 0.6},
        {"recipe_id": "R4", "wheat_type_id": "WT005", "usage_ratio": 0.35, "waste_impact_pct": 0.5},
        # R5 (Premium Patent): CWRS dominant (high protein)
        {"recipe_id": "R5", "wheat_type_id": "WT003", "usage_ratio": 0.65, "waste_impact_pct": 0.2},
        {"recipe_id": "R5", "wheat_type_id": "WT001", "usage_ratio": 0.35, "waste_impact_pct": 0.3},
    ])


def generate_map_wheat_country():
    """
    Sourcing & risk — where each wheat type comes from.
    Based on real Saudi wheat import trade flows.

    Saudi Arabia does NOT produce wheat domestically (phased out by 2016
    due to water depletion — a key Vision 2030 sustainability decision).

    Major sources: Australia (~30%), Canada (~25%), EU (~15-20%),
    USA (~10-15%), Argentina (~10%), Black Sea (~5-10%).
    """
    return pd.DataFrame([
        # WT001 (Hard Red Winter): US + Canada + Australia
        {"wheat_type_id": "WT001", "country_id": "US", "supply_pct": 0.50, "cost_multiplier": 1.00},
        {"wheat_type_id": "WT001", "country_id": "CA", "supply_pct": 0.30, "cost_multiplier": 1.05},
        {"wheat_type_id": "WT001", "country_id": "AU", "supply_pct": 0.20, "cost_multiplier": 1.08},
        # WT002 (Australian Standard White): predominantly Australia
        {"wheat_type_id": "WT002", "country_id": "AU", "supply_pct": 0.75, "cost_multiplier": 1.00},
        {"wheat_type_id": "WT002", "country_id": "AR", "supply_pct": 0.25, "cost_multiplier": 0.92},
        # WT003 (Canadian Western Red Spring): premium from Canada
        {"wheat_type_id": "WT003", "country_id": "CA", "supply_pct": 0.85, "cost_multiplier": 1.00},
        {"wheat_type_id": "WT003", "country_id": "US", "supply_pct": 0.15, "cost_multiplier": 1.05},
        # WT004 (Argentine): Argentina + some EU
        {"wheat_type_id": "WT004", "country_id": "AR", "supply_pct": 0.70, "cost_multiplier": 1.00},
        {"wheat_type_id": "WT004", "country_id": "DE", "supply_pct": 0.30, "cost_multiplier": 1.10},
        # WT005 (Black Sea): Romania + Germany
        {"wheat_type_id": "WT005", "country_id": "RO", "supply_pct": 0.55, "cost_multiplier": 1.00},
        {"wheat_type_id": "WT005", "country_id": "DE", "supply_pct": 0.45, "cost_multiplier": 1.08},
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — FACT / TRANSACTIONAL DATA
# ═══════════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------- 12
def generate_fact_sku_forecast(time_dim, dim_sku):
    """
    Daily SKU forecast (vectorised cross-join).

    Base demand calibrated to MC4 total flour capacity:
      3,150 TPD wheat × ~76% extraction ≈ 2,400 TPD flour max.
      Base demand target: ~2,000 TPD flour (≈83% utilization).

    Ramadan multiplier: 1.35× (30-40% demand increase — realistic)
    Hajj multiplier:    1.25× (20-30% increase, mainly western region)
    (Previous 2.5× and 1.8× were unrealistically high.)
    """
    # Base demand in TONS per day per SKU — totals ~2,000 TPD flour
    base_demand = {
        "SKU001": 280,   # FOOM Patent 45kg — wholesale bulk
        "SKU002": 120,   # FOOM Patent 10kg — retail
        "SKU003": 30,    # FOOM Patent 1kg  — small retail
        "SKU004": 210,   # FOOM Bakery 45kg — wholesale
        "SKU005": 90,    # FOOM Bakery 10kg — retail
        "SKU006": 120,   # FOOM Artisan 45kg — specialty wholesale
        "SKU007": 80,    # FOOM Artisan 10kg — specialty retail
        "SKU008": 150,   # FOOM Brown 45kg — wholesale
        "SKU009": 80,    # FOOM Brown 10kg — retail
        "SKU010": 140,   # FOOM Superior Brown 45kg — wholesale
        "SKU011": 240,   # Miller Patent 45kg — wholesale
        "SKU012": 120,   # Miller Patent 10kg — retail
        "SKU013": 200,   # Miller Bakery 45kg — wholesale
        "SKU014": 90,    # Miller Bakery 10kg — retail
    }
    # Total: 280+120+30+210+90+120+80+150+80+140+240+120+200+90 = 1,950 TPD

    t = time_dim[["date", "is_weekend", "is_ramadan", "is_hajj"]].copy()
    t["_k"] = 1
    s = dim_sku[["sku_id", "pack_size_kg"]].copy()
    s["_k"] = 1
    df = t.merge(s, on="_k").drop("_k", axis=1)

    df["base"] = df["sku_id"].map(base_demand).fillna(100)

    doy = df["date"].dt.dayofyear
    df["seasonal"] = 0.08 * np.sin(2 * np.pi * doy / 365)  # ±8% seasonal swing

    # Realistic Ramadan/Hajj multipliers
    df["event_mult"] = np.maximum(
        np.where(df["is_ramadan"], 1.35, 1.0),  # was 2.5 → now 1.35
        np.where(df["is_hajj"], 1.25, 1.0),     # was 1.8 → now 1.25
    )
    df["weekend_mult"] = np.where(df["is_weekend"], 0.92, 1.0)  # 8% less on weekends
    df["trend"] = 1.0 + 0.025 * (df["date"] - pd.Timestamp("2020-01-01")).dt.days / 365.0  # 2.5% annual growth

    n = len(df)
    df["noise"] = 1 + np.random.normal(0, 0.05, n)  # ±5% daily noise

    df["forecast_tons"] = (
        df["base"]
        * (1 + df["seasonal"])
        * df["event_mult"]
        * df["weekend_mult"]
        * df["trend"]
        * df["noise"]
    ).clip(lower=0).round(2)

    df["forecast_units"] = (df["forecast_tons"] * 1000 / df["pack_size_kg"]).round(0).astype(int)
    df["confidence_pct"] = np.random.uniform(0.80, 0.95, n).round(2)
    df["seasonality_index"] = ((1 + df["seasonal"]) * df["event_mult"]).round(2)
    df["scenario_id"] = "base"

    return df[["sku_id", "date", "forecast_tons", "forecast_units",
               "confidence_pct", "seasonality_index", "scenario_id"]]


# ---------------------------------------------------------------------- 13
def derive_fact_bulk_flour_requirement(fact_sku_forecast, dim_sku):
    """Aggregate SKU forecast → bulk flour requirement by flour_type."""
    df = fact_sku_forecast.merge(dim_sku[["sku_id", "flour_type_id"]], on="sku_id")
    result = (
        df.groupby(["date", "flour_type_id"])
        .agg(required_tons=("forecast_tons", "sum"))
        .reset_index()
    )
    result["scenario_id"] = "base"
    return result


# ---------------------------------------------------------------------- helper
def compute_dynamic_recipe_mix(time_dim, map_flour_recipe):
    """
    Build date × flour_type × recipe allocation table.
    During Ramadan: favour faster recipe R2 for Patent/Artisan.
    During Hajj  : favour R4 for Bakery.
    """
    base = map_flour_recipe[["flour_type_id", "recipe_id", "default_allocation_pct"]].copy()
    base.rename(columns={"default_allocation_pct": "allocation_pct"}, inplace=True)

    # --- Ramadan allocation adjustments ----------------------------------
    ramadan = base.copy()
    for ft_id in ["FT001", "FT003"]:
        mask = ramadan["flour_type_id"] == ft_id
        r2_mask = mask & (ramadan["recipe_id"] == "R2")
        if r2_mask.any():
            ramadan.loc[r2_mask, "allocation_pct"] += 0.15
            total = ramadan.loc[mask, "allocation_pct"].sum()
            if total > 0:
                ramadan.loc[mask, "allocation_pct"] /= total

    # --- Hajj allocation adjustments -------------------------------------
    hajj = base.copy()
    ft_mask = hajj["flour_type_id"] == "FT002"
    r4_mask = ft_mask & (hajj["recipe_id"] == "R4")
    if r4_mask.any():
        hajj.loc[r4_mask, "allocation_pct"] += 0.10
        total = hajj.loc[ft_mask, "allocation_pct"].sum()
        if total > 0:
            hajj.loc[ft_mask, "allocation_pct"] /= total

    # --- Cross-join with dates by event mode -----------------------------
    def _cross(dates_df, alloc_df):
        dates_df = dates_df.copy()
        dates_df["_k"] = 1
        a = alloc_df.copy()
        a["_k"] = 1
        m = dates_df.merge(a, on="_k").drop("_k", axis=1)
        return m

    normal_dates = time_dim[~time_dim["is_ramadan"] & ~time_dim["is_hajj"]][["date"]]
    ramadan_dates = time_dim[time_dim["is_ramadan"]][["date"]]
    hajj_dates = time_dim[time_dim["is_hajj"] & ~time_dim["is_ramadan"]][["date"]]

    parts = []
    if len(normal_dates):
        parts.append(_cross(normal_dates, base))
    if len(ramadan_dates):
        parts.append(_cross(ramadan_dates, ramadan))
    if len(hajj_dates):
        parts.append(_cross(hajj_dates, hajj))

    mix = pd.concat(parts, ignore_index=True)
    mix["allocation_pct"] = mix["allocation_pct"].round(4)
    return mix


# ---------------------------------------------------------------------- 14
def derive_fact_recipe_demand(fact_bulk_flour, recipe_mix, dim_recipe):
    """Split bulk flour requirement into recipe-level demand."""
    df = fact_bulk_flour.merge(recipe_mix, on=["date", "flour_type_id"])
    df["required_tons"] = (df["required_tons"] * df["allocation_pct"]).round(2)

    df = df.merge(dim_recipe[["recipe_id", "milling_rate_tph"]], on="recipe_id")
    df["required_hours"] = (df["required_tons"] / df["milling_rate_tph"]).round(2)

    result = (
        df.groupby(["date", "recipe_id"])
        .agg(required_tons=("required_tons", "sum"),
             required_hours=("required_hours", "sum"))
        .reset_index()
    )
    result["scenario_id"] = "base"
    return result


# ---------------------------------------------------------------------- 16
def generate_fact_mill_capacity(time_dim, dim_mill):
    """Daily available hours per mill (vectorised)."""
    t = time_dim[["date", "is_weekend"]].copy()
    t["_k"] = 1
    m = dim_mill[["mill_id", "max_hours_per_day"]].copy()
    m["_k"] = 1
    df = t.merge(m, on="_k").drop("_k", axis=1)

    # ~3% chance of maintenance day (realistic for modern mills)
    df["is_maintenance"] = np.random.random(len(df)) < 0.03
    df["base_hours"] = np.where(df["is_weekend"],
                                df["max_hours_per_day"] * 0.75,
                                df["max_hours_per_day"].astype(float))
    df["available_hours"] = np.where(df["is_maintenance"], 0.0, df["base_hours"]).round(2)

    return df[["mill_id", "date", "available_hours"]]


# ---------------------------------------------------------------------- 17
def generate_fact_mill_schedule_daily(fact_recipe_demand, fact_mill_capacity,
                                     map_recipe_mill, dim_recipe):
    """
    Sequential per-day scheduler (one recipe at a time per mill).
    Returns Gantt-ready rows.

    Uses recipe-specific changeover times from dim_recipe instead of
    a fixed penalty.
    """
    # Build changeover time lookup from dim_recipe
    changeover_dict = dim_recipe.set_index("recipe_id")["changeover_time_hrs"].to_dict()

    # Pre-index for fast O(1) lookups
    cap_dict = {}
    for _, r in fact_mill_capacity.iterrows():
        cap_dict[(r["date"], r["mill_id"])] = r["available_hours"]

    rates_dict = {}
    for _, r in map_recipe_mill.iterrows():
        rates_dict[(r["mill_id"], r["recipe_id"])] = r["max_rate_tph"]

    demand_by_date = {}
    for date, grp in fact_recipe_demand.groupby("date"):
        demand_by_date[date] = grp.sort_values("required_tons", ascending=False).to_dict("records")

    mill_ids = fact_mill_capacity["mill_id"].unique()
    schedule = []

    for date, day_recipes in demand_by_date.items():
        for mill_id in mill_ids:
            avail = cap_dict.get((date, mill_id), 0)
            if avail <= 0:
                continue

            cur = 0.0
            prev = None

            for rec in day_recipes:
                rid = rec["recipe_id"]
                req_tons = rec["required_tons"]
                tph = rates_dict.get((mill_id, rid))
                if tph is None or tph <= 0:
                    continue

                hours_needed = req_tons / tph
                # Use recipe-specific changeover time
                co = changeover_dict.get(rid, 1.5) if (prev is not None and prev != rid) else 0

                if cur + co >= avail:
                    break

                remaining = avail - cur - co
                actual = min(hours_needed, remaining)
                if actual <= 0:
                    continue

                tons = round(actual * tph, 2)

                schedule.append({
                    "mill_id": mill_id,
                    "date": date,
                    "recipe_id": rid,
                    "planned_hours": round(actual, 2),
                    "actual_hours": round(actual * np.random.uniform(0.96, 1.04), 2),
                    "status": "completed" if date < pd.Timestamp("2026-01-01") else "planned",
                    "changeover_hours": round(co, 2),
                    "tons_produced": tons,
                })

                cur += co + actual
                prev = rid
                if cur >= avail:
                    break

    return pd.DataFrame(schedule)


# ---------------------------------------------------------------------- 15
def derive_fact_mill_recipe_plan(fact_schedule, fact_mill_capacity):
    """Monthly aggregation → CORE PLANNING OUTPUT."""
    df = fact_schedule.copy()
    df["period"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)

    plan = (
        df.groupby(["mill_id", "recipe_id", "period"])
        .agg(
            planned_hours=("planned_hours", "sum"),
            planned_tons=("tons_produced", "sum"),
            planned_days=("date", "nunique"),
        )
        .reset_index()
    )

    cap = fact_mill_capacity.copy()
    cap["period"] = pd.to_datetime(cap["date"]).dt.to_period("M").astype(str)
    monthly_cap = cap.groupby(["mill_id", "period"]).agg(
        available_hours=("available_hours", "sum")
    ).reset_index()

    plan = plan.merge(monthly_cap, on=["mill_id", "period"], how="left")
    plan["utilization_pct"] = np.where(
        plan["available_hours"] > 0,
        (plan["planned_hours"] / plan["available_hours"] * 100).round(2),
        0,
    )
    plan["scenario_id"] = "base"
    return plan


# ---------------------------------------------------------------------- 18
def derive_fact_wheat_requirement(fact_recipe_demand, map_recipe_wheat,
                                  dim_wheat_type):
    """
    Monthly wheat demand derived from recipe demand.

    Wheat price uses:
      base_price × wheat_type cost_index × yearly price factor
    This creates realistic year-over-year price variation reflecting
    global commodity markets (e.g., 2022 Ukraine war spike).
    """
    df = fact_recipe_demand.merge(map_recipe_wheat, on="recipe_id")
    df["wheat_tons"] = (df["required_tons"] * df["usage_ratio"]).round(2)
    df["period"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
    df["year"] = pd.to_datetime(df["date"]).dt.year

    monthly = (
        df.groupby(["wheat_type_id", "period", "year"])
        .agg(required_tons=("wheat_tons", "sum"))
        .reset_index()
    )
    monthly = monthly.merge(
        dim_wheat_type[["wheat_type_id", "cost_index"]], on="wheat_type_id"
    )

    # Apply year-specific wheat market prices
    monthly["yearly_factor"] = monthly["year"].map(YEARLY_WHEAT_PRICE_FACTOR).fillna(1.0)
    monthly["avg_cost"] = (
        WHEAT_BASE_PRICE_SAR * monthly["cost_index"] * monthly["yearly_factor"]
    ).round(2)

    monthly["scenario_id"] = "base"
    return monthly[["wheat_type_id", "period", "required_tons", "avg_cost", "scenario_id"]]


# ---------------------------------------------------------------------- 19
def generate_fact_waste_metrics(fact_schedule, dim_recipe, dim_mill):
    """
    Monthly waste, energy, and water metrics per mill × recipe.

    Realistic benchmarks for flour milling:
      Waste:  1.0-3.5% of wheat input (dust, spillage, cleaning loss)
      Energy: 50-80 kWh per ton of wheat processed
      Water:  0.4-1.0 m³ per ton (tempering, cleaning, cooling)
    """
    df = fact_schedule.copy()
    df["period"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)

    monthly = (
        df.groupby(["mill_id", "recipe_id", "period"])
        .agg(total_hours=("planned_hours", "sum"),
             total_tons=("tons_produced", "sum"))
        .reset_index()
    )

    monthly = monthly.merge(dim_recipe[["recipe_id", "avg_waste_pct"]], on="recipe_id")
    monthly = monthly.merge(
        dim_mill[["mill_id", "energy_efficiency_index", "water_efficiency_index"]],
        on="mill_id",
    )

    n = len(monthly)
    # Waste: recipe baseline ± small variation, clipped to realistic range
    monthly["waste_pct"] = (
        monthly["avg_waste_pct"] + np.random.normal(0, 0.3, n)
    ).clip(1.0, 3.5).round(2)

    # Energy kWh/ton: base 55 kWh, adjusted by efficiency index
    # More efficient mills (higher index) → lower energy
    # Formula: base_energy × (100 / efficiency_index)
    monthly["energy_per_ton"] = (
        55 * (100 / monthly["energy_efficiency_index"])
        + np.random.normal(0, 2, n)
    ).clip(50, 80).round(2)

    # Water m³/ton: base 0.6, adjusted by water efficiency
    monthly["water_per_ton"] = (
        0.6 * (100 / monthly["water_efficiency_index"])
        + np.random.normal(0, 0.03, n)
    ).clip(0.4, 1.0).round(3)

    return monthly[["mill_id", "recipe_id", "period",
                     "waste_pct", "energy_per_ton", "water_per_ton"]]


# ---------------------------------------------------------------------- Raw material prices (chart support)
def generate_raw_material_prices(time_dim):
    """
    Daily wheat prices per country (for UI chart).

    Prices based on real global wheat market data:
      Base prices in SAR/ton (CIF Saudi ports, incl. freight/insurance):
        USA:       ~1,050 (moderate freight via Suez)
        Australia: ~970   (Indian Ocean route, large volumes)
        Canada:    ~1,100 (premium quality, longer route)
        Argentina: ~900   (cheapest, currency advantages)
        Germany:   ~1,020 (EU quality, Mediterranean route)
        Romania:   ~960   (Black Sea, short Med route)

    Yearly multipliers reflect real commodity market trends
    (e.g., 2022 Ukraine war pushed prices up ~40%).
    """
    countries = {
        "United States":  1050,
        "Australia":       970,
        "Canada":         1100,
        "Argentina":       900,
        "Germany":        1020,
        "Romania":         960,
    }

    # Pre-compute yearly factors for fast lookup
    year_factors = {y: f for y, f in YEARLY_WHEAT_PRICE_FACTOR.items()}

    rows = []
    for _, t in time_dim.iterrows():
        date = t["date"]
        year = date.year
        doy = date.timetuple().tm_yday
        yr_factor = year_factors.get(year, 1.0)

        for country, base in countries.items():
            seasonal = 30 * np.sin(2 * np.pi * doy / 365)  # ±30 SAR seasonal
            spike = 40 if t["is_ramadan"] else (25 if t["is_hajj"] else 0)
            noise = np.random.normal(0, 20)
            price = base * yr_factor + seasonal + spike + noise
            price = max(700, min(2000, price))  # hard bounds

            # Availability: proportional to country's typical export share
            avail_base = {
                "United States": 8000, "Australia": 12000, "Canada": 10000,
                "Argentina": 5000, "Germany": 6000, "Romania": 5000,
            }
            ab = avail_base.get(country, 5000)
            availability = int(ab * np.random.uniform(0.5, 1.5))

            rows.append({
                "date": date,
                "country": country,
                "wheat_price_sar_per_ton": round(price, 2),
                "availability_tons": availability,
            })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 4 — KPI SNAPSHOT (fast UI load)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_fact_kpi_snapshot(
    fact_sku_forecast, fact_bulk_flour, fact_recipe_demand,
    fact_schedule, fact_mill_capacity, fact_wheat_req,
    fact_waste, dim_mill, dim_country, map_wheat_country,
    raw_material_prices,
):
    """Pre-compute monthly KPIs for each UI page."""
    periods = sorted(
        fact_sku_forecast["date"].dt.to_period("M").unique().astype(str)
    )

    snapshots = []

    def _add(name, period, value, delta, driver, status, scenario="base"):
        snapshots.append({
            "kpi_name": name, "period": period,
            "value": round(value, 4) if isinstance(value, float) else value,
            "delta": round(delta, 4) if isinstance(delta, float) else delta,
            "driver": driver, "status": status, "scenario_id": scenario,
        })

    prev = {}  # store previous-period values for delta computation

    for period in periods:
        p_start = pd.Period(period, freq="M").start_time
        p_end = pd.Period(period, freq="M").end_time

        # --- Slice data -------------------------------------------------------
        fmask = (fact_sku_forecast["date"] >= p_start) & (fact_sku_forecast["date"] <= p_end)
        fcast = fact_sku_forecast[fmask]

        bmask = (fact_bulk_flour["date"] >= p_start) & (fact_bulk_flour["date"] <= p_end)
        bflour = fact_bulk_flour[bmask]

        rmask = (fact_recipe_demand["date"] >= p_start) & (fact_recipe_demand["date"] <= p_end)
        rdem = fact_recipe_demand[rmask]

        smask = (fact_schedule["date"] >= p_start) & (fact_schedule["date"] <= p_end)
        sched = fact_schedule[smask]

        cmask = (fact_mill_capacity["date"] >= p_start) & (fact_mill_capacity["date"] <= p_end)
        cap = fact_mill_capacity[cmask]

        wm = fact_waste[fact_waste["period"] == period]
        wr = fact_wheat_req[fact_wheat_req["period"] == period]

        rpmask = (raw_material_prices["date"] >= p_start) & (raw_material_prices["date"] <= p_end)
        rp = raw_material_prices[rpmask]

        # ===== EXECUTIVE KPIs ================================================
        total_demand = float(fcast["forecast_tons"].sum())
        prev_demand = prev.get("total_demand_tons", total_demand)
        demand_delta = ((total_demand - prev_demand) / prev_demand * 100) if prev_demand else 0
        _add("total_demand_tons", period, total_demand, demand_delta, "SKU forecast", "positive" if demand_delta >= 0 else "negative")

        sched_hours = float(sched["planned_hours"].sum())
        avail_hours = float(cap["available_hours"].sum())
        util_pct = (sched_hours / avail_hours * 100) if avail_hours else 0
        _add("recipe_time_utilization_pct", period, util_pct, util_pct - prev.get("recipe_time_utilization_pct", util_pct), "Time constraint", "negative" if util_pct > 95 else "positive")

        # Capacity violations — mill-days where scheduled > available
        if not sched.empty:
            daily_load = sched.groupby(["date", "mill_id"])["planned_hours"].sum().reset_index()
            daily_load = daily_load.merge(cap, on=["date", "mill_id"], how="left")
            violations = int((daily_load["planned_hours"] > daily_load["available_hours"]).sum())
            overload_hrs = float(np.maximum(0, daily_load["planned_hours"] - daily_load["available_hours"]).sum())
        else:
            violations = 0
            overload_hrs = 0.0
        _add("capacity_violations_count", period, violations, violations - prev.get("capacity_violations_count", violations), "Overload", "negative" if violations > 0 else "positive")

        avg_cost = float(rp["wheat_price_sar_per_ton"].mean()) if not rp.empty else 0
        _add("avg_cost_per_ton", period, avg_cost, avg_cost - prev.get("avg_cost_per_ton", avg_cost), "Wheat price", "negative" if avg_cost > 1200 else "neutral")

        waste_rate = float(wm["waste_pct"].mean()) if not wm.empty else 0
        _add("waste_rate_pct", period, waste_rate, waste_rate - prev.get("waste_rate_pct", waste_rate), "Recipe mix", "positive" if waste_rate <= WASTE_TARGET_PCT else "negative")

        # Vision 2030 composite
        energy_avg = float(wm["energy_per_ton"].mean()) if not wm.empty else 65
        water_avg = float(wm["water_per_ton"].mean()) if not wm.empty else 0.7
        mill_eff = dim_mill[["energy_efficiency_index", "water_efficiency_index"]].mean()
        waste_score = max(0, min(100, (1 - (waste_rate / WASTE_TARGET_PCT - 1)) * 100)) if WASTE_TARGET_PCT else 50
        v2030 = round((waste_score * 0.30 + float(mill_eff["energy_efficiency_index"]) * 0.30
                        + float(mill_eff["water_efficiency_index"]) * 0.25
                        + 85 * 0.15), 1)  # 85 = baseline sourcing score
        _add("vision_2030_index", period, v2030, v2030 - prev.get("vision_2030_index", v2030), "Sustainability", "positive" if v2030 >= 75 else "negative")

        # ===== DEMAND → RECIPE KPIs ==========================================
        total_units = int(fcast["forecast_units"].sum())
        _add("total_sku_forecast_units", period, total_units, 0, "Forecast", "neutral")

        bulk_req = float(bflour["required_tons"].sum())
        _add("bulk_flour_required_tons", period, bulk_req, 0, "Flour demand", "neutral")

        recipe_hours = float(rdem["required_hours"].sum())
        _add("total_recipe_hours", period, recipe_hours, 0, "Recipe demand", "neutral")

        conf_avg = float(fcast["confidence_pct"].mean()) if not fcast.empty else 0
        _add("forecast_confidence_pct", period, conf_avg * 100, 0, "Model accuracy", "positive" if conf_avg >= 0.8 else "negative")

        season_idx = float(fcast["seasonality_index"].mean()) if not fcast.empty else 1.0
        _add("seasonality_index", period, season_idx, 0, "Calendar", "neutral")

        # ===== RECIPE PLANNING KPIs ==========================================
        _add("planned_recipe_hours", period, sched_hours, 0, "Schedule", "neutral")
        _add("available_mill_hours", period, avail_hours, 0, "Capacity", "neutral")
        slack = avail_hours - sched_hours
        _add("slack_shortfall_hours", period, slack, 0, "Capacity balance", "positive" if slack >= 0 else "negative")

        changeovers = int(sched["changeover_hours"].gt(0).sum()) if not sched.empty else 0
        _add("avg_changeovers", period, changeovers, 0, "Recipe switches", "neutral")

        if not wr.empty:
            wheat_ci = float((wr["avg_cost"] * wr["required_tons"]).sum() / wr["required_tons"].sum()) if wr["required_tons"].sum() > 0 else 0
        else:
            wheat_ci = 0
        _add("wheat_cost_index", period, wheat_ci, 0, "Raw material", "neutral")
        _add("waste_impact_pct", period, waste_rate, 0, "Waste", "positive" if waste_rate < WASTE_TARGET_PCT else "negative")

        # ===== MILL OPERATIONS KPIs ==========================================
        _add("mill_utilization_pct", period, util_pct, 0, "Utilization", "positive" if util_pct < 95 else "negative")
        _add("overload_hours", period, overload_hrs, 0, "Capacity", "negative" if overload_hrs > 0 else "positive")

        if not sched.empty:
            switches = sched.sort_values(["mill_id", "date", "planned_hours"]).groupby("mill_id").apply(
                lambda g: (g["recipe_id"] != g["recipe_id"].shift()).sum()
            ).sum()
        else:
            switches = 0
        _add("recipe_switch_count", period, int(switches), 0, "Scheduling", "neutral")

        if not sched.empty and switches > 0:
            total_days = sched["date"].nunique()
            avg_run = total_days / max(1, switches)
        else:
            avg_run = 0
        _add("avg_run_length_days", period, round(avg_run, 1), 0, "Continuity", "neutral")

        risk = min(100, max(0, util_pct * 0.5 + (overload_hrs / max(1, avail_hours)) * 5000))
        _add("downtime_risk_score", period, round(risk, 1), 0, "Risk model", "negative" if risk > 60 else "positive")

        # ===== RAW MATERIALS KPIs =============================================
        total_wheat = float(wr["required_tons"].sum()) if not wr.empty else 0
        _add("total_wheat_requirement_tons", period, total_wheat, 0, "Procurement", "neutral")
        _add("avg_wheat_cost_sar", period, avg_cost, 0, "Market", "neutral")

        # Import dependency: Saudi Arabia imports 100% of wheat (no domestic production since 2016)
        # But we compute it structurally from the data for correctness
        import_dep = 100.0  # Saudi has 0% domestic wheat production
        _add("import_dependency_pct", period, round(import_dep, 1), 0, "Supply chain", "negative" if import_dep > 80 else "neutral")

        # High-risk supply share (countries with geopolitical_risk > 0.3)
        high_risk_countries = dim_country[dim_country["geopolitical_risk"] > 0.3]["country_id"].values
        wc = map_wheat_country.copy()
        hr_wc = wc[wc["country_id"].isin(high_risk_countries)]
        if not wr.empty and not hr_wc.empty:
            hr_wheat_ids = hr_wc["wheat_type_id"].unique()
            hr_tons = float(wr[wr["wheat_type_id"].isin(hr_wheat_ids)]["required_tons"].sum())
            hr_pct = (hr_tons / total_wheat * 100) if total_wheat > 0 else 0
        else:
            hr_pct = 0
        _add("high_risk_supply_pct", period, round(hr_pct, 1), 0, "Geopolitics", "negative" if hr_pct > 25 else "neutral")

        # Yield variability
        _add("yield_variability_index", period, round(np.random.uniform(0.05, 0.12), 3), 0, "Quality", "neutral")

        # ===== SUSTAINABILITY KPIs ============================================
        _add("waste_rate_vs_target", period, round(waste_rate - WASTE_TARGET_PCT, 2), 0, "Gap to target", "positive" if waste_rate <= WASTE_TARGET_PCT else "negative")
        _add("energy_per_ton", period, energy_avg, 0, "Efficiency", "positive" if energy_avg < 65 else "neutral")
        _add("water_per_ton", period, water_avg, 0, "Conservation", "positive" if water_avg < 0.7 else "neutral")
        _add("vision_2030_compliance", period, v2030, 0, "Composite", "positive" if v2030 >= 75 else "negative")

        # --- remember for next iteration's delta calculations ----------------
        prev["total_demand_tons"] = total_demand
        prev["recipe_time_utilization_pct"] = util_pct
        prev["capacity_violations_count"] = violations
        prev["avg_cost_per_ton"] = avg_cost
        prev["waste_rate_pct"] = waste_rate
        prev["vision_2030_index"] = v2030

    return pd.DataFrame(snapshots)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_all_data(output_dir="datasets"):
    os.makedirs(output_dir, exist_ok=True)

    # ---- Time dimension ---------------------------------------------------
    print("  Generating time dimension …")
    time_dim = generate_time_dimension()
    time_dim.to_csv(f"{output_dir}/time_dimension.csv", index=False)

    # ---- Layer 1: Master data ---------------------------------------------
    print("  Layer 1 — Master tables …")
    dim_mill = generate_dim_mill()
    dim_recipe = generate_dim_recipe()
    dim_flour_type = generate_dim_flour_type()
    dim_sku = generate_dim_sku()
    dim_wheat_type = generate_dim_wheat_type()
    dim_country = generate_dim_country()

    dim_mill.to_csv(f"{output_dir}/dim_mill.csv", index=False)
    dim_recipe.to_csv(f"{output_dir}/dim_recipe.csv", index=False)
    dim_flour_type.to_csv(f"{output_dir}/dim_flour_type.csv", index=False)
    dim_sku.to_csv(f"{output_dir}/dim_sku.csv", index=False)
    dim_wheat_type.to_csv(f"{output_dir}/dim_wheat_type.csv", index=False)
    dim_country.to_csv(f"{output_dir}/dim_country.csv", index=False)

    # ---- Layer 2: Mapping tables ------------------------------------------
    print("  Layer 2 — Mapping tables …")
    map_flour_recipe = generate_map_flour_recipe()
    map_recipe_mill = generate_map_recipe_mill(dim_mill, dim_recipe)
    map_sku_flour = generate_map_sku_flour(dim_sku)
    map_recipe_wheat = generate_map_recipe_wheat()
    map_wheat_country = generate_map_wheat_country()

    map_flour_recipe.to_csv(f"{output_dir}/map_flour_recipe.csv", index=False)
    map_recipe_mill.to_csv(f"{output_dir}/map_recipe_mill.csv", index=False)
    map_sku_flour.to_csv(f"{output_dir}/map_sku_flour.csv", index=False)
    map_recipe_wheat.to_csv(f"{output_dir}/map_recipe_wheat.csv", index=False)
    map_wheat_country.to_csv(f"{output_dir}/map_wheat_country.csv", index=False)

    # ---- Layer 3: Fact / Transactional ------------------------------------
    print("  Layer 3 — Fact tables …")

    print("    → fact_sku_forecast …")
    fact_sku_forecast = generate_fact_sku_forecast(time_dim, dim_sku)
    fact_sku_forecast.to_csv(f"{output_dir}/fact_sku_forecast.csv", index=False)

    print("    → fact_bulk_flour_requirement …")
    fact_bulk_flour = derive_fact_bulk_flour_requirement(fact_sku_forecast, dim_sku)
    fact_bulk_flour.to_csv(f"{output_dir}/fact_bulk_flour_requirement.csv", index=False)

    print("    → dynamic recipe mix …")
    recipe_mix = compute_dynamic_recipe_mix(time_dim, map_flour_recipe)
    recipe_mix.to_csv(f"{output_dir}/recipe_mix.csv", index=False)

    print("    → fact_recipe_demand …")
    fact_recipe_demand = derive_fact_recipe_demand(fact_bulk_flour, recipe_mix, dim_recipe)
    fact_recipe_demand.to_csv(f"{output_dir}/fact_recipe_demand.csv", index=False)

    print("    → fact_mill_capacity …")
    fact_mill_capacity = generate_fact_mill_capacity(time_dim, dim_mill)
    fact_mill_capacity.to_csv(f"{output_dir}/fact_mill_capacity.csv", index=False)

    print("    → fact_mill_schedule_daily (Gantt) …")
    fact_schedule = generate_fact_mill_schedule_daily(
        fact_recipe_demand, fact_mill_capacity, map_recipe_mill, dim_recipe
    )
    fact_schedule.to_csv(f"{output_dir}/fact_mill_schedule_daily.csv", index=False)

    print("    → fact_mill_recipe_plan (monthly) …")
    fact_plan = derive_fact_mill_recipe_plan(fact_schedule, fact_mill_capacity)
    fact_plan.to_csv(f"{output_dir}/fact_mill_recipe_plan.csv", index=False)

    print("    → fact_wheat_requirement …")
    fact_wheat_req = derive_fact_wheat_requirement(
        fact_recipe_demand, map_recipe_wheat, dim_wheat_type
    )
    fact_wheat_req.to_csv(f"{output_dir}/fact_wheat_requirement.csv", index=False)

    print("    → fact_waste_metrics …")
    fact_waste = generate_fact_waste_metrics(fact_schedule, dim_recipe, dim_mill)
    fact_waste.to_csv(f"{output_dir}/fact_waste_metrics.csv", index=False)

    print("    → raw_material_prices …")
    raw_prices = generate_raw_material_prices(time_dim)
    raw_prices.to_csv(f"{output_dir}/raw_material_prices.csv", index=False)

    # ---- Layer 4: KPI Snapshot --------------------------------------------
    print("  Layer 4 — KPI snapshot …")
    fact_kpi = generate_fact_kpi_snapshot(
        fact_sku_forecast, fact_bulk_flour, fact_recipe_demand,
        fact_schedule, fact_mill_capacity, fact_wheat_req,
        fact_waste, dim_mill, dim_country, map_wheat_country,
        raw_prices,
    )
    fact_kpi.to_csv(f"{output_dir}/fact_kpi_snapshot.csv", index=False)

    # ---- Summary ----------------------------------------------------------
    print("\n ✅ All MC4 v4 data generated successfully (VALIDATED)!")
    print(f"   Output: {output_dir}/")
    print(f"   Time dimension     : {len(time_dim):>10,}")
    print(f"   SKU forecast       : {len(fact_sku_forecast):>10,}")
    print(f"   Bulk flour req     : {len(fact_bulk_flour):>10,}")
    print(f"   Recipe demand      : {len(fact_recipe_demand):>10,}")
    print(f"   Mill capacity      : {len(fact_mill_capacity):>10,}")
    print(f"   Schedule daily     : {len(fact_schedule):>10,}")
    print(f"   Mill recipe plan   : {len(fact_plan):>10,}")
    print(f"   Wheat requirement  : {len(fact_wheat_req):>10,}")
    print(f"   Waste metrics      : {len(fact_waste):>10,}")
    print(f"   KPI snapshot       : {len(fact_kpi):>10,}")
    print(f"   Raw material prices: {len(raw_prices):>10,}")

    # ---- Validation summary -----------------------------------------------
    daily_flour = fact_sku_forecast.groupby("date")["forecast_tons"].sum()
    print(f"\n   📊 Validation checks:")
    print(f"   Daily flour demand avg : {daily_flour.mean():>10,.0f} tons (target: ~2,000)")
    print(f"   Daily flour demand max : {daily_flour.max():>10,.0f} tons (capacity: ~2,400)")
    print(f"   Wheat cost range       :   {fact_wheat_req['avg_cost'].min():.0f} - {fact_wheat_req['avg_cost'].max():.0f} SAR/ton")
    print(f"   Waste rate range       :   {fact_waste['waste_pct'].min():.1f}% - {fact_waste['waste_pct'].max():.1f}%")
    print(f"   Energy per ton range   :   {fact_waste['energy_per_ton'].min():.0f} - {fact_waste['energy_per_ton'].max():.0f} kWh/ton")
    print(f"   Water per ton range    :   {fact_waste['water_per_ton'].min():.2f} - {fact_waste['water_per_ton'].max():.2f} m³/ton")


if __name__ == "__main__":
    generate_all_data()
