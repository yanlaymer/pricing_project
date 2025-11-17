"""
Configuration file for Dynamic Pricing Analytics System
Contains all application settings, constants, and parameters
"""
import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXPORTS_DIR = DATA_DIR / "exports"
DATABASE_PATH = DATA_DIR / "pricing_system.db"

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EXPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATABASE SETTINGS
# ============================================================================
DATABASE_CONFIG = {
    "echo": False,  # Set to True for SQL debugging
    "pool_size": 5,
    "max_overflow": 10,
    "pool_recycle": 3600,
}

# ============================================================================
# PRICING MODEL PARAMETERS
# ============================================================================

# IMPORTANT: Base price (construction_cost_m2) already includes ALL costs:
# - Land cost
# - Construction materials and labor
# - Infrastructure
# - Overhead and administrative costs
# This is the actual all-in cost from the company's accounting.

# Default coefficients for unit attributes (can be adjusted via UI)
# These are applied to the market-based pricing, NOT to construction costs
DEFAULT_COEFFICIENTS = {
    "location_coef": 1.08,  # Location premium
    "floor_coef": 1.03,     # Floor premium
    "view_coef": 1.05,      # View premium
    "finish_coef": 0.98,    # Finish type adjustment (0.98 for черновая)
}

# Developer margin ranges by housing class
# Applied on top of the all-in construction cost
MARGIN_BY_CLASS = {
    "Комфорт": {"min": 0.15, "default": 0.20, "max": 0.25},
    "Комфорт+": {"min": 0.18, "default": 0.23, "max": 0.28},
    "Бизнес": {"min": 0.20, "default": 0.25, "max": 0.30},
    "Премиум": {"min": 0.25, "default": 0.30, "max": 0.35},
}

# These are ONLY used if cost breakdown data becomes available in the future
# Currently NOT used since construction_cost_m2 is all-inclusive
ESTIMATED_COSTS = {
    "land_cost_m2_default": 50000,      # KZT per m² (not used - for future)
    "infra_cost_m2_default": 75000,     # KZT per m² (not used - for future)
    "overhead_m2_default": 30000,       # KZT per m² (not used - for future)
}

# Sales velocity targets (units per month by unit type)
TARGET_VELOCITY = {
    "1-комнатная": 15,
    "2-комнатная": 12,
    "3-комнатная": 8,
    "4-комнатная": 5,
    "5-комнатная": 3,
}

# ============================================================================
# DEMAND CURVE PARAMETERS
# ============================================================================

# Linear regression settings for demand curve fitting
DEMAND_MODEL_CONFIG = {
    "min_data_points": 3,        # Minimum sales records needed for regression
    "lookback_period_days": 90,  # Look back 90 days for demand analysis
    "confidence_threshold": 0.6, # R² threshold for model confidence
}

# ============================================================================
# COMPETITIVE ANALYSIS SETTINGS
# ============================================================================

# Competitive positioning strategies
POSITIONING_STRATEGIES = {
    "aggressive": -0.05,    # 5% below market average
    "neutral": 0.00,        # At market average
    "premium": 0.05,        # 5% above market average
    "luxury": 0.10,         # 10% above market average
}

# Competitor weight by proximity (km radius)
COMPETITOR_PROXIMITY_WEIGHTS = {
    "radius_0_2km": 1.0,   # Same district - full weight
    "radius_2_5km": 0.7,   # Nearby - reduced weight
    "radius_5_10km": 0.4,  # Moderate distance
    "radius_10plus": 0.2,  # Far - minimal weight
}

# ============================================================================
# DATA VALIDATION RULES
# ============================================================================

VALIDATION_RULES = {
    "price_m2": {
        "min": 200000,    # Minimum reasonable price per m²
        "max": 2000000,   # Maximum reasonable price per m²
    },
    "area_m2": {
        "min": 20,        # Minimum apartment area
        "max": 300,       # Maximum apartment area
    },
    "margin_pct": {
        "min": 0.10,      # Minimum margin (10%)
        "max": 0.50,      # Maximum margin (50%)
    },
}

# ============================================================================
# UI SETTINGS
# ============================================================================

# Streamlit page configuration
STREAMLIT_CONFIG = {
    "page_title": "Dynamic Pricing Analytics",
    "page_icon": "📊",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# Color scheme
COLOR_SCHEME = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "success": "#2ca02c",
    "warning": "#ffd700",
    "danger": "#d62728",
    "info": "#17becf",
}

# User roles and permissions
USER_ROLES = {
    "analyst": {
        "name": "Analyst",
        "permissions": ["data_import", "data_validation", "report_generation"],
        "icon": "📈",
    },
    "sales_manager": {
        "name": "Sales Manager",
        "permissions": ["view_prices", "approve_prices", "override_prices"],
        "icon": "💼",
    },
    "executive": {
        "name": "Executive",
        "permissions": ["view_all", "strategic_decisions", "approve_final"],
        "icon": "👔",
    },
    "marketing": {
        "name": "Marketing",
        "permissions": ["competitor_analysis", "market_research", "view_prices"],
        "icon": "📣",
    },
}

# ============================================================================
# EXCEL IMPORT/EXPORT SETTINGS
# ============================================================================

# Expected sheet names in source Excel file
EXCEL_SHEETS = {
    "projects": "residential_projects",
    "base_pricing": "pricing_start_base",
    "dynamic_signals": "pricing_dynamic_signals",
    "competitors": "competitor_market_data",
    "reference": "Cправочник",
}

# Column mappings for import (handles Russian column names)
COLUMN_MAPPINGS = {
    "data": "date",
    "м2": "area_m2",
    "цена": "price",
}

# Export template settings
EXPORT_CONFIG = {
    "crm_template": "CRM_Price_Upload_Template.xlsx",
    "report_template": "Monthly_Analysis_Report.xlsx",
    "date_format": "%Y-%m-%d",
    "number_format": "#,##0.00",
}

# ============================================================================
# LOGGING SETTINGS
# ============================================================================

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "file": {
            "class": "logging.FileHandler",
            "filename": str(BASE_DIR / "logs" / "pricing_system.log"),
            "formatter": "standard",
            "level": "INFO",
        },
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO",
        },
    },
    "loggers": {
        "": {
            "handlers": ["file", "console"],
            "level": "INFO",
            "propagate": True,
        },
    },
}

# Create logs directory
(BASE_DIR / "logs").mkdir(exist_ok=True)

# ============================================================================
# VERSION INFO
# ============================================================================

VERSION = "1.0.0"
APP_NAME = "Dynamic Pricing Analytics System"
COMPANY = "Sensata Real Estate"
