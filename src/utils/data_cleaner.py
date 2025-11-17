"""
Data Cleaning Module
Fixes data quality issues identified in the initial analysis
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import re
import logging
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    ESTIMATED_COSTS,
    VALIDATION_RULES,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaner:
    """Handles all data cleaning operations"""

    def __init__(self, excel_path: str):
        """
        Initialize DataCleaner

        Args:
            excel_path: Path to the Excel file to clean
        """
        self.excel_path = Path(excel_path)
        self.excel_file = pd.ExcelFile(self.excel_path)
        self.cleaned_data = {}

    def clean_all_sheets(self) -> Dict[str, pd.DataFrame]:
        """
        Clean all sheets in the Excel file

        Returns:
            Dictionary of cleaned DataFrames by sheet name
        """
        logger.info(f"Starting data cleaning for {self.excel_path}")

        # Clean each sheet
        self.cleaned_data["residential_projects"] = self.clean_projects_sheet()
        self.cleaned_data["pricing_start_base"] = self.clean_pricing_base_sheet()
        self.cleaned_data["pricing_dynamic_signals"] = self.clean_dynamic_signals_sheet()
        self.cleaned_data["competitor_market_data"] = self.clean_competitor_sheet()

        logger.info("Data cleaning completed successfully")
        return self.cleaned_data

    def clean_projects_sheet(self) -> pd.DataFrame:
        """Clean residential_projects sheet"""
        logger.info("Cleaning residential_projects sheet...")

        df = pd.read_excel(self.excel_path, sheet_name="residential_projects")

        # Fix floors_total (handle ranges like "9-12")
        df["floors_min"] = df["floors_total"].apply(self._extract_min_floors)
        df["floors_max"] = df["floors_total"].apply(self._extract_max_floors)

        # Keep original for reference
        df["floors_total_original"] = df["floors_total"]

        # Use average for single numeric value
        df["floors_total"] = ((df["floors_min"] + df["floors_max"]) / 2).astype(int)

        logger.info(f"✓ Cleaned {len(df)} project records")
        return df

    def clean_pricing_base_sheet(self) -> pd.DataFrame:
        """Clean pricing_start_base sheet"""
        logger.info("Cleaning pricing_start_base sheet...")

        df = pd.read_excel(self.excel_path, sheet_name="pricing_start_base")

        # Note: land_cost_m2, infra_cost_m2, overhead_m2 are intentionally empty
        # because construction_cost_m2 already includes ALL costs (all-in cost)
        # Just flag them as not available (NULL/NaN is correct state)
        if df["land_cost_m2"].isna().all():
            logger.info(
                "ℹ️  land_cost_m2 is empty (expected - construction_cost_m2 is all-inclusive)"
            )

        if df["infra_cost_m2"].isna().all():
            logger.info(
                "ℹ️  infra_cost_m2 is empty (expected - construction_cost_m2 is all-inclusive)"
            )

        if df["overhead_m2"].isna().all():
            logger.info(
                "ℹ️  overhead_m2 is empty (expected - construction_cost_m2 is all-inclusive)"
            )

        # Fill missing market_price_avg with regional average
        if df["market_price_avg"].isna().sum() > 0:
            missing_count = df["market_price_avg"].isna().sum()
            for region in df["region_n2"].unique():
                region_avg = df[df["region_n2"] == region]["market_price_avg"].mean()
                df.loc[
                    (df["region_n2"] == region) & (df["market_price_avg"].isna()),
                    "market_price_avg",
                ] = region_avg
            logger.info(f"✓ Filled {missing_count} missing market_price_avg with regional averages")

        # Handle zero values in pricing fields
        df = self._handle_zero_prices(df)

        # Validate price ranges
        df = self._validate_prices(df)

        logger.info(f"✓ Cleaned {len(df)} pricing base records")
        return df

    def clean_dynamic_signals_sheet(self) -> pd.DataFrame:
        """Clean pricing_dynamic_signals sheet"""
        logger.info("Cleaning pricing_dynamic_signals sheet...")

        df = pd.read_excel(self.excel_path, sheet_name="pricing_dynamic_signals")

        # Fix units_sold data type (stored as text like "4-комнатная")
        # Extract numeric unit type
        df["unit_type_name"] = df["units_sold"]
        df["units_sold"] = df["unit_type"]  # Use the numeric unit_type column

        # Standardize unit type names
        df["unit_type_name"] = df["unit_type_name"].astype(str)

        # Convert date columns to datetime
        df["date"] = pd.to_datetime(df["date"])
        df["data"] = pd.to_datetime(df["data"])

        # Handle zero/missing values in key metrics
        # Many zeros might be legitimate (no sales that period), but flag suspicious patterns
        df["has_suspicious_zeros"] = (
            (df["market_demand_index"] == 0)
            & (df["sales_velocity"] == 0)
            & (df["leads_received"] == 0)
        )

        # Calculate derived metrics if missing
        if "conversion_rate" not in df.columns or df["conversion_rate"].isna().all():
            df["conversion_rate"] = np.where(
                df["leads_received"] > 0,
                df["contracts_signed"] / df["leads_received"],
                0,
            )

        logger.info(f"✓ Cleaned {len(df)} dynamic signal records")
        return df

    def clean_competitor_sheet(self) -> pd.DataFrame:
        """Clean competitor_market_data sheet"""
        logger.info("Cleaning competitor_market_data sheet...")

        df = pd.read_excel(self.excel_path, sheet_name="competitor_market_data")

        # Convert date column to datetime
        df["date"] = pd.to_datetime(df["date"])

        # Handle zero values in avg_price_m2
        zero_prices = (df["avg_price_m2"] == 0).sum()
        if zero_prices > 0:
            logger.warning(f"⚠️  Found {zero_prices} competitor records with zero price")
            # Mark these for review but don't remove
            df["price_missing"] = df["avg_price_m2"] == 0

        # Standardize location names (remove extra spaces)
        df["location"] = df["location"].str.strip()
        df["competitor_name"] = df["competitor_name"].str.strip()
        df["complex_name"] = df["complex_name"].str.strip()

        logger.info(f"✓ Cleaned {len(df)} competitor records")
        return df

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _extract_min_floors(self, value: str) -> int:
        """Extract minimum floor number from range or single value"""
        if pd.isna(value):
            return 0

        value_str = str(value).strip()

        # Check if it's a range (e.g., "9-12")
        if "-" in value_str:
            try:
                return int(value_str.split("-")[0])
            except ValueError:
                return 0

        # Single number
        try:
            return int(value_str)
        except ValueError:
            return 0

    def _extract_max_floors(self, value: str) -> int:
        """Extract maximum floor number from range or single value"""
        if pd.isna(value):
            return 0

        value_str = str(value).strip()

        # Check if it's a range (e.g., "9-12")
        if "-" in value_str:
            try:
                return int(value_str.split("-")[1])
            except ValueError:
                return 0

        # Single number
        try:
            return int(value_str)
        except ValueError:
            return 0

    def _handle_zero_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle zero values in pricing fields"""
        zero_base_price = (df["base_price_m2"] == 0).sum()
        zero_unit_price = (df["base_unit_price"] == 0).sum()

        if zero_base_price > 0:
            logger.warning(f"⚠️  Found {zero_base_price} records with zero base_price_m2")
            # Calculate from unit price if available, otherwise use construction cost + margin
            mask = df["base_price_m2"] == 0
            df.loc[mask, "base_price_m2"] = np.where(
                df.loc[mask, "base_unit_price"] > 0,
                df.loc[mask, "base_unit_price"] / df.loc[mask, "area_m2"],
                df.loc[mask, "construction_cost_m2"] * 1.25,  # Fallback: construction cost + 25% margin
            )

        if zero_unit_price > 0:
            logger.warning(f"⚠️  Found {zero_unit_price} records with zero base_unit_price")
            # Calculate from price per m2 if available
            mask = df["base_unit_price"] == 0
            df.loc[mask, "base_unit_price"] = (
                df.loc[mask, "base_price_m2"] * df.loc[mask, "area_m2"]
            )

        return df

    def _validate_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate price ranges against business rules"""
        # Check base_price_m2
        min_price = VALIDATION_RULES["price_m2"]["min"]
        max_price = VALIDATION_RULES["price_m2"]["max"]

        out_of_range = (
            (df["base_price_m2"] < min_price) | (df["base_price_m2"] > max_price)
        ).sum()

        if out_of_range > 0:
            logger.warning(
                f"⚠️  Found {out_of_range} records with prices outside valid range "
                f"({min_price:,} - {max_price:,} KZT/m²)"
            )
            df["price_validation_flag"] = (
                df["base_price_m2"] < min_price
            ) | (df["base_price_m2"] > max_price)

        # Check area ranges
        min_area = VALIDATION_RULES["area_m2"]["min"]
        max_area = VALIDATION_RULES["area_m2"]["max"]

        invalid_area = (
            (df["area_m2"] < min_area) | (df["area_m2"] > max_area)
        ).sum()

        if invalid_area > 0:
            logger.warning(
                f"⚠️  Found {invalid_area} records with area outside valid range "
                f"({min_area} - {max_area} m²)"
            )
            df["area_validation_flag"] = (df["area_m2"] < min_area) | (
                df["area_m2"] > max_area
            )

        return df

    # ========================================================================
    # EXPORT METHODS
    # ========================================================================

    def save_cleaned_data(self, output_format: str = "both") -> Dict[str, Path]:
        """
        Save cleaned data to files

        Args:
            output_format: 'excel', 'csv', or 'both'

        Returns:
            Dictionary of output file paths
        """
        output_paths = {}

        if output_format in ["excel", "both"]:
            excel_path = PROCESSED_DATA_DIR / "cleaned_pricing_data.xlsx"
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                for sheet_name, df in self.cleaned_data.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            output_paths["excel"] = excel_path
            logger.info(f"✓ Saved cleaned data to Excel: {excel_path}")

        if output_format in ["csv", "both"]:
            csv_dir = PROCESSED_DATA_DIR / "csv"
            csv_dir.mkdir(exist_ok=True)
            for sheet_name, df in self.cleaned_data.items():
                csv_path = csv_dir / f"{sheet_name}.csv"
                df.to_csv(csv_path, index=False)
                output_paths[f"csv_{sheet_name}"] = csv_path
            logger.info(f"✓ Saved cleaned data to CSV: {csv_dir}")

        return output_paths

    def generate_quality_report(self) -> pd.DataFrame:
        """
        Generate data quality report

        Returns:
            DataFrame with quality metrics by sheet
        """
        report_data = []

        for sheet_name, df in self.cleaned_data.items():
            report_data.append(
                {
                    "sheet": sheet_name,
                    "total_records": len(df),
                    "total_columns": len(df.columns),
                    "missing_values": df.isna().sum().sum(),
                    "duplicate_records": df.duplicated().sum(),
                    "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
                }
            )

        report_df = pd.DataFrame(report_data)
        logger.info("\n" + "=" * 80)
        logger.info("DATA QUALITY REPORT")
        logger.info("=" * 80)
        logger.info("\n" + report_df.to_string(index=False))
        logger.info("=" * 80)

        return report_df


# ============================================================================
# STANDALONE SCRIPT EXECUTION
# ============================================================================
if __name__ == "__main__":
    import sys

    # Check if file path provided
    if len(sys.argv) < 2:
        print("Usage: python data_cleaner.py <path_to_excel_file>")
        sys.exit(1)

    excel_file = sys.argv[1]

    # Create cleaner and run
    cleaner = DataCleaner(excel_file)
    cleaner.clean_all_sheets()
    cleaner.save_cleaned_data(output_format="both")
    cleaner.generate_quality_report()

    print("\n✅ Data cleaning completed successfully!")
