"""
Data Import Pipeline
Handles Excel data import, cleaning, validation, and database loading
"""
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List
import logging
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RAW_DATA_DIR, EXCEL_SHEETS
from src.utils.data_cleaner import DataCleaner
from src.database.models import (
    Base,
    Project,
    Unit,
    BasePricing,
    DynamicSignal,
    CompetitorData,
    AuditLog,
    get_session,
    init_database,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataImporter:
    """Handles data import from Excel to database"""

    def __init__(self, user_role: str = "analyst"):
        """
        Initialize DataImporter

        Args:
            user_role: Role of user performing the import (for audit log)
        """
        self.user_role = user_role
        self.session = get_session()
        self.import_stats = {
            "projects": 0,
            "base_pricing": 0,
            "dynamic_signals": 0,
            "competitors": 0,
            "errors": [],
        }

    def import_excel_file(
        self, excel_path: str, clean_first: bool = True
    ) -> Dict[str, int]:
        """
        Import data from Excel file to database

        Args:
            excel_path: Path to Excel file
            clean_first: Whether to clean data before import

        Returns:
            Dictionary with import statistics
        """
        logger.info(f"Starting data import from: {excel_path}")

        try:
            # Step 1: Clean data if requested
            if clean_first:
                logger.info("Step 1/4: Cleaning data...")
                cleaner = DataCleaner(excel_path)
                cleaned_data = cleaner.clean_all_sheets()
            else:
                logger.info("Step 1/4: Loading raw data (skipping cleaning)...")
                cleaned_data = self._load_raw_data(excel_path)

            # Step 2: Import projects
            logger.info("Step 2/4: Importing projects...")
            self._import_projects(cleaned_data["residential_projects"])

            # Step 3: Import pricing data
            logger.info("Step 3/4: Importing pricing data...")
            self._import_base_pricing(cleaned_data["pricing_start_base"])
            self._import_dynamic_signals(cleaned_data["pricing_dynamic_signals"])

            # Step 4: Import competitor data
            logger.info("Step 4/4: Importing competitor data...")
            self._import_competitor_data(cleaned_data["competitor_market_data"])

            # Commit all changes
            self.session.commit()

            # Log the import action
            self._log_action(
                action_type="data_import",
                description=f"Imported data from {Path(excel_path).name}",
                metadata=self.import_stats,
            )

            logger.info("✅ Data import completed successfully!")
            logger.info(f"   Projects: {self.import_stats['projects']}")
            logger.info(f"   Base Pricing: {self.import_stats['base_pricing']}")
            logger.info(f"   Dynamic Signals: {self.import_stats['dynamic_signals']}")
            logger.info(f"   Competitors: {self.import_stats['competitors']}")

            if self.import_stats["errors"]:
                logger.warning(f"   Errors: {len(self.import_stats['errors'])}")

            return self.import_stats

        except Exception as e:
            self.session.rollback()
            error_msg = f"Import failed: {str(e)}"
            logger.error(error_msg)
            self.import_stats["errors"].append(error_msg)
            raise

        finally:
            self.session.close()

    def _load_raw_data(self, excel_path: str) -> Dict[str, pd.DataFrame]:
        """Load raw data without cleaning"""
        return {
            "residential_projects": pd.read_excel(
                excel_path, sheet_name="residential_projects"
            ),
            "pricing_start_base": pd.read_excel(excel_path, sheet_name="pricing_start_base"),
            "pricing_dynamic_signals": pd.read_excel(
                excel_path, sheet_name="pricing_dynamic_signals"
            ),
            "competitor_market_data": pd.read_excel(
                excel_path, sheet_name="competitor_market_data"
            ),
        }

    def _import_projects(self, df: pd.DataFrame) -> None:
        """Import project data"""
        for _, row in df.iterrows():
            try:
                # Check if project already exists
                existing = (
                    self.session.query(Project)
                    .filter_by(project_id=row["project_id"])
                    .first()
                )

                if existing:
                    # Update existing project
                    existing.project_name = row["project_name"]
                    existing.location = row["location"]
                    existing.region = row["region"]
                    existing.housing_class = row["housing_class"]
                    existing.construction_material = row.get("construction_material")
                    existing.floors_min = row.get("floors_min")
                    existing.floors_max = row.get("floors_max")
                    existing.floors_total = row.get("floors_total")
                    existing.blocks_total = row.get("blocks_total")
                    existing.developer = row.get("developer")
                    existing.status = row.get("status")
                    existing.crm_id = row.get("crm_id")
                    existing.updated_at = datetime.utcnow()
                else:
                    # Create new project
                    project = Project(
                        project_id=row["project_id"],
                        project_name=row["project_name"],
                        location=row["location"],
                        region=row["region"],
                        housing_class=row["housing_class"],
                        construction_material=row.get("construction_material"),
                        floors_min=row.get("floors_min"),
                        floors_max=row.get("floors_max"),
                        floors_total=row.get("floors_total"),
                        blocks_total=row.get("blocks_total"),
                        developer=row.get("developer"),
                        status=row.get("status"),
                        crm_id=row.get("crm_id"),
                    )
                    self.session.add(project)
                    self.import_stats["projects"] += 1

            except Exception as e:
                error_msg = f"Error importing project {row.get('project_id')}: {str(e)}"
                logger.error(error_msg)
                self.import_stats["errors"].append(error_msg)

    def _import_base_pricing(self, df: pd.DataFrame) -> None:
        """Import base pricing data"""
        # Clear existing base pricing (we'll reimport everything)
        self.session.query(BasePricing).delete()

        for _, row in df.iterrows():
            try:
                pricing = BasePricing(
                    base_id=row.get("base_id"),
                    project_id=row["project_id"],
                    unit_type=row["unit_type"],
                    area_m2=row["area_m2"],
                    finish_type=row.get("finish_type"),
                    land_cost_m2=row.get("land_cost_m2"),
                    construction_cost_m2=row["construction_cost_m2"],
                    infra_cost_m2=row.get("infra_cost_m2"),
                    overhead_m2=row.get("overhead_m2"),
                    land_cost_estimated=row.get("land_cost_estimated", False),
                    infra_cost_estimated=row.get("infra_cost_estimated", False),
                    overhead_cost_estimated=row.get("overhead_cost_estimated", False),
                    developer_margin_pct=row["developer_margin_pct"],
                    location_coef=row.get("location_coef", 1.0),
                    floor_coef=row.get("floor_coef", 1.0),
                    view_coef=row.get("view_coef", 1.0),
                    finish_coef=row.get("finish_coef", 1.0),
                    market_price_avg=row.get("market_price_avg"),
                    base_price_m2=row["base_price_m2"],
                    base_unit_price=row["base_unit_price"],
                    baseprice_m2=row.get("baseprice_m2"),
                    region_n2=row.get("region_n2"),
                    version_tag=row.get("version_tag"),
                    effective_date=pd.to_datetime(row.get("data")).date()
                    if pd.notna(row.get("data"))
                    else None,
                    price_validation_flag=row.get("price_validation_flag", False),
                    area_validation_flag=row.get("area_validation_flag", False),
                )
                self.session.add(pricing)
                self.import_stats["base_pricing"] += 1

            except Exception as e:
                error_msg = f"Error importing base pricing for {row.get('project_id')}/{row.get('unit_type')}: {str(e)}"
                logger.error(error_msg)
                self.import_stats["errors"].append(error_msg)

    def _import_dynamic_signals(self, df: pd.DataFrame) -> None:
        """Import dynamic signals data"""
        # Clear existing dynamic signals (we'll reimport everything)
        self.session.query(DynamicSignal).delete()

        for _, row in df.iterrows():
            try:
                signal = DynamicSignal(
                    signal_id=row.get("signal_id"),
                    project_id=row["project_id"],
                    date=pd.to_datetime(row["date"]).date(),
                    unit_type=row.get("unit_type"),
                    unit_type_name=row.get("unit_type_name", row.get("units_sold")),
                    units_sold=row.get("units_sold", 0)
                    if isinstance(row.get("units_sold"), (int, float))
                    else 0,
                    m2_sold=row.get("m2_sold", 0.0),
                    total_sales_value=row.get("total_sales_value", 0.0),
                    avg_price_m2=row.get("avg_price_m2"),
                    stock_remaining=row.get("stock_remaining", 0),
                    market_demand_index=row.get("market_demand_index", 0.0),
                    sales_velocity=row.get("sales_velocity", 0),
                    price_change_pct=row.get("price_change_pct", 0.0),
                    interest_rate_pct=row.get("interest_rate_pct", 0.0),
                    leads_received=row.get("leads_received", 0),
                    office_visits=row.get("office_visits", 0),
                    contracts_signed=row.get("contracts_signed", 0),
                    mortgage_selected=row.get("mortgage_selected", 0),
                    conversion_rate=row.get("conversion_rate", 0.0),
                    conversion_rate2=row.get("conversion_rate2", 0.0),
                    has_suspicious_zeros=row.get("has_suspicious_zeros", False),
                )
                self.session.add(signal)
                self.import_stats["dynamic_signals"] += 1

            except Exception as e:
                error_msg = f"Error importing dynamic signal for {row.get('project_id')}/{row.get('date')}: {str(e)}"
                logger.error(error_msg)
                self.import_stats["errors"].append(error_msg)

    def _import_competitor_data(self, df: pd.DataFrame) -> None:
        """Import competitor data"""
        # Clear existing competitor data (we'll reimport everything)
        self.session.query(CompetitorData).delete()

        for _, row in df.iterrows():
            try:
                competitor = CompetitorData(
                    comp_id=row.get("comp_id"),
                    date=pd.to_datetime(row["date"]).date(),
                    location=row["location"],
                    competitor_name=row["competitor_name"],
                    complex_name=row["complex_name"],
                    housing_class=row.get("class"),
                    unit_type=row.get("unit_type"),
                    data_period=row.get("data"),
                    avg_price_m2=row.get("avg_price_m2"),
                    discount_flag=row.get("discount_flag", False) == 1,
                    price_missing=row.get("price_missing", False),
                    source_url=row.get("source_url"),
                )
                self.session.add(competitor)
                self.import_stats["competitors"] += 1

            except Exception as e:
                error_msg = f"Error importing competitor data for {row.get('competitor_name')}/{row.get('complex_name')}: {str(e)}"
                logger.error(error_msg)
                self.import_stats["errors"].append(error_msg)

    def _log_action(
        self,
        action_type: str,
        description: str,
        entity_type: str = None,
        entity_id: str = None,
        metadata: Dict = None,
    ) -> None:
        """Log an action to the audit log"""
        try:
            import json

            log_entry = AuditLog(
                user_role=self.user_role,
                action_type=action_type,
                entity_type=entity_type,
                entity_id=entity_id,
                description=description,
                metadata_json=json.dumps(metadata) if metadata else None,
            )
            self.session.add(log_entry)
        except Exception as e:
            logger.error(f"Failed to write audit log: {str(e)}")

    def get_import_summary(self) -> str:
        """Generate human-readable import summary"""
        summary = f"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                          DATA IMPORT SUMMARY                               ║
╚═══════════════════════════════════════════════════════════════════════════╝

✅ Import Status: {"SUCCESS" if not self.import_stats["errors"] else "COMPLETED WITH ERRORS"}

📊 Records Imported:
   • Projects: {self.import_stats['projects']}
   • Base Pricing: {self.import_stats['base_pricing']}
   • Dynamic Signals: {self.import_stats['dynamic_signals']}
   • Competitors: {self.import_stats['competitors']}

Total Records: {sum([v for k, v in self.import_stats.items() if k != "errors"])}
"""

        if self.import_stats["errors"]:
            summary += f"\n⚠️  Errors Encountered: {len(self.import_stats['errors'])}\n"
            for i, error in enumerate(self.import_stats["errors"][:5], 1):
                summary += f"   {i}. {error}\n"
            if len(self.import_stats["errors"]) > 5:
                summary += f"   ... and {len(self.import_stats['errors']) - 5} more\n"

        summary += "\n" + "=" * 80

        return summary


# ============================================================================
# STANDALONE SCRIPT EXECUTION
# ============================================================================
if __name__ == "__main__":
    import sys

    # Check if file path provided
    if len(sys.argv) < 2:
        print("Usage: python data_importer.py <path_to_excel_file>")
        sys.exit(1)

    excel_file = sys.argv[1]

    # Initialize database if it doesn't exist
    if not Path(PROJECT_ROOT / "data" / "pricing_system.db").exists():
        print("Initializing database...")
        init_database()
        print()

    # Create importer and run
    importer = DataImporter(user_role="analyst")
    stats = importer.import_excel_file(excel_file, clean_first=True)

    # Print summary
    print(importer.get_import_summary())
