"""
Data Export Pipeline
Exports pricing recommendations and reports to Excel for CRM integration
"""
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import EXPORTS_DIR, EXPORT_CONFIG
from src.database.models import (
    Project,
    BasePricing,
    PriceRecommendation,
    DynamicSignal,
    CompetitorData,
    PriceHistory,
    get_session,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataExporter:
    """Handles data export from database to Excel"""

    def __init__(self, user_role: str = "analyst"):
        """
        Initialize DataExporter

        Args:
            user_role: Role of user performing the export
        """
        self.user_role = user_role
        self.session = get_session()

    def export_crm_price_update(
        self,
        project_ids: Optional[List[str]] = None,
        status_filter: str = "approved",
        filename: Optional[str] = None,
    ) -> Path:
        """
        Export approved prices in CRM-ready format

        Args:
            project_ids: List of project IDs to export (None = all)
            status_filter: Filter by approval status (approved, applied)
            filename: Custom filename (defaults to timestamped)

        Returns:
            Path to exported Excel file
        """
        logger.info(f"Exporting CRM price update (status={status_filter})...")

        # Query approved price recommendations
        query = self.session.query(
            PriceRecommendation.project_id,
            Project.project_name,
            PriceRecommendation.unit_type,
            PriceRecommendation.approved_price_m2,
            PriceRecommendation.recommendation_date,
            PriceRecommendation.approved_by,
            PriceRecommendation.approved_at,
        ).join(Project, PriceRecommendation.project_id == Project.project_id)

        # Apply filters
        if status_filter:
            query = query.filter(PriceRecommendation.status == status_filter)
        if project_ids:
            query = query.filter(PriceRecommendation.project_id.in_(project_ids))

        # Execute query
        results = query.all()

        if not results:
            logger.warning("No approved prices found to export")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(
            results,
            columns=[
                "project_id",
                "project_name",
                "unit_type",
                "price_m2",
                "recommendation_date",
                "approved_by",
                "approved_at",
            ],
        )

        # Add metadata columns for CRM
        df["export_date"] = datetime.now()
        df["export_by"] = self.user_role
        df["status"] = "ready_for_import"

        # Generate filename
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"CRM_Price_Update_{timestamp}.xlsx"

        output_path = EXPORTS_DIR / filename

        # Export to Excel with formatting
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Price_Updates", index=False)

            # Get workbook and worksheet
            workbook = writer.book
            worksheet = writer.sheets["Price_Updates"]

            # Format columns
            worksheet.column_dimensions["A"].width = 15  # project_id
            worksheet.column_dimensions["B"].width = 35  # project_name
            worksheet.column_dimensions["C"].width = 15  # unit_type
            worksheet.column_dimensions["D"].width = 12  # price_m2
            worksheet.column_dimensions["E"].width = 18  # recommendation_date
            worksheet.column_dimensions["F"].width = 18  # approved_by
            worksheet.column_dimensions["G"].width = 20  # approved_at

            # Add summary sheet
            summary_df = pd.DataFrame(
                {
                    "Metric": [
                        "Total Projects",
                        "Total Unit Types",
                        "Average Price (KZT/m²)",
                        "Export Date",
                        "Exported By",
                    ],
                    "Value": [
                        df["project_id"].nunique(),
                        len(df),
                        f"{df['price_m2'].mean():,.0f}",
                        datetime.now().strftime("%Y-%m-%d %H:%M"),
                        self.user_role,
                    ],
                }
            )
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

        logger.info(f"✓ Exported {len(df)} price updates to: {output_path}")
        return output_path

    def export_monthly_analysis_report(
        self, month: Optional[datetime] = None, filename: Optional[str] = None
    ) -> Path:
        """
        Export comprehensive monthly analysis report

        Args:
            month: Month to analyze (defaults to current month)
            filename: Custom filename

        Returns:
            Path to exported Excel file
        """
        if month is None:
            month = datetime.now()

        logger.info(f"Generating monthly analysis report for {month.strftime('%B %Y')}...")

        # Calculate date range for the month
        month_start = month.replace(day=1)
        if month.month == 12:
            month_end = month.replace(year=month.year + 1, month=1, day=1)
        else:
            month_end = month.replace(month=month.month + 1, day=1)

        # Collect data for report
        report_data = {}

        # 1. Sales Performance by Project
        sales_query = (
            self.session.query(
                DynamicSignal.project_id,
                Project.project_name,
                DynamicSignal.unit_type_name,
                DynamicSignal.units_sold,
                DynamicSignal.m2_sold,
                DynamicSignal.total_sales_value,
                DynamicSignal.avg_price_m2,
                DynamicSignal.stock_remaining,
                DynamicSignal.sales_velocity,
                DynamicSignal.conversion_rate,
            )
            .join(Project, DynamicSignal.project_id == Project.project_id)
            .filter(DynamicSignal.date >= month_start.date())
            .filter(DynamicSignal.date < month_end.date())
        )
        report_data["Sales_Performance"] = pd.read_sql(
            sales_query.statement, self.session.bind
        )

        # 2. Price Recommendations Status
        rec_query = (
            self.session.query(
                PriceRecommendation.project_id,
                Project.project_name,
                PriceRecommendation.unit_type,
                PriceRecommendation.current_price_m2,
                PriceRecommendation.recommended_price_m2,
                PriceRecommendation.price_change_pct,
                PriceRecommendation.status,
                PriceRecommendation.confidence_score,
            )
            .join(Project, PriceRecommendation.project_id == Project.project_id)
            .filter(PriceRecommendation.recommendation_date >= month_start.date())
            .filter(PriceRecommendation.recommendation_date < month_end.date())
        )
        report_data["Price_Recommendations"] = pd.read_sql(
            rec_query.statement, self.session.bind
        )

        # 3. Competitor Analysis
        comp_query = (
            self.session.query(
                CompetitorData.competitor_name,
                CompetitorData.complex_name,
                CompetitorData.location,
                CompetitorData.housing_class,
                CompetitorData.unit_type,
                CompetitorData.avg_price_m2,
                CompetitorData.discount_flag,
            )
            .filter(CompetitorData.date >= month_start.date())
            .filter(CompetitorData.date < month_end.date())
        )
        report_data["Competitor_Prices"] = pd.read_sql(
            comp_query.statement, self.session.bind
        )

        # 4. Projects Overview
        projects_query = self.session.query(
            Project.project_id,
            Project.project_name,
            Project.location,
            Project.region,
            Project.housing_class,
            Project.status,
            Project.developer,
        )
        report_data["Projects_Overview"] = pd.read_sql(
            projects_query.statement, self.session.bind
        )

        # Generate filename
        if filename is None:
            month_str = month.strftime("%Y_%m")
            filename = f"Monthly_Analysis_Report_{month_str}.xlsx"

        output_path = EXPORTS_DIR / filename

        # Export to Excel with multiple sheets
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for sheet_name, df in report_data.items():
                if not df.empty:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

            # Add executive summary
            summary_data = {
                "Metric": [
                    "Report Period",
                    "Total Projects",
                    "Total Sales (Units)",
                    "Total Sales Value (KZT)",
                    "Average Price (KZT/m²)",
                    "Pending Recommendations",
                    "Approved Recommendations",
                    "Competitors Tracked",
                    "Generated Date",
                ],
                "Value": [
                    month.strftime("%B %Y"),
                    report_data["Projects_Overview"]["project_id"].nunique(),
                    (
                        report_data["Sales_Performance"]["units_sold"].sum()
                        if not report_data["Sales_Performance"].empty
                        else 0
                    ),
                    (
                        f"{report_data['Sales_Performance']['total_sales_value'].sum():,.0f}"
                        if not report_data["Sales_Performance"].empty
                        else "0"
                    ),
                    (
                        f"{report_data['Sales_Performance']['avg_price_m2'].mean():,.0f}"
                        if not report_data["Sales_Performance"].empty
                        else "0"
                    ),
                    (
                        len(
                            report_data["Price_Recommendations"][
                                report_data["Price_Recommendations"]["status"] == "pending"
                            ]
                        )
                        if not report_data["Price_Recommendations"].empty
                        else 0
                    ),
                    (
                        len(
                            report_data["Price_Recommendations"][
                                report_data["Price_Recommendations"]["status"] == "approved"
                            ]
                        )
                        if not report_data["Price_Recommendations"].empty
                        else 0
                    ),
                    (
                        report_data["Competitor_Prices"]["competitor_name"].nunique()
                        if not report_data["Competitor_Prices"].empty
                        else 0
                    ),
                    datetime.now().strftime("%Y-%m-%d %H:%M"),
                ],
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name="Executive_Summary", index=False)

        logger.info(f"✓ Exported monthly report to: {output_path}")
        return output_path

    def export_project_pricing_template(
        self, project_id: str, filename: Optional[str] = None
    ) -> Path:
        """
        Export pricing template for a specific project

        Args:
            project_id: Project ID to export
            filename: Custom filename

        Returns:
            Path to exported Excel file
        """
        logger.info(f"Exporting pricing template for project: {project_id}")

        # Get project info
        project = (
            self.session.query(Project).filter_by(project_id=project_id).first()
        )
        if not project:
            raise ValueError(f"Project not found: {project_id}")

        # Get base pricing data
        pricing_query = (
            self.session.query(
                BasePricing.unit_type,
                BasePricing.area_m2,
                BasePricing.finish_type,
                BasePricing.base_price_m2,
                BasePricing.base_unit_price,
                BasePricing.market_price_avg,
                BasePricing.developer_margin_pct,
            )
            .filter_by(project_id=project_id)
            .order_by(BasePricing.unit_type)
        )
        pricing_df = pd.read_sql(pricing_query.statement, self.session.bind)

        # Get recent sales data
        sales_query = (
            self.session.query(
                DynamicSignal.date,
                DynamicSignal.unit_type_name,
                DynamicSignal.units_sold,
                DynamicSignal.avg_price_m2,
                DynamicSignal.stock_remaining,
                DynamicSignal.sales_velocity,
            )
            .filter_by(project_id=project_id)
            .order_by(DynamicSignal.date.desc())
            .limit(100)
        )
        sales_df = pd.read_sql(sales_query.statement, self.session.bind)

        # Generate filename
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d")
            filename = f"Pricing_Template_{project_id}_{timestamp}.xlsx"

        output_path = EXPORTS_DIR / filename

        # Export to Excel
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            # Project info sheet
            info_df = pd.DataFrame(
                {
                    "Field": ["Project ID", "Project Name", "Location", "Region", "Class", "Status"],
                    "Value": [
                        project.project_id,
                        project.project_name,
                        project.location,
                        project.region,
                        project.housing_class,
                        project.status,
                    ],
                }
            )
            info_df.to_excel(writer, sheet_name="Project_Info", index=False)

            # Current pricing
            pricing_df.to_excel(writer, sheet_name="Current_Pricing", index=False)

            # Sales history
            sales_df.to_excel(writer, sheet_name="Sales_History", index=False)

        logger.info(f"✓ Exported pricing template to: {output_path}")
        return output_path

    def __del__(self):
        """Cleanup: close database session"""
        if hasattr(self, "session"):
            self.session.close()


# ============================================================================
# STANDALONE SCRIPT EXECUTION
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export data from pricing system")
    parser.add_argument(
        "export_type",
        choices=["crm", "monthly", "project"],
        help="Type of export to generate",
    )
    parser.add_argument(
        "--project-id", help="Project ID (for project export)", default=None
    )
    parser.add_argument("--filename", help="Custom output filename", default=None)

    args = parser.parse_args()

    exporter = DataExporter(user_role="analyst")

    if args.export_type == "crm":
        output = exporter.export_crm_price_update(filename=args.filename)
        print(f"✅ CRM price update exported to: {output}")

    elif args.export_type == "monthly":
        output = exporter.export_monthly_analysis_report(filename=args.filename)
        print(f"✅ Monthly report exported to: {output}")

    elif args.export_type == "project":
        if not args.project_id:
            print("❌ Error: --project-id required for project export")
            sys.exit(1)
        output = exporter.export_project_pricing_template(
            args.project_id, filename=args.filename
        )
        print(f"✅ Project template exported to: {output}")
