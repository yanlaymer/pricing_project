"""
Analyst UI Pages
Data import, validation, and model parameter management
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RAW_DATA_DIR, EXPORTS_DIR
from config.translations import t
from src.database.data_importer import DataImporter
from src.database.data_exporter import DataExporter
from src.database.models import get_session, Project, BasePricing, DynamicSignal, CompetitorData


def show_data_import_page():
    """Data import page for analysts"""
    st.header(f"📥 {t('data_import_title')}")

    st.markdown(t('data_import_desc'))

    # File uploader
    uploaded_file = st.file_uploader(
        t('choose_excel'),
        type=["xlsx", "xls"],
        help=t('choose_excel_help'),
    )

    col1, col2 = st.columns(2)
    with col1:
        clean_data = st.checkbox(
            t('clean_before_import'),
            value=True,
            help=t('clean_help'),
        )
    with col2:
        show_preview = st.checkbox(t('show_preview'), value=False)

    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_path = RAW_DATA_DIR / uploaded_file.name
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"✅ {t('file_uploaded')}: {uploaded_file.name}")

        # Show preview if requested
        if show_preview:
            with st.expander(f"📊 {t('data_preview')}"):
                try:
                    sheets = pd.ExcelFile(temp_path).sheet_names
                    selected_sheet = st.selectbox(t('select_sheet'), sheets)
                    preview_df = pd.read_excel(temp_path, sheet_name=selected_sheet, nrows=10)
                    st.dataframe(preview_df)
                except Exception as e:
                    st.error(f"{t('error')}: {e}")

        # Import button
        if st.button(f"🚀 {t('import_data')}", type="primary", use_container_width=True):
            with st.spinner(f"{t('importing_data')}"):
                try:
                    # Create importer
                    importer = DataImporter(user_role="analyst")

                    # Import data
                    stats = importer.import_excel_file(str(temp_path), clean_first=clean_data)

                    # Show success message
                    st.success(f"✅ {t('import_completed')}")

                    # Display statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(t('total_projects'), stats["projects"])
                    with col2:
                        st.metric(t('pricing_records'), stats["base_pricing"])
                    with col3:
                        st.metric(t('sales_signals'), stats["dynamic_signals"])
                    with col4:
                        st.metric(t('competitor_data'), stats["competitors"])

                    # Show errors if any
                    if stats["errors"]:
                        with st.expander(f"⚠️ {t('import_warnings')}", expanded=False):
                            for error in stats["errors"]:
                                st.warning(error)

                    # Show summary
                    with st.expander(f"📋 {t('import_summary')}", expanded=True):
                        st.code(importer.get_import_summary())

                except Exception as e:
                    st.error(f"❌ {t('import_failed')}: {str(e)}")
                    st.exception(e)


def show_data_validation_page():
    """Data validation dashboard"""
    st.header(f"✅ {t('data_validation_title')}")

    session = get_session()

    # Database statistics
    st.subheader(f"📊 {t('database_overview')}")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        project_count = session.query(Project).count()
        st.metric(t('total_projects'), project_count)

    with col2:
        pricing_count = session.query(BasePricing).count()
        st.metric(t('pricing_records'), pricing_count)

    with col3:
        signal_count = session.query(DynamicSignal).count()
        st.metric(t('sales_signals'), signal_count)

    with col4:
        comp_count = session.query(CompetitorData).count()
        st.metric(t('competitor_data'), comp_count)

    # Data quality checks
    st.subheader(f"🔍 {t('data_quality_checks')}")

    # Check for validation flags
    validation_issues = (
        session.query(BasePricing)
        .filter(
            (BasePricing.price_validation_flag == True)
            | (BasePricing.area_validation_flag == True)
        )
        .count()
    )

    if validation_issues > 0:
        st.warning(f"⚠️ {validation_issues} {t('validation_issues')}")

        with st.expander(t('view_flagged')):
            flagged_query = (
                session.query(
                    BasePricing.project_id,
                    BasePricing.unit_type,
                    BasePricing.base_price_m2,
                    BasePricing.area_m2,
                    BasePricing.price_validation_flag,
                    BasePricing.area_validation_flag,
                )
                .filter(
                    (BasePricing.price_validation_flag == True)
                    | (BasePricing.area_validation_flag == True)
                )
            )
            flagged_df = pd.read_sql(flagged_query.statement, session.bind)
            st.dataframe(flagged_df, use_container_width=True)
    else:
        st.success(f"✅ {t('no_issues_found')}")

    # Check for missing competitor prices
    missing_comp_prices = (
        session.query(CompetitorData)
        .filter(CompetitorData.price_missing == True)
        .count()
    )

    if missing_comp_prices > 0:
        st.info(f"ℹ️ {missing_comp_prices} {t('missing_comp_prices')}")

    # Project-level validation
    st.subheader(f"📋 {t('projects_validation')}")

    projects = session.query(Project).all()
    project_data = []

    for project in projects:
        pricing_count = (
            session.query(BasePricing).filter_by(project_id=project.project_id).count()
        )
        signal_count = (
            session.query(DynamicSignal)
            .filter_by(project_id=project.project_id)
            .count()
        )

        project_data.append(
            {
                "Project ID": project.project_id,
                "Project Name": project.project_name,
                "Status": project.status,
                t('pricing_records'): pricing_count,
                t('sales_signals'): signal_count,
                "Complete": "✅" if pricing_count > 0 and signal_count > 0 else "⚠️",
            }
        )

    project_df = pd.DataFrame(project_data)
    st.dataframe(project_df, use_container_width=True)

    session.close()


def show_model_parameters_page():
    """Model parameters configuration"""
    st.header(f"🔧 {t('model_params_title')}")

    st.markdown(t('model_params_desc'))

    # Import configuration
    from config import (
        MARGIN_BY_CLASS,
        DEFAULT_COEFFICIENTS,
        TARGET_VELOCITY,
        POSITIONING_STRATEGIES,
    )

    # Margin settings
    st.subheader(f"💰 {t('developer_margins')}")

    margin_data = []
    for housing_class, margins in MARGIN_BY_CLASS.items():
        margin_data.append(
            {
                t('housing_class'): housing_class,
                t('min_margin'): f"{margins['min']:.1%}",
                t('default_margin'): f"{margins['default']:.1%}",
                t('max_margin'): f"{margins['max']:.1%}",
            }
        )

    st.dataframe(pd.DataFrame(margin_data), use_container_width=True)

    # Attribute coefficients
    st.subheader(f"📐 {t('unit_coefficients')}")

    coef_data = pd.DataFrame(
        {
            t('attribute'): [
                t('location_premium'),
                t('floor_premium'),
                t('view_premium'),
                t('finish_adjustment'),
            ],
            t('coefficient'): [
                DEFAULT_COEFFICIENTS["location_coef"],
                DEFAULT_COEFFICIENTS["floor_coef"],
                DEFAULT_COEFFICIENTS["view_coef"],
                DEFAULT_COEFFICIENTS["finish_coef"],
            ],
            t('effect'): [
                f"{(DEFAULT_COEFFICIENTS['location_coef']-1)*100:+.0f}%",
                f"{(DEFAULT_COEFFICIENTS['floor_coef']-1)*100:+.0f}%",
                f"{(DEFAULT_COEFFICIENTS['view_coef']-1)*100:+.0f}%",
                f"{(DEFAULT_COEFFICIENTS['finish_coef']-1)*100:+.0f}%",
            ],
        }
    )
    st.dataframe(coef_data, use_container_width=True)

    # Sales velocity targets
    st.subheader(f"🎯 {t('target_velocity')}")

    velocity_data = pd.DataFrame(
        {
            t('unit_type'): list(TARGET_VELOCITY.keys()),
            t('target_units_month'): list(TARGET_VELOCITY.values()),
        }
    )
    st.dataframe(velocity_data, use_container_width=True)

    # Positioning strategies
    st.subheader(f"🎲 {t('positioning_strategies')}")

    strategy_data = pd.DataFrame(
        {
            t('strategy'): list(POSITIONING_STRATEGIES.keys()),
            t('adjustment'): [
                f"{v:+.1%}" for v in POSITIONING_STRATEGIES.values()
            ],
        }
    )
    st.dataframe(strategy_data, use_container_width=True)

    st.info(t('params_editing_note'))


def show_generate_reports_page():
    """Report generation page"""
    st.header(f"📊 {t('reports_title')}")

    st.markdown(t('reports_desc'))

    # Report type selection
    report_type = st.selectbox(
        t('select_report_type'),
        [
            t('monthly_report'),
            t('crm_price_update'),
            t('project_template'),
        ],
    )

    if report_type == t('monthly_report'):
        st.subheader(f"📅 {t('monthly_report')}")

        col1, col2 = st.columns(2)
        with col1:
            month = st.date_input(t('select_month'), value=pd.Timestamp.now())

        if st.button(t('generate_report'), type="primary"):
            with st.spinner(f"{t('generating_report')}"):
                try:
                    exporter = DataExporter(user_role="analyst")
                    output_path = exporter.export_monthly_analysis_report(
                        month=pd.Timestamp(month)
                    )

                    st.success(f"✅ {t('report_generated')}")
                    st.info(f"📁 {t('saved_to')}: {output_path}")

                    with open(output_path, "rb") as f:
                        st.download_button(
                            label=f"⬇️ {t('download_report')}",
                            data=f,
                            file_name=output_path.name,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )

                except Exception as e:
                    st.error(f"❌ {t('report_failed')}: {e}")

    elif report_type == t('crm_price_update'):
        st.subheader(f"💰 {t('crm_price_update')}")

        st.info(t('crm_export_desc'))

        status_filter = st.selectbox(
            t('price_status'), ["approved", "applied"], index=0
        )

        if st.button(t('generate_export'), type="primary"):
            with st.spinner(f"{t('generating_export')}"):
                try:
                    exporter = DataExporter(user_role="analyst")
                    output_path = exporter.export_crm_price_update(
                        status_filter=status_filter
                    )

                    if output_path:
                        st.success(f"✅ {t('export_generated')}")
                        st.info(f"📁 {t('saved_to')}: {output_path}")

                        with open(output_path, "rb") as f:
                            st.download_button(
                                label=f"⬇️ {t('download_export')}",
                                data=f,
                                file_name=output_path.name,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            )
                    else:
                        st.warning(f"⚠️ {t('no_approved_prices')}")

                except Exception as e:
                    st.error(f"❌ {t('export_failed')}: {e}")

    elif report_type == t('project_template'):
        st.subheader(f"📋 {t('project_template')}")

        # Get list of projects
        session = get_session()
        projects = session.query(Project.project_id, Project.project_name).all()
        project_options = {f"{p.project_id} - {p.project_name}": p.project_id for p in projects}
        session.close()

        selected_project = st.selectbox(
            t('select_project'), options=list(project_options.keys())
        )

        if st.button(t('generate_template'), type="primary"):
            with st.spinner(f"{t('generating_template')}"):
                try:
                    project_id = project_options[selected_project]
                    exporter = DataExporter(user_role="analyst")
                    output_path = exporter.export_project_pricing_template(project_id)

                    st.success(f"✅ {t('template_generated')}")
                    st.info(f"📁 {t('saved_to')}: {output_path}")

                    with open(output_path, "rb") as f:
                        st.download_button(
                            label=f"⬇️ {t('download_template')}",
                            data=f,
                            file_name=output_path.name,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )

                except Exception as e:
                    st.error(f"❌ {t('template_failed')}: {e}")
