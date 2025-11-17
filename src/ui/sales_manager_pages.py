"""
Sales Manager UI Pages
Price recommendations, approval workflow, and sales performance tracking
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.translations import t
from src.database.models import (
    get_session,
    Project,
    BasePricing,
    PriceRecommendation,
    DynamicSignal,
    PriceHistory,
)
from src.models.pricing_engine import PricingEngine
from src.models.demand_analysis import DemandCurveAnalyzer


def show_price_recommendations_page():
    """Price recommendations dashboard"""
    st.header(f"💰 {t('price_recs_title')}")

    session = get_session()

    # Action buttons at top
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader(t('ai_generated_recs'))
    with col2:
        if st.button(f"🔄 {t('generate_new_recs')}", use_container_width=True):
            st.session_state.show_generation_modal = True
    with col3:
        strategy_filter = st.selectbox(t('pricing_strategy'), ["All", "balanced", "aggressive", "premium"])

    # Show generation modal
    if st.session_state.get("show_generation_modal", False):
        with st.expander(f"🎯 {t('generate_new_recs')}", expanded=True):
            projects = session.query(Project.project_id, Project.project_name).all()
            project_options = {f"{p.project_id} - {p.project_name}": p.project_id for p in projects}

            selected_project = st.selectbox(t('select_project'), list(project_options.keys()))
            gen_strategy = st.selectbox(t('pricing_strategy'), ["balanced", "aggressive", "neutral", "premium"])

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button(t('generate'), type="primary", use_container_width=True):
                    with st.spinner(f"{t('generating_recs')}"):
                        try:
                            project_id = project_options[selected_project]
                            engine = PricingEngine()
                            recommendations = engine.generate_recommendations_for_project(
                                project_id, gen_strategy
                            )
                            saved = engine.save_recommendations_to_db(recommendations, "sales_manager")
                            st.success(f"✅ {saved} {t('recs_saved')}")
                            st.session_state.show_generation_modal = False
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ {t('rec_failed')}: {e}")
            with col_b:
                if st.button(t('cancel'), use_container_width=True):
                    st.session_state.show_generation_modal = False
                    st.rerun()

    # Get pending recommendations
    query = (
        session.query(
            PriceRecommendation.id,
            PriceRecommendation.project_id,
            Project.project_name,
            PriceRecommendation.unit_type,
            PriceRecommendation.current_price_m2,
            PriceRecommendation.recommended_price_m2,
            PriceRecommendation.price_change_pct,
            PriceRecommendation.price_change_amount,
            PriceRecommendation.confidence_score,
            PriceRecommendation.status,
            PriceRecommendation.recommendation_date,
        )
        .join(Project, PriceRecommendation.project_id == Project.project_id)
    )

    # Apply filters
    if strategy_filter != "All":
        # Note: strategy not stored in DB currently, would need to add
        pass

    recommendations = query.order_by(PriceRecommendation.recommendation_date.desc()).all()

    if not recommendations:
        st.info(f"📭 {t('no_recs_available')}")
        session.close()
        return

    # Group by status
    pending = [r for r in recommendations if r.status == "pending"]
    approved = [r for r in recommendations if r.status == "approved"]
    rejected = [r for r in recommendations if r.status == "rejected"]

    # Status tabs
    tab1, tab2, tab3 = st.tabs([
        f"⏳ {t('pending_recs')} ({len(pending)})",
        f"✅ {t('approved_recs')} ({len(approved)})",
        f"❌ {t('rejected_recs')} ({len(rejected)})"
    ])

    with tab1:
        if pending:
            _display_recommendations_table(pending, session, show_actions=True)
        else:
            st.info(t('no_pending'))

    with tab2:
        if approved:
            _display_recommendations_table(approved, session, show_actions=False)
        else:
            st.info(t('no_approved'))

    with tab3:
        if rejected:
            _display_recommendations_table(rejected, session, show_actions=False)
        else:
            st.info(t('no_rejected'))

    session.close()


def _display_recommendations_table(recommendations, session, show_actions=True):
    """Display recommendations in a table with optional action buttons"""

    for rec in recommendations:
        rec_id, project_id, project_name, unit_type, current, recommended, change_pct, change_amt, confidence, status, rec_date = rec

        # Create expandable card for each recommendation
        with st.expander(f"**{project_name}** - {unit_type} | {change_pct:+.1%} ({change_amt:+,.0f} {t('tg_m2')})"):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(t('current_price'), f"{current:,.0f} {t('tg_m2')}")
            with col2:
                st.metric(t('recommended'), f"{recommended:,.0f} {t('tg_m2')}", f"{change_pct:+.1%}")
            with col3:
                st.metric(t('confidence'), f"{confidence:.0%}")
            with col4:
                st.metric(t('status'), status.upper())

            # Get full recommendation details
            full_rec = session.query(PriceRecommendation).filter_by(id=rec_id).first()

            if full_rec.rationale_text:
                st.markdown(f"**{t('rationale')}:**")
                st.info(full_rec.rationale_text)

            # Show demand curve visualization
            with st.expander(f"📈 {t('view_demand_curve')}", expanded=False):
                analyzer = DemandCurveAnalyzer()

                # Get project name
                project = session.query(Project).filter_by(project_id=project_id).first()
                project_name = project.project_name if project else project_id

                # Create demand curve plot
                fig = analyzer.create_demand_curve_plot(project_id, unit_type, project_name)
                st.plotly_chart(fig, use_container_width=True, key=f"demand_curve_{rec_id}")

                # Show summary stats
                stats = analyzer.get_demand_summary_stats(project_id, unit_type)
                if stats["status"] == "success":
                    col_stat1, col_stat2, col_stat3 = st.columns(3)

                    with col_stat1:
                        st.metric(
                            t('data_quality'),
                            stats["data_quality"]["confidence"],
                            f"R² = {stats['data_quality']['r2']:.3f}"
                        )

                    with col_stat2:
                        if stats["demand"]["elasticity"]:
                            elasticity = stats["demand"]["elasticity"]
                            elasticity_type = t('elastic') if abs(elasticity) > 1 else t('inelastic')
                            st.metric(
                                t('price_elasticity'),
                                f"{elasticity:.2f}",
                                elasticity_type
                            )

                    with col_stat3:
                        st.metric(
                            t('avg_sales'),
                            f"{stats['demand']['avg_units_sold']:.1f} {t('units')}",
                            f"{t('target')}: {stats['demand']['target_velocity']}"
                        )

                # Revenue optimization curve
                st.markdown(f"**{t('revenue_optimization')}:**")
                revenue_fig = analyzer.create_price_elasticity_plot(project_id, unit_type)
                st.plotly_chart(revenue_fig, use_container_width=True, key=f"revenue_curve_{rec_id}")

            # Action buttons for pending items
            if show_actions and status == "pending":
                st.markdown("---")
                col_a, col_b, col_c, col_d = st.columns([1, 1, 1, 2])

                with col_a:
                    if st.button(f"✅ {t('approve')}", key=f"approve_{rec_id}", use_container_width=True):
                        _approve_recommendation(session, rec_id, recommended)
                        st.success(f"{t('approved')}")
                        st.rerun()

                with col_b:
                    if st.button(f"❌ {t('reject')}", key=f"reject_{rec_id}", use_container_width=True):
                        _reject_recommendation(session, rec_id)
                        st.success(f"{t('rejected_msg')}")
                        st.rerun()

                with col_c:
                    if st.button(f"✏️ {t('override')}", key=f"override_{rec_id}", use_container_width=True):
                        st.session_state[f"override_modal_{rec_id}"] = True

                # Override modal
                if st.session_state.get(f"override_modal_{rec_id}", False):
                    with col_d:
                        override_price = st.number_input(
                            t('override_price'),
                            value=float(recommended),
                            key=f"override_input_{rec_id}"
                        )
                        col_x, col_y = st.columns(2)
                        with col_x:
                            if st.button(t('save'), key=f"save_override_{rec_id}"):
                                _approve_recommendation(session, rec_id, override_price, is_override=True)
                                st.session_state[f"override_modal_{rec_id}"] = False
                                st.success(f"{t('override_saved')}")
                                st.rerun()
                        with col_y:
                            if st.button(t('cancel'), key=f"cancel_override_{rec_id}"):
                                st.session_state[f"override_modal_{rec_id}"] = False
                                st.rerun()


def _approve_recommendation(session, rec_id, approved_price, is_override=False):
    """Approve a price recommendation"""
    rec = session.query(PriceRecommendation).filter_by(id=rec_id).first()
    rec.status = "approved"
    rec.approved_price_m2 = approved_price
    rec.approved_by = "sales_manager"  # Would use actual user in production
    rec.approved_at = datetime.utcnow()
    if is_override:
        rec.override_reason = "Manual override by sales manager"
    session.commit()


def _reject_recommendation(session, rec_id):
    """Reject a price recommendation"""
    rec = session.query(PriceRecommendation).filter_by(id=rec_id).first()
    rec.status = "rejected"
    rec.rejection_reason = "Rejected by sales manager"
    rec.approved_by = "sales_manager"
    rec.approved_at = datetime.utcnow()
    session.commit()


def show_unit_comparison_page():
    """Unit comparison tool"""
    st.header(f"📋 {t('unit_comparison_title')}")

    session = get_session()

    # Project selector
    projects = session.query(Project.project_id, Project.project_name).all()
    project_options = {f"{p.project_id} - {p.project_name}": p.project_id for p in projects}

    selected_project = st.selectbox(t('select_project'), list(project_options.keys()))
    project_id = project_options[selected_project]

    # Get all unit types for selected project
    units = (
        session.query(
            BasePricing.unit_type,
            BasePricing.area_m2,
            BasePricing.base_price_m2,
            BasePricing.base_unit_price,
            BasePricing.market_price_avg,
            BasePricing.developer_margin_pct,
        )
        .filter_by(project_id=project_id)
        .all()
    )

    if not units:
        st.warning(t('no_pricing_data'))
        session.close()
        return

    # Convert to DataFrame
    df = pd.DataFrame(
        units,
        columns=[t('unit_type'), t('area'), t('price_per_m2'), t('total_price'), t('market_avg'), t('margin_percent')]
    )

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(t('unit_types'), len(df))
    with col2:
        st.metric(t('avg_price_m2'), f"{df[t('price_per_m2')].mean():,.0f} {t('tg_m2')}")
    with col3:
        st.metric(t('avg_margin'), f"{df[t('margin_percent')].mean():.1%}")
    with col4:
        price_vs_market = ((df[t('price_per_m2')].mean() / df[t('market_avg')].mean()) - 1) * 100
        st.metric(t('vs_market'), f"{price_vs_market:+.1f}%")

    # Comparison table
    st.subheader(f"📊 {t('unit_type_comparison')}")
    st.dataframe(df, use_container_width=True)

    # Visualization
    st.subheader(f"📈 {t('price_comparison_chart')}")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name=t('current_price_label'),
        x=df[t('unit_type')],
        y=df[t('price_per_m2')],
        marker_color='lightblue'
    ))

    fig.add_trace(go.Bar(
        name=t('market_average'),
        x=df[t('unit_type')],
        y=df[t('market_avg')],
        marker_color='orange'
    ))

    fig.update_layout(
        barmode='group',
        xaxis_title=t('unit_type'),
        yaxis_title=t('price_per_m2'),
        height=400
    )

    st.plotly_chart(fig, use_container_width=True, key="unit_comparison_chart")

    session.close()


def show_price_approval_page():
    """Batch price approval interface"""
    st.header(f"✅ {t('price_approval_title')}")

    session = get_session()

    st.markdown(t('price_approval_desc'))

    # Get all pending recommendations
    pending_recs = (
        session.query(
            PriceRecommendation.id,
            PriceRecommendation.project_id,
            Project.project_name,
            PriceRecommendation.unit_type,
            PriceRecommendation.current_price_m2,
            PriceRecommendation.recommended_price_m2,
            PriceRecommendation.price_change_pct,
            PriceRecommendation.confidence_score,
        )
        .join(Project, PriceRecommendation.project_id == Project.project_id)
        .filter(PriceRecommendation.status == "pending")
        .order_by(Project.project_name, PriceRecommendation.unit_type)
        .all()
    )

    if not pending_recs:
        st.info(f"📭 {t('no_pending_approvals')}")
        session.close()
        return

    # Create approval DataFrame
    approval_data = []
    for rec in pending_recs:
        approval_data.append({
            "Select": False,
            "ID": rec.id,
            t('project'): rec.project_name,
            t('unit_type'): rec.unit_type,
            f"{t('current_price')} ({t('tg_m2')})": f"{rec.current_price_m2:,.0f}",
            f"{t('recommended')} ({t('tg_m2')})": f"{rec.recommended_price_m2:,.0f}",
            t('change'): f"{rec.price_change_pct:+.1%}",
            t('confidence'): f"{rec.confidence_score:.0%}",
        })

    df = pd.DataFrame(approval_data)

    # Batch action buttons
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader(f"{t('pending_approvals')} ({len(df)})")
    with col2:
        if st.button(f"✅ {t('approve_selected')}", type="primary", use_container_width=True):
            selected_ids = df[df["Select"]]["ID"].tolist()
            if selected_ids:
                for rec_id in selected_ids:
                    rec = session.query(PriceRecommendation).filter_by(id=rec_id).first()
                    _approve_recommendation(session, rec_id, rec.recommended_price_m2)
                st.success(f"✅ {len(selected_ids)} {t('approved_count')}")
                st.rerun()
            else:
                st.warning(t('no_items_selected'))
    with col3:
        if st.button(f"❌ {t('reject_selected')}", use_container_width=True):
            selected_ids = df[df["Select"]]["ID"].tolist()
            if selected_ids:
                for rec_id in selected_ids:
                    _reject_recommendation(session, rec_id)
                st.success(f"❌ {len(selected_ids)} {t('rejected_count')}")
                st.rerun()
            else:
                st.warning(t('no_items_selected'))

    # Interactive table with checkboxes
    st.dataframe(df, use_container_width=True)

    # Select all/none
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button(t('select_all')):
            st.session_state.select_all = True
    with col_b:
        if st.button(t('select_none')):
            st.session_state.select_all = False

    session.close()


def show_sales_performance_page():
    """Sales performance tracking dashboard"""
    st.header(f"📈 {t('sales_perf_title')}")

    session = get_session()

    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            t('from_date'),
            value=datetime.now() - timedelta(days=90)
        )
    with col2:
        end_date = st.date_input(t('to_date'), value=datetime.now())

    # Get sales data
    sales_query = (
        session.query(
            DynamicSignal.project_id,
            Project.project_name,
            DynamicSignal.unit_type_name,
            DynamicSignal.date,
            DynamicSignal.units_sold,
            DynamicSignal.m2_sold,
            DynamicSignal.total_sales_value,
            DynamicSignal.avg_price_m2,
            DynamicSignal.sales_velocity,
            DynamicSignal.conversion_rate,
        )
        .join(Project, DynamicSignal.project_id == Project.project_id)
        .filter(DynamicSignal.date >= start_date)
        .filter(DynamicSignal.date <= end_date)
    )

    sales_df = pd.read_sql(sales_query.statement, session.bind)

    if sales_df.empty:
        st.warning(t('no_sales_data'))
        session.close()
        return

    # Key metrics
    st.subheader(f"📊 {t('key_metrics')}")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_units = sales_df["units_sold"].sum()
        st.metric(t('units_sold'), f"{int(total_units)}")

    with col2:
        total_value = sales_df["total_sales_value"].sum()
        st.metric(t('total_sales'), f"{total_value/1e6:.1f}{t('m_tg')}")

    with col3:
        avg_price = sales_df["avg_price_m2"].mean()
        st.metric(t('avg_price_m2'), f"{avg_price:,.0f} {t('tg_m2')}")

    with col4:
        avg_velocity = sales_df["sales_velocity"].mean()
        st.metric(t('sales_velocity_label'), f"{avg_velocity:.1f}")

    # Sales over time
    st.subheader(f"📈 {t('sales_trend')}")

    daily_sales = sales_df.groupby("date")["units_sold"].sum().reset_index()

    fig = px.line(
        daily_sales,
        x="date",
        y="units_sold",
        title=t('sales_trend')
    )
    fig.update_layout(xaxis_title=t('date'), yaxis_title=t('units_sold'))
    st.plotly_chart(fig, use_container_width=True, key="sales_trend_chart")

    # Sales by project
    st.subheader(f"🏢 {t('sales_by_project')}")

    project_sales = (
        sales_df.groupby("project_name")
        .agg({
            "units_sold": "sum",
            "total_sales_value": "sum",
            "avg_price_m2": "mean"
        })
        .reset_index()
    )

    fig = px.bar(
        project_sales,
        x="project_name",
        y="units_sold",
        title=t('sales_by_project'),
        color="units_sold",
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig, use_container_width=True, key="sales_by_project_chart")

    # Sales by unit type
    st.subheader(f"📋 {t('sales_by_unit_type')}")

    unit_sales = (
        sales_df.groupby("unit_type_name")["units_sold"]
        .sum()
        .reset_index()
    )

    fig = px.pie(
        unit_sales,
        values="units_sold",
        names="unit_type_name",
        title=t('distribution_by_type')
    )
    st.plotly_chart(fig, use_container_width=True, key="sales_by_unit_type_chart")

    # Detailed table
    st.subheader(f"📋 {t('detailed_sales_data')}")
    st.dataframe(
        sales_df[["date", "project_name", "unit_type_name", "units_sold", "avg_price_m2", "sales_velocity"]],
        use_container_width=True
    )

    session.close()
