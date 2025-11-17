"""
Executive UI Pages
Strategic dashboards, profit analysis, and high-level KPIs
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

from src.database.models import (
    get_session,
    Project,
    BasePricing,
    DynamicSignal,
    PriceRecommendation,
    CompetitorData,
)
from src.models.demand_analysis import DemandCurveAnalyzer
from config.translations import t


def show_executive_dashboard_page():
    """High-level executive dashboard"""
    st.header(f"📊 {t('exec_dash_title')}")

    session = get_session()

    # Date range selector
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(t('strategic_overview_label'))
    with col2:
        period = st.selectbox(t('period'), [t('last_30_days'), t('last_90_days'), t('ytd'), t('all_time')])

    # Calculate date range
    if period == t('last_30_days'):
        start_date = datetime.now() - timedelta(days=30)
    elif period == t('last_90_days'):
        start_date = datetime.now() - timedelta(days=90)
    elif period == t('ytd'):
        start_date = datetime(datetime.now().year, 1, 1)
    else:
        start_date = datetime(2020, 1, 1)

    # ========================================================================
    # KEY METRICS
    # ========================================================================
    st.subheader(f"🎯 {t('kpis')}")

    col1, col2, col3, col4 = st.columns(4)

    # Total projects
    with col1:
        active_projects = session.query(Project).filter(
            Project.status.in_(["Строится", "Планируется"])
        ).count()
        st.metric(t('active_projects'), active_projects)

    # Total sales
    with col2:
        sales_query = session.query(DynamicSignal).filter(
            DynamicSignal.date >= start_date.date()
        )
        total_units = sum([s.units_sold for s in sales_query if s.units_sold])
        st.metric(t('units_sold'), f"{int(total_units)}")

    # Total revenue
    with col3:
        total_revenue = sum([s.total_sales_value for s in sales_query if s.total_sales_value])
        st.metric(t('revenue'), f"{total_revenue/1e9:.2f}B тг")

    # Pending recommendations
    with col4:
        pending_recs = session.query(PriceRecommendation).filter(
            PriceRecommendation.status == "pending"
        ).count()
        st.metric(t('pending_approvals_label'), pending_recs)

    # ========================================================================
    # REVENUE TREND
    # ========================================================================
    st.subheader(f"💰 {t('revenue_trend')}")

    revenue_query = (
        session.query(
            DynamicSignal.date,
            DynamicSignal.total_sales_value
        )
        .filter(DynamicSignal.date >= start_date.date())
        .order_by(DynamicSignal.date)
    )
    revenue_df = pd.read_sql(revenue_query.statement, session.bind)

    if not revenue_df.empty:
        daily_revenue = revenue_df.groupby("date")["total_sales_value"].sum().reset_index()
        daily_revenue["cumulative"] = daily_revenue["total_sales_value"].cumsum()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_revenue["date"],
            y=daily_revenue["cumulative"] / 1e6,
            mode='lines+markers',
            name=t('revenue'),
            line=dict(color='#2E86AB', width=3)
        ))
        fig.update_layout(
            xaxis_title=t('date'),
            yaxis_title=f"{t('revenue')} ({t('m_tg')})",
            height=350,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True, key="revenue_trend_chart")
    else:
        st.info(t('no_revenue_data'))

    # ========================================================================
    # PROJECT PORTFOLIO
    # ========================================================================
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader(f"🏢 {t('project_portfolio')}")

        # Get sales by project
        project_sales = (
            session.query(
                DynamicSignal.project_id,
                Project.project_name,
                Project.housing_class
            )
            .join(Project, DynamicSignal.project_id == Project.project_id)
            .filter(DynamicSignal.date >= start_date.date())
        )
        project_df = pd.read_sql(project_sales.statement, session.bind)

        if not project_df.empty:
            class_dist = project_df.groupby("housing_class").size().reset_index(name="count")

            fig = px.pie(
                class_dist,
                values="count",
                names="housing_class",
                title=t('projects_by_class'),
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            st.plotly_chart(fig, use_container_width=True, key="project_portfolio_chart")

    with col_right:
        st.subheader(f"📈 {t('sales_velocity_label')}")

        velocity_query = (
            session.query(
                DynamicSignal.project_id,
                Project.project_name,
                DynamicSignal.sales_velocity
            )
            .join(Project, DynamicSignal.project_id == Project.project_id)
            .filter(DynamicSignal.date >= start_date.date())
            .filter(DynamicSignal.sales_velocity > 0)
        )
        velocity_df = pd.read_sql(velocity_query.statement, session.bind)

        if not velocity_df.empty:
            avg_velocity = velocity_df.groupby("project_name")["sales_velocity"].mean().reset_index()
            avg_velocity = avg_velocity.sort_values("sales_velocity", ascending=False).head(10)

            fig = px.bar(
                avg_velocity,
                x="sales_velocity",
                y="project_name",
                orientation="h",
                title=t('top_projects_velocity'),
                labels={"sales_velocity": t('avg_velocity'), "project_name": t('project')}
            )
            st.plotly_chart(fig, use_container_width=True, key="sales_velocity_chart")

    # ========================================================================
    # PRICING ACTIONS
    # ========================================================================
    st.subheader(f"💡 {t('recommended_actions')}")

    # Get recommendations needing attention
    high_confidence_pending = (
        session.query(PriceRecommendation)
        .filter(PriceRecommendation.status == "pending")
        .filter(PriceRecommendation.confidence_score >= 0.7)
        .count()
    )

    if high_confidence_pending > 0:
        st.warning(f"⚠️ {high_confidence_pending} {t('high_confidence_pending')}")

    # Projects with low sales velocity
    low_velocity = (
        session.query(DynamicSignal)
        .filter(DynamicSignal.date >= (datetime.now() - timedelta(days=30)).date())
        .filter(DynamicSignal.sales_velocity < 50)
        .count()
    )

    if low_velocity > 0:
        st.info(f"ℹ️ {low_velocity} {t('low_velocity_warning')}")

    session.close()


def show_profit_analysis_page():
    """Profit and margin analysis"""
    st.header(f"💼 {t('profit_analysis_title')}")

    session = get_session()

    st.markdown(t('profit_analysis_desc'))

    # Project selector
    projects = session.query(Project.project_id, Project.project_name, Project.housing_class).all()

    if not projects:
        st.warning(t('no_projects'))
        session.close()
        return

    # ========================================================================
    # MARGIN ANALYSIS
    # ========================================================================
    st.subheader(f"📊 {t('margin_analysis')}")

    margin_data = []
    for project in projects:
        # Get pricing data
        pricing = session.query(BasePricing).filter_by(project_id=project.project_id).all()

        if pricing:
            avg_margin = sum([p.developer_margin_pct for p in pricing]) / len(pricing)
            avg_price = sum([p.base_price_m2 for p in pricing]) / len(pricing)
            avg_cost = sum([p.construction_cost_m2 for p in pricing]) / len(pricing)

            margin_data.append({
                t('project'): project.project_name,
                t('class'): project.housing_class,
                t('avg_margin_label'): avg_margin,
                t('avg_price_label'): avg_price,
                t('avg_cost'): avg_cost,
                t('profit_m2'): avg_price - avg_cost,
            })

    if margin_data:
        margin_df = pd.DataFrame(margin_data)

        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(t('avg_margin_label'), f"{margin_df[t('avg_margin_label')].mean():.1%}")
        with col2:
            st.metric(t('profit_m2'), f"{margin_df[t('profit_m2')].mean():,.0f} тг")
        with col3:
            highest_margin = margin_df.loc[margin_df[t('avg_margin_label')].idxmax(), t('project')]
            st.metric(t('highest_margin'), highest_margin)

        # Margin comparison chart
        fig = px.bar(
            margin_df.sort_values(t('avg_margin_label'), ascending=False),
            x=t('project'),
            y=t('avg_margin_label'),
            color=t('class'),
            title=t('margin_by_project'),
            labels={t('avg_margin_label'): t('margin_pct')}
        )
        fig.update_layout(xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig, use_container_width=True, key="margin_comparison_chart")

        # Detailed table
        st.dataframe(margin_df, use_container_width=True)

    # ========================================================================
    # PROFITABILITY BY UNIT TYPE
    # ========================================================================
    st.subheader(f"📋 {t('profitability_by_type')}")

    unit_profit_data = []
    unit_types = ["1-комнатная", "2-комнатная", "3-комнатная", "4-комнатная"]

    for unit_type in unit_types:
        pricing = session.query(BasePricing).filter_by(unit_type=unit_type).all()

        if pricing:
            total_units = len(pricing)
            avg_price = sum([p.base_price_m2 for p in pricing]) / total_units
            avg_cost = sum([p.construction_cost_m2 for p in pricing]) / total_units
            avg_margin = sum([p.developer_margin_pct for p in pricing]) / total_units

            unit_profit_data.append({
                t('unit_type'): unit_type,
                t('count'): total_units,
                t('avg_price_label'): avg_price,
                t('avg_cost'): avg_cost,
                t('margin_percent'): avg_margin,
                t('profit_m2'): avg_price - avg_cost,
            })

    if unit_profit_data:
        unit_df = pd.DataFrame(unit_profit_data)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name=t('cost'),
            x=unit_df[t('unit_type')],
            y=unit_df[t('avg_cost')],
            marker_color='lightcoral'
        ))
        fig.add_trace(go.Bar(
            name=t('profit'),
            x=unit_df[t('unit_type')],
            y=unit_df[t('profit_m2')],
            marker_color='lightgreen'
        ))
        fig.update_layout(
            barmode='stack',
            title=t('cost_vs_profit'),
            xaxis_title=t('unit_type'),
            yaxis_title=t('tg_m2'),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True, key="unit_type_profitability_chart")

        st.dataframe(unit_df, use_container_width=True)

    session.close()


def show_strategic_pricing_page():
    """Strategic pricing decisions"""
    st.header(f"🎯 {t('strategic_pricing_title')}")

    st.markdown(t('strategic_pricing_desc'))

    session = get_session()

    # ========================================================================
    # DEMAND CURVE EXPLORER
    # ========================================================================
    st.subheader(f"📈 {t('demand_curve_explorer')}")

    st.markdown(t('demand_curve_desc'))

    # Project and unit type selector
    projects = session.query(Project.project_id, Project.project_name).all()
    project_options = {f"{p.project_id} - {p.project_name}": p.project_id for p in projects}

    col_proj, col_unit = st.columns(2)

    with col_proj:
        selected_project_display = st.selectbox(t('select_project'), list(project_options.keys()))
        selected_project_id = project_options[selected_project_display]

    with col_unit:
        # Get unit types for selected project
        unit_types_query = (
            session.query(BasePricing.unit_type)
            .filter_by(project_id=selected_project_id)
            .distinct()
            .all()
        )
        unit_types = [ut[0] for ut in unit_types_query]
        selected_unit = st.selectbox(t('unit_type'), unit_types) if unit_types else None

    if selected_unit:
        analyzer = DemandCurveAnalyzer()

        # Create tabs for different analyses
        tab_demand, tab_revenue, tab_stats = st.tabs([
            f"📉 {t('demand_curve_tab')}",
            f"💰 {t('revenue_opt_tab')}",
            f"📊 {t('statistics_tab')}"
        ])

        with tab_demand:
            project = session.query(Project).filter_by(project_id=selected_project_id).first()
            fig = analyzer.create_demand_curve_plot(
                selected_project_id,
                selected_unit,
                project.project_name if project else None
            )
            st.plotly_chart(fig, use_container_width=True, key="demand_curve_tab_chart")

            st.info(t('demand_chart_legend'))

        with tab_revenue:
            revenue_fig = analyzer.create_price_elasticity_plot(selected_project_id, selected_unit)
            st.plotly_chart(revenue_fig, use_container_width=True, key="revenue_optimization_tab_chart")

            st.info(t('revenue_opt_legend'))

        with tab_stats:
            stats = analyzer.get_demand_summary_stats(selected_project_id, selected_unit)

            if stats["status"] == "success":
                # Data quality metrics
                st.markdown(f"### 📊 {t('model_quality')}")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        t('confidence_level'),
                        stats["data_quality"]["confidence"],
                        "✓" if stats["data_quality"]["reliable"] else f"⚠ {t('use_with_caution')}"
                    )

                with col2:
                    st.metric(
                        t('r2_score'),
                        f"{stats['data_quality']['r2']:.3f}",
                        t('good') if stats['data_quality']['r2'] >= 0.7 else t('fair') if stats['data_quality']['r2'] >= 0.5 else t('weak')
                    )

                with col3:
                    st.metric(t('data_points'), stats["data_quality"]["points"])

                # Pricing insights
                st.markdown(f"### 💰 {t('pricing_insights')}")
                col1, col2, col3 = st.columns(3)

                with col1:
                    if stats["pricing"]["current_price"]:
                        st.metric(
                            t('current_price_label'),
                            f"{stats['pricing']['current_price']:,.0f} {t('tg_m2')}"
                        )

                with col2:
                    if stats["pricing"]["optimal_price_velocity"]:
                        st.metric(
                            t('optimal_price_velocity'),
                            f"{stats['pricing']['optimal_price_velocity']:,.0f} {t('tg_m2')}"
                        )

                with col3:
                    st.metric(
                        t('price_range'),
                        f"{stats['pricing']['price_range_min']:,.0f} - {stats['pricing']['price_range_max']:,.0f}"
                    )

                # Demand characteristics
                st.markdown(f"### 📉 {t('demand_characteristics')}")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        t('avg_units_sold'),
                        f"{stats['demand']['avg_units_sold']:.1f}"
                    )

                with col2:
                    st.metric(
                        t('target_velocity_label'),
                        f"{stats['demand']['target_velocity']} {t('units_month')}"
                    )

                with col3:
                    if stats["demand"]["elasticity"]:
                        elasticity = stats["demand"]["elasticity"]
                        elasticity_type = t('elastic') if abs(elasticity) > 1 else t('inelastic')
                        st.metric(
                            t('price_elasticity'),
                            f"{elasticity:.2f}",
                            elasticity_type
                        )

                # Interpretation
                st.markdown(f"### 💡 {t('interpretation')}")

                if stats["demand"]["elasticity"]:
                    elasticity = stats["demand"]["elasticity"]
                    if abs(elasticity) > 1:
                        elastic_msg = t('elastic_desc').format(elasticity=elasticity)
                        st.success(f"""
                        **{t('elastic_demand')}** (|{elasticity:.2f}| > 1): Спрос очень чувствителен к изменениям цен.
                        - Небольшое снижение цен может значительно увеличить продажи
                        - Рассмотрите агрессивную стратегию ценообразования для доли рынка
                        - Рост объема может компенсировать снижение маржи
                        """)
                    else:
                        inelastic_msg = t('inelastic_desc').format(elasticity=elasticity)
                        st.info(f"""
                        **{t('inelastic_demand')}** (|{elasticity:.2f}| < 1): Спрос относительно нечувствителен к цене.
                        - Изменения цен имеют ограниченное влияние на объем продаж
                        - Рассмотрите премиальную стратегию ценообразования
                        - Фокус на дифференциации ценности, а не на ценовой конкуренции
                        """)

                if not stats["data_quality"]["reliable"]:
                    st.warning(t('low_confidence_warning'))

    # ========================================================================
    # PRICING STRATEGY SCENARIOS
    # ========================================================================
    st.subheader(f"🎲 {t('pricing_scenarios')}")

    strategy_options = {
        t('aggressive_growth'): {
            "description": t('aggressive_desc'),
            "adjustment": -0.05,
            "color": "red"
        },
        t('market_neutral'): {
            "description": t('neutral_desc'),
            "adjustment": 0.00,
            "color": "blue"
        },
        t('premium_positioning'): {
            "description": t('premium_desc'),
            "adjustment": 0.05,
            "color": "green"
        },
        t('luxury_tier'): {
            "description": t('luxury_desc'),
            "adjustment": 0.10,
            "color": "purple"
        }
    }

    selected_strategy = st.select_slider(
        t('select_strategy'),
        options=list(strategy_options.keys()),
        value=t('market_neutral')
    )

    strategy_info = strategy_options[selected_strategy]
    st.info(f"**{selected_strategy}**: {strategy_info['description']}")

    # Simulate impact
    st.subheader(f"📊 {t('projected_impact')}")

    # Get current pricing
    all_pricing = session.query(BasePricing).all()

    if all_pricing:
        current_avg_price = sum([p.base_price_m2 for p in all_pricing]) / len(all_pricing)
        market_avg = sum([p.market_price_avg for p in all_pricing if p.market_price_avg]) / len([p for p in all_pricing if p.market_price_avg])

        new_price = market_avg * (1 + strategy_info['adjustment'])
        price_change = new_price - current_avg_price
        price_change_pct = (price_change / current_avg_price) * 100

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(t('current_avg'), f"{current_avg_price:,.0f} {t('tg_m2')}")
        with col2:
            st.metric(t('market_avg'), f"{market_avg:,.0f} {t('tg_m2')}")
        with col3:
            st.metric(t('new_price'), f"{new_price:,.0f} {t('tg_m2')}", f"{price_change_pct:+.1f}%")
        with col4:
            # Rough revenue impact estimate
            total_area = sum([p.area_m2 for p in all_pricing])
            revenue_impact = price_change * total_area / 1e6
            st.metric(t('revenue_impact'), f"{revenue_impact:+,.0f}{t('m_tg')}")

    # ========================================================================
    # MARKET POSITIONING
    # ========================================================================
    st.subheader(f"📍 {t('market_positioning_matrix')}")

    # Get competitor data
    competitors = session.query(CompetitorData).filter(
        CompetitorData.avg_price_m2 > 0
    ).all()

    # Get our pricing
    our_pricing = session.query(
        BasePricing.unit_type,
        BasePricing.base_price_m2,
        Project.housing_class
    ).join(Project, BasePricing.project_id == Project.project_id).all()

    if competitors and our_pricing:
        # Create positioning matrix
        positioning_data = []

        for unit_type in ["1-комнатная", "2-комнатная", "3-комнатная", "4-комнатная"]:
            our_prices = [p.base_price_m2 for p in our_pricing if p.unit_type == unit_type]
            comp_prices = [c.avg_price_m2 for c in competitors if c.unit_type == unit_type]

            if our_prices and comp_prices:
                our_avg = sum(our_prices) / len(our_prices)
                comp_avg = sum(comp_prices) / len(comp_prices)

                positioning_data.append({
                    t('unit_type'): unit_type,
                    t('our_price_label'): our_avg,
                    t('market_price_label'): comp_avg,
                    t('position'): "Above" if our_avg > comp_avg else "Below",
                    "Difference": ((our_avg / comp_avg) - 1) * 100
                })

        if positioning_data:
            pos_df = pd.DataFrame(positioning_data)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pos_df[t('market_price_label')],
                y=pos_df[t('our_price_label')],
                mode='markers+text',
                text=pos_df[t('unit_type')],
                textposition="top center",
                marker=dict(size=15, color='blue'),
                name=t('our_projects')
            ))

            # Add diagonal line (price parity)
            min_price = min(pos_df[t('market_price_label')].min(), pos_df[t('our_price_label')].min())
            max_price = max(pos_df[t('market_price_label')].max(), pos_df[t('our_price_label')].max())
            fig.add_trace(go.Scatter(
                x=[min_price, max_price],
                y=[min_price, max_price],
                mode='lines',
                line=dict(dash='dash', color='red'),
                name=t('market_parity')
            ))

            fig.update_layout(
                title=t('price_positioning'),
                xaxis_title=f"{t('market_avg_price')} ({t('tg_m2')})",
                yaxis_title=f"{t('our_price')} ({t('tg_m2')})",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True, key="market_positioning_matrix_chart")

            st.dataframe(pos_df, use_container_width=True)

    session.close()


def show_market_overview_page():
    """Market overview and trends"""
    st.header(f"📈 {t('market_overview_title')}")

    session = get_session()

    st.markdown(t('market_overview_desc'))

    # ========================================================================
    # MARKET SUMMARY
    # ========================================================================
    st.subheader(f"🌍 {t('market_summary')}")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_competitors = session.query(CompetitorData.competitor_name).distinct().count()
        st.metric(t('competitors_tracked'), total_competitors)

    with col2:
        total_complexes = session.query(CompetitorData.complex_name).distinct().count()
        st.metric(t('competitor_complexes'), total_complexes)

    with col3:
        our_projects = session.query(Project).count()
        st.metric(t('our_projects_label'), our_projects)

    with col4:
        market_share_est = (our_projects / (our_projects + total_complexes)) * 100
        st.metric(t('est_market_share'), f"{market_share_est:.1f}%")

    # ========================================================================
    # PRICE TRENDS
    # ========================================================================
    st.subheader(f"📊 {t('market_price_trends')}")

    # Get competitor prices over time
    comp_trends = (
        session.query(
            CompetitorData.date,
            CompetitorData.unit_type,
            CompetitorData.avg_price_m2
        )
        .filter(CompetitorData.avg_price_m2 > 0)
        .order_by(CompetitorData.date)
    )
    comp_df = pd.read_sql(comp_trends.statement, session.bind)

    if not comp_df.empty:
        # Calculate monthly average
        comp_df['month'] = pd.to_datetime(comp_df['date']).dt.to_period('M').astype(str)
        monthly_avg = comp_df.groupby(['month', 'unit_type'])['avg_price_m2'].mean().reset_index()

        fig = px.line(
            monthly_avg,
            x='month',
            y='avg_price_m2',
            color='unit_type',
            title=t('price_trends_by_type'),
            labels={'avg_price_m2': f"{t('avg_price_m2')} ({t('tg_m2')})", 'month': t('month')}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True, key="market_price_trends_chart")

    # ========================================================================
    # COMPETITIVE LANDSCAPE
    # ========================================================================
    st.subheader(f"🏢 {t('competitive_landscape')}")

    # Top competitors by complexes
    top_competitors = (
        session.query(
            CompetitorData.competitor_name,
            CompetitorData.complex_name,
            CompetitorData.avg_price_m2,
            CompetitorData.housing_class
        )
        .filter(CompetitorData.avg_price_m2 > 0)
        .order_by(CompetitorData.avg_price_m2.desc())
        .limit(20)
    )
    comp_list_df = pd.read_sql(top_competitors.statement, session.bind)

    if not comp_list_df.empty:
        st.dataframe(comp_list_df, use_container_width=True)

    session.close()
