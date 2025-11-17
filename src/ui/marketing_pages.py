"""
Marketing UI Pages
Competitor analysis, market intelligence, and price comparison
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
    CompetitorData,
)
from config.translations import t


def show_competitor_analysis_page():
    """Competitor analysis dashboard"""
    st.header(f"🔍 {t('comp_analysis_title')}")

    session = get_session()

    st.markdown(t('comp_analysis_desc'))

    # ========================================================================
    # FILTERS
    # ========================================================================
    col1, col2, col3 = st.columns(3)

    with col1:
        # Location filter
        locations = [loc[0] for loc in session.query(CompetitorData.location).distinct().all()]
        selected_location = st.selectbox(t('location'), [t('all')] + locations)

    with col2:
        # Housing class filter
        classes = [c[0] for c in session.query(CompetitorData.housing_class).distinct().all() if c[0]]
        selected_class = st.selectbox(t('housing_class'), [t('all')] + classes)

    with col3:
        # Time period
        period = st.selectbox(t('time_period'), [t('last_month'), t('last_3_months'), t('last_6_months'), t('all_time')])

    # Calculate date filter
    if period == t('last_month'):
        date_filter = datetime.now() - timedelta(days=30)
    elif period == t('last_3_months'):
        date_filter = datetime.now() - timedelta(days=90)
    elif period == t('last_6_months'):
        date_filter = datetime.now() - timedelta(days=180)
    else:
        date_filter = datetime(2020, 1, 1)

    # ========================================================================
    # COMPETITOR OVERVIEW
    # ========================================================================
    st.subheader(f"📊 {t('competitor_overview')}")

    # Build query with filters
    query = session.query(CompetitorData).filter(
        CompetitorData.date >= date_filter.date()
    )

    if selected_location != t('all'):
        query = query.filter(CompetitorData.location == selected_location)
    if selected_class != t('all'):
        query = query.filter(CompetitorData.housing_class == selected_class)

    competitors = query.all()

    if not competitors:
        st.warning(t('no_competitor_data'))
        session.close()
        return

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        unique_competitors = len(set([c.competitor_name for c in competitors]))
        st.metric(t('competitors'), unique_competitors)

    with col2:
        unique_complexes = len(set([c.complex_name for c in competitors]))
        st.metric(t('complexes'), unique_complexes)

    with col3:
        avg_price = sum([c.avg_price_m2 for c in competitors if c.avg_price_m2]) / len([c for c in competitors if c.avg_price_m2])
        st.metric(t('avg_market_price'), f"{avg_price:,.0f} {t('tg_m2')}")

    with col4:
        discounted = len([c for c in competitors if c.discount_flag])
        discount_pct = (discounted / len(competitors)) * 100 if competitors else 0
        st.metric(t('with_discounts'), f"{discount_pct:.0f}%")

    # ========================================================================
    # PRICE COMPARISON
    # ========================================================================
    st.subheader(f"💰 {t('price_by_competitor')}")

    # Get average price by competitor
    comp_df = pd.DataFrame([
        {
            t('competitor'): c.competitor_name,
            t('complex'): c.complex_name,
            t('price'): c.avg_price_m2,
            t('class'): c.housing_class,
            t('discount'): t('yes') if c.discount_flag else t('no'),
            t('date'): c.date
        }
        for c in competitors if c.avg_price_m2 > 0
    ])

    if not comp_df.empty:
        # Average by competitor
        comp_avg = comp_df.groupby(t('competitor'))[t('price')].mean().reset_index()
        comp_avg = comp_avg.sort_values(t('price'), ascending=False).head(15)

        fig = px.bar(
            comp_avg,
            x=t('price'),
            y=t('competitor'),
            orientation='h',
            title=t('top_competitors_price'),
            color=t('price'),
            color_continuous_scale="RdYlGn_r"
        )
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True, key="competitor_price_comparison_chart")

        # Detailed table
        st.subheader(f"📋 {t('detailed_competitor_data')}")
        st.dataframe(
            comp_df.sort_values(t('price'), ascending=False),
            use_container_width=True
        )

    session.close()


def show_market_positioning_page():
    """Market positioning analysis"""
    st.header(f"📍 {t('market_pos_title')}")

    session = get_session()

    st.markdown(t('market_pos_desc'))

    # ========================================================================
    # POSITIONING MATRIX
    # ========================================================================
    st.subheader(f"🎯 {t('positioning_matrix')}")

    # Get our pricing by unit type
    our_pricing = (
        session.query(
            Project.project_name,
            Project.housing_class,
            BasePricing.unit_type,
            BasePricing.base_price_m2,
            BasePricing.area_m2
        )
        .join(BasePricing, Project.project_id == BasePricing.project_id)
    )
    our_df = pd.read_sql(our_pricing.statement, session.bind)

    # Get competitor pricing
    comp_pricing = (
        session.query(
            CompetitorData.complex_name,
            CompetitorData.housing_class,
            CompetitorData.unit_type,
            CompetitorData.avg_price_m2
        )
        .filter(CompetitorData.avg_price_m2 > 0)
    )
    comp_df = pd.read_sql(comp_pricing.statement, session.bind)

    if not our_df.empty and not comp_df.empty:
        # Unit type selector
        unit_types = our_df["unit_type"].unique()
        selected_unit = st.selectbox(t('unit_type'), unit_types)

        # Filter by unit type
        our_unit = our_df[our_df["unit_type"] == selected_unit]
        comp_unit = comp_df[comp_df["unit_type"] == selected_unit]

        if not our_unit.empty and not comp_unit.empty:
            # Create scatter plot
            fig = go.Figure()

            # Add our projects
            fig.add_trace(go.Scatter(
                x=our_unit["area_m2"],
                y=our_unit["base_price_m2"],
                mode='markers',
                name=t('our_projects'),
                marker=dict(size=12, color='blue', symbol='star'),
                text=our_unit["project_name"],
                hovertemplate=f'<b>%{{text}}</b><br>{t("area")}: %{{x:.1f}} m²<br>{t("price")}: %{{y:,.0f}} {t("tg_m2")}'
            ))

            # Add competitors
            fig.add_trace(go.Scatter(
                x=[comp_unit["area_m2"].mean()] * len(comp_unit),  # Approximate area
                y=comp_unit["avg_price_m2"],
                mode='markers',
                name=t('competitors'),
                marker=dict(size=8, color='red', symbol='circle'),
                text=comp_unit["complex_name"],
                hovertemplate=f'<b>%{{text}}</b><br>{t("price")}: %{{y:,.0f}} {t("tg_m2")}'
            ))

            fig.update_layout(
                title=f"{t('price_positioning')} - {selected_unit}",
                xaxis_title=t('area_m2'),
                yaxis_title=t('price_per_m2'),
                height=500,
                hovermode='closest'
            )

            st.plotly_chart(fig, use_container_width=True, key="positioning_matrix_scatter_chart")

            # Statistics
            col1, col2, col3 = st.columns(3)

            with col1:
                our_avg = our_unit["base_price_m2"].mean()
                st.metric(t('our_avg_price'), f"{our_avg:,.0f} {t('tg_m2')}")

            with col2:
                comp_avg = comp_unit["avg_price_m2"].mean()
                st.metric(t('market_avg_price'), f"{comp_avg:,.0f} {t('tg_m2')}")

            with col3:
                position = ((our_avg / comp_avg) - 1) * 100
                st.metric(t('our_position'), f"{position:+.1f}% {t('vs_market')}")

    # ========================================================================
    # POSITIONING BY HOUSING CLASS
    # ========================================================================
    st.subheader(f"🏢 {t('positioning_by_class')}")

    if not our_df.empty and not comp_df.empty:
        # Group by housing class
        our_class = our_df.groupby("housing_class")["base_price_m2"].mean().reset_index()
        our_class.columns = [t('class'), t('our_price_label')]

        comp_class = comp_df.groupby("housing_class")["avg_price_m2"].mean().reset_index()
        comp_class.columns = [t('class'), t('market_price_label')]

        # Merge
        class_comparison = pd.merge(our_class, comp_class, on=t('class'), how="outer")
        class_comparison[t('position')] = ((class_comparison[t('our_price_label')] / class_comparison[t('market_price_label')]) - 1) * 100

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name=t('our_price_label'),
            x=class_comparison[t('class')],
            y=class_comparison[t('our_price_label')],
            marker_color='lightblue'
        ))
        fig.add_trace(go.Bar(
            name=t('market_price_label'),
            x=class_comparison[t('class')],
            y=class_comparison[t('market_price_label')],
            marker_color='lightcoral'
        ))

        fig.update_layout(
            barmode='group',
            title=t('price_by_class'),
            xaxis_title=t('housing_class'),
            yaxis_title=f"{t('avg_price_m2')} ({t('tg_m2')})",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True, key="positioning_by_class_chart")

        st.dataframe(class_comparison, use_container_width=True)

    session.close()


def show_price_comparison_page():
    """Detailed price comparison"""
    st.header(f"📊 {t('price_comp_title')}")

    session = get_session()

    st.markdown(t('price_comp_desc'))

    # ========================================================================
    # BENCHMARK TABLE
    # ========================================================================
    st.subheader(f"📋 {t('benchmark_table')}")

    # Unit type filter
    unit_types = [t('all'), "1-комнатная", "2-комнатная", "3-комнатная", "4-комнатная"]
    selected_unit = st.selectbox(t('unit_type'), unit_types)

    # Get our projects
    our_query = (
        session.query(
            Project.project_name,
            Project.region,
            BasePricing.unit_type,
            BasePricing.base_price_m2
        )
        .join(BasePricing, Project.project_id == BasePricing.project_id)
    )

    if selected_unit != t('all'):
        our_query = our_query.filter(BasePricing.unit_type == selected_unit)

    our_df = pd.read_sql(our_query.statement, session.bind)

    # Get competitors
    comp_query = session.query(
        CompetitorData.competitor_name,
        CompetitorData.complex_name,
        CompetitorData.location,
        CompetitorData.unit_type,
        CompetitorData.avg_price_m2
    ).filter(CompetitorData.avg_price_m2 > 0)

    if selected_unit != t('all'):
        comp_query = comp_query.filter(CompetitorData.unit_type == selected_unit)

    comp_df = pd.read_sql(comp_query.statement, session.bind)

    if not our_df.empty:
        # Calculate statistics
        our_stats = our_df.groupby("unit_type")["base_price_m2"].agg(['mean', 'min', 'max']).reset_index()
        our_stats.columns = [t('unit_type'), t('our_avg'), t('our_min'), t('our_max')]

        if not comp_df.empty:
            comp_stats = comp_df.groupby("unit_type")["avg_price_m2"].agg(['mean', 'min', 'max']).reset_index()
            comp_stats.columns = [t('unit_type'), t('market_avg'), t('market_min'), t('market_max')]

            # Merge
            benchmark = pd.merge(our_stats, comp_stats, on=t('unit_type'), how="outer")
            benchmark[t('position_vs_market')] = ((benchmark[t('our_avg')] / benchmark[t('market_avg')]) - 1) * 100

            st.dataframe(benchmark, use_container_width=True)

            # Visualization
            fig = go.Figure()

            for unit_type in benchmark[t('unit_type')]:
                unit_data = benchmark[benchmark[t('unit_type')] == unit_type].iloc[0]

                # Our range
                fig.add_trace(go.Box(
                    y=[unit_data[t('our_min')], unit_data[t('our_avg')], unit_data[t('our_max')]],
                    name=f"{unit_type} ({t('ours')})",
                    marker_color='lightblue'
                ))

                # Market range
                fig.add_trace(go.Box(
                    y=[unit_data[t('market_min')], unit_data[t('market_avg')], unit_data[t('market_max')]],
                    name=f"{unit_type} ({t('market')})",
                    marker_color='lightcoral'
                ))

            fig.update_layout(
                title=t('price_range_comparison'),
                yaxis_title=f"{t('price_per_m2')} ({t('tg_m2')})",
                showlegend=True,
                height=500
            )

            st.plotly_chart(fig, use_container_width=True, key="price_range_comparison_chart")

    session.close()


def show_market_intelligence_page():
    """Market intelligence and insights"""
    st.header(f"🎯 {t('market_intel_title')}")

    session = get_session()

    st.markdown(t('market_intel_desc'))

    # ========================================================================
    # MARKET TRENDS
    # ========================================================================
    st.subheader(f"📈 {t('market_trends')}")

    # Get competitor data over time
    comp_trends = (
        session.query(
            CompetitorData.date,
            CompetitorData.avg_price_m2,
            CompetitorData.discount_flag
        )
        .filter(CompetitorData.avg_price_m2 > 0)
        .order_by(CompetitorData.date)
    )
    trends_df = pd.read_sql(comp_trends.statement, session.bind)

    if not trends_df.empty:
        # Monthly average
        trends_df['month'] = pd.to_datetime(trends_df['date']).dt.to_period('M').astype(str)
        monthly = trends_df.groupby('month')['avg_price_m2'].mean().reset_index()

        fig = px.line(
            monthly,
            x='month',
            y='avg_price_m2',
            title=t('market_price_trend'),
            labels={'avg_price_m2': f"{t('avg_price_m2')} ({t('tg_m2')})", 'month': t('month')}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True, key="market_intelligence_price_trend_chart")

        # Discount trends
        discount_trend = trends_df.groupby('month')['discount_flag'].mean().reset_index()
        discount_trend['discount_pct'] = discount_trend['discount_flag'] * 100

        fig = px.bar(
            discount_trend,
            x='month',
            y='discount_pct',
            title=t('discount_trend'),
            labels={'discount_pct': t('projects_with_discounts'), 'month': t('month')}
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True, key="discount_trend_chart")

    # ========================================================================
    # COMPETITIVE INSIGHTS
    # ========================================================================
    st.subheader(f"💡 {t('competitive_insights')}")

    # Most active competitors
    active_comps = (
        session.query(
            CompetitorData.competitor_name,
        )
        .filter(CompetitorData.date >= (datetime.now() - timedelta(days=90)).date())
        .all()
    )

    if active_comps:
        comp_counts = pd.Series([c[0] for c in active_comps]).value_counts().head(10)

        st.markdown(f"**{t('most_active_competitors')}**")
        for comp, count in comp_counts.items():
            st.write(f"• **{comp}**: {count} {t('price_updates')}")

    # Discount leaders
    discount_leaders = (
        session.query(CompetitorData.competitor_name)
        .filter(CompetitorData.discount_flag == True)
        .filter(CompetitorData.date >= (datetime.now() - timedelta(days=90)).date())
        .all()
    )

    if discount_leaders:
        discount_counts = pd.Series([c[0] for c in discount_leaders]).value_counts().head(5)

        st.markdown(f"**{t('competitors_with_discounts')}**")
        for comp, count in discount_counts.items():
            st.write(f"• **{comp}**: {count} {t('discounted_offerings')}")

    # ========================================================================
    # STRATEGIC RECOMMENDATIONS
    # ========================================================================
    st.subheader(f"🎯 {t('strategic_recommendations')}")

    # Get our average price
    our_avg = session.query(BasePricing.base_price_m2).all()
    if our_avg:
        our_price = sum([p[0] for p in our_avg]) / len(our_avg)

        # Get market average
        market_avg = session.query(CompetitorData.avg_price_m2).filter(
            CompetitorData.avg_price_m2 > 0
        ).all()

        if market_avg:
            market_price = sum([p[0] for p in market_avg]) / len(market_avg)
            position = ((our_price / market_price) - 1) * 100

            if position > 10:
                st.warning(f"⚠️ {t('prices_above_market').format(position)}")
            elif position < -10:
                st.info(f"ℹ️ {t('prices_below_market').format(abs(position))}")
            else:
                st.success(f"✅ {t('prices_well_positioned').format(position)}")

    # Discount activity alert
    recent_discounts = (
        session.query(CompetitorData)
        .filter(CompetitorData.discount_flag == True)
        .filter(CompetitorData.date >= (datetime.now() - timedelta(days=30)).date())
        .count()
    )

    if recent_discounts > 10:
        st.warning(f"⚠️ {t('high_discount_activity').format(recent_discounts)}")

    session.close()
