"""
Dynamic Pricing Analytics System - Main Application Entry Point
Run with: streamlit run app.py
"""
import streamlit as st
from pathlib import Path
import sys

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import STREAMLIT_CONFIG, VERSION, APP_NAME, COMPANY, USER_ROLES
from config.translations import t
from src.auth import check_password, show_logout_button


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(**STREAMLIT_CONFIG)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
def init_session_state():
    """Initialize session state variables"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user_role" not in st.session_state:
        st.session_state.user_role = None
    if "current_project" not in st.session_state:
        st.session_state.current_project = None
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False


# ============================================================================
# AUTHENTICATION / ROLE SELECTION
# ============================================================================
def show_role_selector():
    """Show role selection page"""
    st.title(f"🏢 {t('app_name')}")
    st.markdown(f"**{t('company')}** | {t('version')} {VERSION}")
    st.markdown("---")

    st.subheader(t('role_selector_welcome'))

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button(
            f"{USER_ROLES['analyst']['icon']}\n\n**{t('analyst')}**",
            use_container_width=True,
            help=t('analyst_help'),
        ):
            st.session_state.user_role = "analyst"
            st.rerun()

    with col2:
        if st.button(
            f"{USER_ROLES['sales_manager']['icon']}\n\n**{t('sales_manager')}**",
            use_container_width=True,
            help=t('sales_manager_help'),
        ):
            st.session_state.user_role = "sales_manager"
            st.rerun()

    with col3:
        if st.button(
            f"{USER_ROLES['executive']['icon']}\n\n**{t('executive')}**",
            use_container_width=True,
            help=t('executive_help'),
        ):
            st.session_state.user_role = "executive"
            st.rerun()

    with col4:
        if st.button(
            f"{USER_ROLES['marketing']['icon']}\n\n**{t('marketing')}**",
            use_container_width=True,
            help=t('marketing_help'),
        ):
            st.session_state.user_role = "marketing"
            st.rerun()

    st.markdown("---")
    st.info(f"""
    **{t('about_system')}**

    {t('about_description')}

    {t('key_features')}:
    - 📊 {t('feature_analytics')}
    - 💰 {t('feature_recommendations')}
    - 🎯 {t('feature_velocity')}
    - 🔍 {t('feature_intelligence')}
    - 📈 {t('feature_optimization')}
    """)


# ============================================================================
# MAIN NAVIGATION
# ============================================================================
def show_sidebar_navigation():
    """Show sidebar navigation based on user role"""
    role = st.session_state.user_role
    role_info = USER_ROLES.get(role, {})

    with st.sidebar:
        st.title(f"{role_info.get('icon', '')} {role_info.get('name', 'User')}")

        # Show logged-in user info
        if st.session_state.get("username"):
            st.caption(f"👤 {st.session_state.username}")

        st.markdown("---")

        # Role-specific navigation
        if role == "analyst":
            page = st.radio(
                t('navigation'),
                [
                    f"🏠 {t('home')}",
                    f"📥 {t('data_import')}",
                    f"✅ {t('data_validation')}",
                    f"🔧 {t('model_parameters')}",
                    f"📊 {t('generate_reports')}",
                ],
            )
        elif role == "sales_manager":
            page = st.radio(
                t('navigation'),
                [
                    f"🏠 {t('home')}",
                    f"💰 {t('price_recommendations')}",
                    f"📋 {t('unit_comparison')}",
                    f"✅ {t('price_approval')}",
                    f"📈 {t('sales_performance')}",
                ],
            )
        elif role == "executive":
            page = st.radio(
                t('navigation'),
                [
                    f"🏠 {t('home')}",
                    f"📊 {t('executive_dashboard')}",
                    f"💼 {t('profit_analysis')}",
                    f"🎯 {t('strategic_pricing')}",
                    f"📈 {t('market_overview')}",
                ],
            )
        elif role == "marketing":
            page = st.radio(
                t('navigation'),
                [
                    f"🏠 {t('home')}",
                    f"🔍 {t('competitor_analysis')}",
                    f"📍 {t('market_positioning')}",
                    f"📊 {t('price_comparison')}",
                    f"🎯 {t('market_intelligence')}",
                ],
            )
        else:
            page = f"🏠 {t('home')}"

        st.markdown("---")

        # Change role button
        if st.button(f"🔄 {t('change_role')}"):
            st.session_state.user_role = None
            st.rerun()

        # Logout button
        show_logout_button()

        # System info
        st.markdown("---")
        st.caption(f"{t('version')} {VERSION}")
        st.caption(f"© {t('company')}")

        return page


# ============================================================================
# HOME PAGE
# ============================================================================
def show_home_page():
    """Show home page for current role"""
    role = st.session_state.user_role
    role_info = USER_ROLES.get(role, {})

    st.title(f"{t('welcome')}, {t(role)}!")

    # Role-specific home content
    if role == "analyst":
        st.markdown(f"""
        ### {t('your_tools')}:
        - **{t('data_import')}**: {t('analyst_tool_import')}
        - **{t('data_validation')}**: {t('analyst_tool_validation')}
        - **{t('model_parameters')}**: {t('analyst_tool_parameters')}
        - **{t('generate_reports')}**: {t('analyst_tool_reports')}

        ### {t('quick_start')}:
        1. {t('analyst_step1')}
        2. {t('analyst_step2')}
        3. {t('analyst_step3')}
        4. {t('analyst_step4')}
        """)

    elif role == "sales_manager":
        st.markdown(f"""
        ### {t('your_dashboard')}:
        - **{t('price_recommendations')}**: {t('sm_tool_recommendations')}
        - **{t('unit_comparison')}**: {t('sm_tool_comparison')}
        - **{t('price_approval')}**: {t('sm_tool_approval')}
        - **{t('sales_performance')}**: {t('sm_tool_performance')}

        ### {t('quick_actions')}:
        - {t('sm_action1')}
        - {t('sm_action2')}
        - {t('sm_action3')}
        """)

    elif role == "executive":
        st.markdown(f"""
        ### {t('strategic_overview')}:
        - **{t('executive_dashboard')}**: {t('exec_tool_dashboard')}
        - **{t('profit_analysis')}**: {t('exec_tool_profit')}
        - **{t('strategic_pricing')}**: {t('exec_tool_strategy')}
        - **{t('market_overview')}**: {t('exec_tool_market')}

        ### {t('key_metrics')}:
        - {t('exec_metric1')}
        - {t('exec_metric2')}
        - {t('exec_metric3')}
        - {t('exec_metric4')}
        """)

    elif role == "marketing":
        st.markdown(f"""
        ### {t('market_intelligence')}:
        - **{t('competitor_analysis')}**: {t('mkt_tool_competitor')}
        - **{t('market_positioning')}**: {t('mkt_tool_positioning')}
        - **{t('price_comparison')}**: {t('mkt_tool_comparison')}
        - **{t('market_intelligence')}**: {t('mkt_tool_intelligence')}

        ### {t('competitive_insights')}:
        - {t('mkt_insight1')}
        - {t('mkt_insight2')}
        - {t('mkt_insight3')}
        - {t('mkt_insight4')}
        """)

    # System status
    st.markdown("---")
    st.subheader(t('system_status'))

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.session_state.data_loaded:
            st.success(f"✅ {t('data_loaded')}")
        else:
            st.warning(f"⚠️ {t('no_data_loaded')}")

    with col2:
        st.info(f"🔄 {t('ready_for_update')}")

    with col3:
        st.info(f"📊 {t('analytics_active')}")


# ============================================================================
# IMPORT UI PAGES
# ============================================================================
from src.ui import analyst_pages, sales_manager_pages, executive_pages, marketing_pages


# ============================================================================
# PLACEHOLDER PAGES (To be implemented)
# ============================================================================
def show_placeholder_page(page_name):
    """Show placeholder for pages not yet implemented"""
    st.title(page_name)
    st.info(f"🚧 **{page_name}** {t('under_construction')}")
    st.markdown(f"""
    {t('feature_coming_soon')}:
    - {t('placeholder_implementation')}
    - {t('placeholder_design')}
    - {t('placeholder_integration')}
    - {t('placeholder_testing')}

    {t('check_back_later')}
    """)


# ============================================================================
# MAIN APPLICATION LOGIC
# ============================================================================
def main():
    """Main application logic"""
    init_session_state()

    # AUTHENTICATION CHECK - Must be first!
    if not check_password():
        st.stop()  # Stop execution if not authenticated

    # Show role selector if no role selected
    if st.session_state.user_role is None:
        show_role_selector()
        return

    # Show navigation and get selected page
    page = show_sidebar_navigation()

    # Route to appropriate page (check translated page names)
    if t('home') in page:
        show_home_page()
    # Analyst pages
    elif t('data_import') in page:
        analyst_pages.show_data_import_page()
    elif t('data_validation') in page:
        analyst_pages.show_data_validation_page()
    elif t('model_parameters') in page:
        analyst_pages.show_model_parameters_page()
    elif t('generate_reports') in page:
        analyst_pages.show_generate_reports_page()
    # Sales Manager pages
    elif t('price_recommendations') in page:
        sales_manager_pages.show_price_recommendations_page()
    elif t('unit_comparison') in page:
        sales_manager_pages.show_unit_comparison_page()
    elif t('price_approval') in page:
        sales_manager_pages.show_price_approval_page()
    elif t('sales_performance') in page:
        sales_manager_pages.show_sales_performance_page()
    # Executive pages
    elif t('executive_dashboard') in page:
        executive_pages.show_executive_dashboard_page()
    elif t('profit_analysis') in page:
        executive_pages.show_profit_analysis_page()
    elif t('strategic_pricing') in page:
        executive_pages.show_strategic_pricing_page()
    elif t('market_overview') in page:
        executive_pages.show_market_overview_page()
    # Marketing pages
    elif t('competitor_analysis') in page:
        marketing_pages.show_competitor_analysis_page()
    elif t('market_positioning') in page:
        marketing_pages.show_market_positioning_page()
    elif t('price_comparison') in page:
        marketing_pages.show_price_comparison_page()
    elif t('market_intelligence') in page:
        marketing_pages.show_market_intelligence_page()
    else:
        show_placeholder_page(page)


# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == "__main__":
    main()
