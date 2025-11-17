"""
Authentication Module for Streamlit Community Cloud
Simple username/password authentication using st.secrets
"""
import streamlit as st
import hashlib
from typing import Optional


def hash_password(password: str) -> str:
    """
    Hash password using SHA256

    Args:
        password: Plain text password

    Returns:
        Hashed password string
    """
    return hashlib.sha256(password.encode()).hexdigest()


def check_password() -> bool:
    """
    Returns True if user has entered correct credentials.
    Uses session state to persist authentication across reruns.

    Returns:
        bool: True if authenticated, False otherwise
    """
    # Check if already authenticated
    if st.session_state.get("authenticated", False):
        return True

    # Show login form
    st.markdown("## 🔐 Dynamic Pricing Analytics System")
    st.markdown("### Login Required")
    st.markdown("Please enter your credentials to access the application.")

    # Add some spacing
    st.markdown("")

    # Create centered login form
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        with st.form("login_form"):
            st.markdown("#### Enter Credentials")

            username = st.text_input(
                "Username",
                key="login_username",
                placeholder="Enter your username"
            )

            password = st.text_input(
                "Password",
                type="password",
                key="login_password",
                placeholder="Enter your password"
            )

            st.markdown("")  # Spacing

            submit = st.form_submit_button(
                "🔓 Login",
                use_container_width=True,
                type="primary"
            )

            if submit:
                if verify_credentials(username, password):
                    # Set authentication state
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.login_time = st.session_state.get("login_time", None)

                    st.success("✅ Login successful! Redirecting...")
                    st.rerun()
                else:
                    st.error("❌ Incorrect username or password")
                    return False

    # Add footer
    st.markdown("---")
    st.caption("🏢 Sensata Real Estate | Dynamic Pricing Analytics v1.0.0")

    return False


def verify_credentials(username: str, password: str) -> bool:
    """
    Verify username and password against stored credentials

    Args:
        username: Provided username
        password: Provided password

    Returns:
        bool: True if credentials are correct
    """
    try:
        # Get credentials from secrets
        correct_username = st.secrets["auth"]["username"]
        correct_password = st.secrets["auth"]["password"]

        # Simple comparison (you can enhance with hashing)
        return username == correct_username and password == correct_password

    except KeyError:
        st.error("⚠️ Authentication configuration error. Please contact administrator.")
        st.error("Make sure secrets.toml is configured correctly.")
        return False
    except Exception as e:
        st.error(f"⚠️ Authentication error: {str(e)}")
        return False


def logout():
    """
    Clear authentication state and reset session
    """
    # Clear all authentication-related session state
    st.session_state.authenticated = False
    st.session_state.user_role = None

    # Remove username if exists
    if "username" in st.session_state:
        del st.session_state.username

    # Clear any other session data
    if "current_project" in st.session_state:
        st.session_state.current_project = None

    if "data_loaded" in st.session_state:
        st.session_state.data_loaded = False


def show_logout_button():
    """
    Display logout button in sidebar
    Should be called within sidebar context
    """
    if st.session_state.get("authenticated", False):
        st.markdown("---")

        # Show logged-in user info
        if st.session_state.get("username"):
            st.caption(f"👤 Logged in as: **{st.session_state.username}**")

        # Logout button
        if st.button("🔓 Logout", use_container_width=True, key="logout_button"):
            logout()
            st.rerun()


def require_authentication(func):
    """
    Decorator to require authentication for a function
    Usage: @require_authentication

    Args:
        func: Function to wrap

    Returns:
        Wrapped function that checks authentication first
    """
    def wrapper(*args, **kwargs):
        if not st.session_state.get("authenticated", False):
            st.warning("⚠️ Please login to access this feature")
            st.stop()
        return func(*args, **kwargs)
    return wrapper


def get_current_user() -> Optional[str]:
    """
    Get currently logged-in username

    Returns:
        str: Username if authenticated, None otherwise
    """
    if st.session_state.get("authenticated", False):
        return st.session_state.get("username")
    return None


def is_authenticated() -> bool:
    """
    Check if user is currently authenticated

    Returns:
        bool: True if authenticated
    """
    return st.session_state.get("authenticated", False)
