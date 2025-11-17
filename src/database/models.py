"""
Database Models using SQLAlchemy ORM
Defines all tables and relationships for the pricing system
"""
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Boolean,
    ForeignKey,
    Text,
    Date,
    Enum,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
import enum
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATABASE_PATH, DATABASE_CONFIG

# Base class for all models
Base = declarative_base()


# ============================================================================
# ENUM TYPES
# ============================================================================
class ProjectStatus(enum.Enum):
    """Project status enumeration"""

    PLANNING = "Планируется"
    CONSTRUCTION = "Строится"
    COMPLETED = "Сдан"
    PAUSED = "Приостановлен"


class HousingClass(enum.Enum):
    """Housing class enumeration"""

    COMFORT = "Комфорт"
    COMFORT_PLUS = "Комфорт+"
    BUSINESS = "Бизнес"
    PREMIUM = "Премиум"


class FinishType(enum.Enum):
    """Finish type enumeration"""

    ROUGH = "Черновая"
    STANDARD = "Стандартная"
    PREMIUM = "Премиум"


class ApprovalStatus(enum.Enum):
    """Price approval status"""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"


# ============================================================================
# PROJECT MODELS
# ============================================================================
class Project(Base):
    """Residential project model"""

    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(String(50), unique=True, nullable=False, index=True)
    project_name = Column(String(200), nullable=False)
    location = Column(String(300), nullable=False)
    region = Column(String(100), nullable=False)
    housing_class = Column(String(50), nullable=False)
    construction_material = Column(String(100))
    floors_min = Column(Integer)
    floors_max = Column(Integer)
    floors_total = Column(Integer)  # Average or typical
    blocks_total = Column(Integer)
    developer = Column(String(200))
    status = Column(String(50))
    crm_id = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    units = relationship("Unit", back_populates="project", cascade="all, delete-orphan")
    base_pricing = relationship(
        "BasePricing", back_populates="project", cascade="all, delete-orphan"
    )
    dynamic_signals = relationship(
        "DynamicSignal", back_populates="project", cascade="all, delete-orphan"
    )
    price_recommendations = relationship(
        "PriceRecommendation", back_populates="project", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Project(id={self.project_id}, name={self.project_name})>"


class Unit(Base):
    """Individual unit/apartment model"""

    __tablename__ = "units"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(String(50), ForeignKey("projects.project_id"), nullable=False)
    unit_number = Column(String(50))
    unit_type = Column(String(50), nullable=False)  # 1-комнатная, 2-комнатная, etc.
    area_m2 = Column(Float, nullable=False)
    floor_number = Column(Integer)
    block_number = Column(Integer)
    finish_type = Column(String(50))
    has_view = Column(Boolean, default=False)
    is_corner = Column(Boolean, default=False)
    is_available = Column(Boolean, default=True)
    current_price_m2 = Column(Float)
    current_total_price = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    project = relationship("Project", back_populates="units")

    def __repr__(self):
        return f"<Unit(project={self.project_id}, type={self.unit_type}, area={self.area_m2})>"


# ============================================================================
# PRICING MODELS
# ============================================================================
class BasePricing(Base):
    """Base pricing configuration and cost structure"""

    __tablename__ = "base_pricing"

    id = Column(Integer, primary_key=True, autoincrement=True)
    base_id = Column(Integer)
    project_id = Column(String(50), ForeignKey("projects.project_id"), nullable=False)
    unit_type = Column(String(50), nullable=False)
    area_m2 = Column(Float, nullable=False)
    finish_type = Column(String(50))

    # Cost components (KZT per m²)
    land_cost_m2 = Column(Float)
    construction_cost_m2 = Column(Float, nullable=False)
    infra_cost_m2 = Column(Float)
    overhead_m2 = Column(Float)
    land_cost_estimated = Column(Boolean, default=False)
    infra_cost_estimated = Column(Boolean, default=False)
    overhead_cost_estimated = Column(Boolean, default=False)

    # Pricing coefficients
    developer_margin_pct = Column(Float, nullable=False)
    location_coef = Column(Float, default=1.0)
    floor_coef = Column(Float, default=1.0)
    view_coef = Column(Float, default=1.0)
    finish_coef = Column(Float, default=1.0)

    # Reference prices
    market_price_avg = Column(Float)
    base_price_m2 = Column(Float, nullable=False)
    base_unit_price = Column(Float, nullable=False)
    baseprice_m2 = Column(Float)  # Alternative calculation

    # Metadata
    region_n2 = Column(String(100))
    version_tag = Column(String(50))
    effective_date = Column(Date)
    price_validation_flag = Column(Boolean, default=False)
    area_validation_flag = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    project = relationship("Project", back_populates="base_pricing")

    def __repr__(self):
        return f"<BasePricing(project={self.project_id}, type={self.unit_type}, price_m2={self.base_price_m2})>"


class DynamicSignal(Base):
    """Dynamic pricing signals and sales data"""

    __tablename__ = "dynamic_signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_id = Column(Integer)
    project_id = Column(String(50), ForeignKey("projects.project_id"), nullable=False)
    date = Column(Date, nullable=False, index=True)
    unit_type = Column(Integer)  # Numeric code
    unit_type_name = Column(String(50))  # Human-readable name

    # Sales metrics
    units_sold = Column(Integer, default=0)
    m2_sold = Column(Float, default=0.0)
    total_sales_value = Column(Float, default=0.0)
    avg_price_m2 = Column(Float)
    stock_remaining = Column(Integer)

    # Market indicators
    market_demand_index = Column(Float, default=0.0)
    sales_velocity = Column(Integer, default=0)
    price_change_pct = Column(Float, default=0.0)
    interest_rate_pct = Column(Float, default=0.0)

    # Lead funnel metrics
    leads_received = Column(Integer, default=0)
    office_visits = Column(Integer, default=0)
    contracts_signed = Column(Integer, default=0)
    mortgage_selected = Column(Integer, default=0)
    conversion_rate = Column(Float, default=0.0)
    conversion_rate2 = Column(Float, default=0.0)

    # Flags
    has_suspicious_zeros = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    project = relationship("Project", back_populates="dynamic_signals")

    def __repr__(self):
        return f"<DynamicSignal(project={self.project_id}, date={self.date}, units_sold={self.units_sold})>"


# ============================================================================
# COMPETITOR MODELS
# ============================================================================
class CompetitorData(Base):
    """Competitor pricing data"""

    __tablename__ = "competitor_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    comp_id = Column(Integer)
    date = Column(Date, nullable=False, index=True)
    location = Column(String(200), nullable=False)
    competitor_name = Column(String(200), nullable=False)
    complex_name = Column(String(200), nullable=False)
    housing_class = Column(String(50))
    unit_type = Column(String(50))
    data_period = Column(String(50))  # Month name (июль, август, etc.)
    avg_price_m2 = Column(Float)
    discount_flag = Column(Boolean, default=False)
    price_missing = Column(Boolean, default=False)
    source_url = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<CompetitorData(competitor={self.competitor_name}, complex={self.complex_name}, price_m2={self.avg_price_m2})>"


# ============================================================================
# PRICE RECOMMENDATION & APPROVAL MODELS
# ============================================================================
class PriceRecommendation(Base):
    """AI-generated price recommendations awaiting approval"""

    __tablename__ = "price_recommendations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(String(50), ForeignKey("projects.project_id"), nullable=False)
    unit_type = Column(String(50), nullable=False)
    recommendation_date = Column(Date, nullable=False, default=datetime.utcnow)

    # Current vs recommended pricing
    current_price_m2 = Column(Float)
    recommended_price_m2 = Column(Float, nullable=False)
    price_change_pct = Column(Float)
    price_change_amount = Column(Float)

    # Recommendation rationale
    demand_score = Column(Float)  # Demand curve analysis score
    velocity_score = Column(Float)  # Sales velocity score
    competition_score = Column(Float)  # Competitive positioning score
    profit_score = Column(Float)  # Profit optimization score
    confidence_score = Column(Float)  # Overall confidence (0-1)
    rationale_text = Column(Text)  # Human-readable explanation

    # Approval workflow
    status = Column(String(20), default="pending")  # pending, approved, rejected, applied
    approved_price_m2 = Column(Float)  # Final approved price (may differ from recommendation)
    approved_by = Column(String(100))
    approved_at = Column(DateTime)
    rejection_reason = Column(Text)
    override_reason = Column(Text)  # If manager overrides recommendation

    # Application tracking
    applied_at = Column(DateTime)
    applied_by = Column(String(100))

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    project = relationship("Project", back_populates="price_recommendations")

    def __repr__(self):
        return f"<PriceRecommendation(project={self.project_id}, recommended_price={self.recommended_price_m2}, status={self.status})>"


class PriceHistory(Base):
    """Historical record of all price changes"""

    __tablename__ = "price_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(String(50), nullable=False)
    unit_type = Column(String(50), nullable=False)
    effective_date = Column(Date, nullable=False, index=True)
    price_m2 = Column(Float, nullable=False)
    price_source = Column(String(50))  # 'manual', 'recommendation', 'import'
    changed_by = Column(String(100))
    change_reason = Column(Text)
    recommendation_id = Column(Integer, ForeignKey("price_recommendations.id"))
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<PriceHistory(project={self.project_id}, date={self.effective_date}, price_m2={self.price_m2})>"


# ============================================================================
# AUDIT LOG
# ============================================================================
class AuditLog(Base):
    """System audit log for tracking user actions"""

    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    user_role = Column(String(50))  # analyst, sales_manager, executive, marketing
    action_type = Column(String(100), nullable=False)  # data_import, price_approval, etc.
    entity_type = Column(String(50))  # project, price_recommendation, etc.
    entity_id = Column(String(100))
    description = Column(Text)
    metadata_json = Column(Text)  # JSON string for additional data
    ip_address = Column(String(50))

    def __repr__(self):
        return f"<AuditLog(user={self.user_role}, action={self.action_type}, time={self.timestamp})>"


# ============================================================================
# DATABASE INITIALIZATION
# ============================================================================
def init_database(database_path: str = None):
    """
    Initialize the database and create all tables

    Args:
        database_path: Path to SQLite database file (defaults to config setting)
    """
    if database_path is None:
        database_path = DATABASE_PATH

    # Create engine
    engine = create_engine(
        f"sqlite:///{database_path}", echo=DATABASE_CONFIG.get("echo", False)
    )

    # Create all tables
    Base.metadata.create_all(engine)

    print(f"✅ Database initialized at: {database_path}")
    print(f"📊 Created {len(Base.metadata.tables)} tables:")
    for table_name in Base.metadata.tables.keys():
        print(f"   - {table_name}")

    return engine


def get_session(database_path: str = None):
    """
    Get a database session
    Automatically creates tables if database doesn't exist

    Args:
        database_path: Path to SQLite database file (defaults to config setting)

    Returns:
        SQLAlchemy session object
    """
    if database_path is None:
        database_path = DATABASE_PATH

    # Create database directory if it doesn't exist
    database_path.parent.mkdir(parents=True, exist_ok=True)

    # Create engine
    engine = create_engine(
        f"sqlite:///{database_path}", echo=DATABASE_CONFIG.get("echo", False)
    )

    # Auto-create tables if they don't exist
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    return Session()


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================
if __name__ == "__main__":
    # Initialize database
    engine = init_database()

    print("\n✅ Database schema created successfully!")
    print("\nYou can now import data using the data import pipeline.")
