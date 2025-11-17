"""
Core Pricing Engine
Implements dynamic pricing algorithms based on Habr articles methodology:
1. Cost-plus baseline with margin
2. Linear demand curve fitting
3. Market-based adjustments
4. Multi-criteria optimization
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    MARGIN_BY_CLASS,
    DEFAULT_COEFFICIENTS,
    TARGET_VELOCITY,
    DEMAND_MODEL_CONFIG,
    POSITIONING_STRATEGIES,
)
from src.database.models import (
    Project,
    BasePricing,
    DynamicSignal,
    CompetitorData,
    PriceRecommendation,
    get_session,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PricingEngine:
    """
    Dynamic pricing engine implementing multiple pricing strategies

    Pricing Formula:
    Recommended_Price = max(
        Cost_Based_Price,  # Floor: Cost + Margin
        Market_Based_Price,  # Based on demand curve and competition
    )

    Where:
    - Cost_Based_Price = construction_cost_m2 × (1 + margin%)
    - Market_Based_Price = base_market_price × demand_multiplier × attribute_adjustments
    """

    def __init__(self):
        self.session = get_session()

    def calculate_recommended_price(
        self,
        project_id: str,
        unit_type: str,
        strategy: str = "balanced",
    ) -> Dict:
        """
        Calculate recommended price for a unit type in a project

        Args:
            project_id: Project identifier
            unit_type: Unit type (e.g., "1-комнатная")
            strategy: Pricing strategy ("aggressive", "neutral", "balanced", "premium")

        Returns:
            Dictionary with recommendation details
        """
        logger.info(f"Calculating price for {project_id} / {unit_type} (strategy={strategy})")

        # Get project info
        project = self.session.query(Project).filter_by(project_id=project_id).first()
        if not project:
            raise ValueError(f"Project not found: {project_id}")

        # Get base pricing data
        base_pricing = (
            self.session.query(BasePricing)
            .filter_by(project_id=project_id, unit_type=unit_type)
            .first()
        )
        if not base_pricing:
            raise ValueError(f"No base pricing found for {project_id} / {unit_type}")

        # Calculate different pricing components
        cost_based_price = self._calculate_cost_based_price(project, base_pricing)

        market_price_estimate = self._calculate_market_based_price(
            project, base_pricing, unit_type
        )

        demand_adjusted_price = self._calculate_demand_adjusted_price(
            project_id, unit_type, market_price_estimate
        )

        competition_adjusted_price = self._calculate_competition_adjusted_price(
            project, unit_type, demand_adjusted_price, strategy
        )

        # Final recommendation: ensure we're above cost floor
        recommended_price = max(cost_based_price["price_m2"], competition_adjusted_price)

        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            project_id, unit_type, base_pricing
        )

        # Generate recommendation rationale
        rationale = self._generate_rationale(
            cost_based_price,
            market_price_estimate,
            demand_adjusted_price,
            competition_adjusted_price,
            recommended_price,
            confidence_score,
        )

        return {
            "project_id": project_id,
            "unit_type": unit_type,
            "current_price_m2": base_pricing.base_price_m2,
            "recommended_price_m2": round(recommended_price, 0),
            "price_change_pct": (
                (recommended_price - base_pricing.base_price_m2) / base_pricing.base_price_m2
                if base_pricing.base_price_m2 > 0
                else 0
            ),
            "price_change_amount": recommended_price - base_pricing.base_price_m2,
            "cost_floor_price": cost_based_price["price_m2"],
            "margin_pct": cost_based_price["margin_pct"],
            "market_price": market_price_estimate,
            "demand_adjusted": demand_adjusted_price,
            "competition_adjusted": competition_adjusted_price,
            "confidence_score": confidence_score,
            "rationale": rationale,
            "strategy": strategy,
        }

    def _calculate_cost_based_price(
        self, project: Project, base_pricing: BasePricing
    ) -> Dict:
        """
        Calculate cost-based pricing floor

        Formula: Cost_Price = construction_cost_m2 × (1 + margin%)

        Note: construction_cost_m2 is ALL-IN cost (includes land, infra, overhead)
        """
        # Get margin for housing class
        housing_class = project.housing_class
        margin_config = MARGIN_BY_CLASS.get(
            housing_class, MARGIN_BY_CLASS["Комфорт"]
        )

        # Use current margin or default
        margin_pct = base_pricing.developer_margin_pct or margin_config["default"]

        # Calculate cost-based price
        cost_price = base_pricing.construction_cost_m2 * (1 + margin_pct)

        return {
            "price_m2": cost_price,
            "margin_pct": margin_pct,
            "construction_cost": base_pricing.construction_cost_m2,
        }

    def _calculate_market_based_price(
        self, project: Project, base_pricing: BasePricing, unit_type: str
    ) -> float:
        """
        Calculate market-based price using comparable projects

        Uses market_price_avg with attribute adjustments
        """
        # Start with market average
        if base_pricing.market_price_avg and base_pricing.market_price_avg > 0:
            market_price = base_pricing.market_price_avg
        else:
            # Fallback: use regional average
            market_price = self._get_regional_average_price(
                project.region, unit_type
            )

        # Apply attribute adjustments
        # These coefficients adjust for specific unit features
        location_adj = base_pricing.location_coef or DEFAULT_COEFFICIENTS["location_coef"]
        floor_adj = base_pricing.floor_coef or DEFAULT_COEFFICIENTS["floor_coef"]
        view_adj = base_pricing.view_coef or DEFAULT_COEFFICIENTS["view_coef"]
        finish_adj = base_pricing.finish_coef or DEFAULT_COEFFICIENTS["finish_coef"]

        adjusted_price = market_price * location_adj * floor_adj * view_adj * finish_adj

        return adjusted_price

    def _calculate_demand_adjusted_price(
        self, project_id: str, unit_type: str, base_price: float
    ) -> float:
        """
        Adjust price based on demand curve analysis (Habr article method)

        Uses linear regression on recent sales to find optimal price point
        """
        # Get recent sales data (last 90 days)
        lookback_days = DEMAND_MODEL_CONFIG["lookback_period_days"]
        cutoff_date = datetime.now().date() - timedelta(days=lookback_days)

        sales_data = (
            self.session.query(DynamicSignal)
            .filter_by(project_id=project_id)
            .filter(DynamicSignal.unit_type_name == unit_type)
            .filter(DynamicSignal.date >= cutoff_date)
            .all()
        )

        if len(sales_data) < DEMAND_MODEL_CONFIG["min_data_points"]:
            logger.info(
                f"Insufficient data for demand curve ({len(sales_data)} points) - using base price"
            )
            return base_price

        # Extract price and units sold
        prices = []
        units_sold = []
        for signal in sales_data:
            if signal.avg_price_m2 and signal.avg_price_m2 > 0:
                prices.append(signal.avg_price_m2)
                units_sold.append(signal.units_sold if signal.units_sold else 0)

        if len(prices) < 3:
            return base_price

        # Fit linear demand curve: units_sold = α × price + β
        X = np.array(prices).reshape(-1, 1)
        y = np.array(units_sold)

        try:
            model = LinearRegression()
            model.fit(X, y)
            r2 = r2_score(y, model.predict(X))

            if r2 < DEMAND_MODEL_CONFIG["confidence_threshold"]:
                logger.info(f"Low R² ({r2:.2f}) - demand curve not reliable")
                return base_price

            # Calculate target velocity
            target = TARGET_VELOCITY.get(unit_type, 10)

            # Solve for price: target = α × price + β
            # price = (target - β) / α
            if model.coef_[0] != 0:
                optimal_price = (target - model.intercept_) / model.coef_[0]

                # Sanity check: price should be within reasonable range of base
                if 0.7 * base_price <= optimal_price <= 1.5 * base_price:
                    logger.info(
                        f"Demand curve suggests price: {optimal_price:,.0f} (R²={r2:.2f})"
                    )
                    return optimal_price

        except Exception as e:
            logger.warning(f"Demand curve fitting failed: {e}")

        return base_price

    def _calculate_competition_adjusted_price(
        self,
        project: Project,
        unit_type: str,
        base_price: float,
        strategy: str,
    ) -> float:
        """
        Adjust price based on competitive positioning

        Args:
            strategy: "aggressive" (below market), "neutral", "premium" (above market)
        """
        # Get competitor prices in same location and class
        competitors = (
            self.session.query(CompetitorData)
            .filter_by(location=project.region, unit_type=unit_type)
            .filter(CompetitorData.avg_price_m2 > 0)
            .order_by(CompetitorData.date.desc())
            .limit(50)
            .all()
        )

        if not competitors:
            logger.info("No competitor data available")
            return base_price

        # Calculate competitive average
        comp_prices = [c.avg_price_m2 for c in competitors if c.avg_price_m2 > 0]
        if not comp_prices:
            return base_price

        comp_avg = np.mean(comp_prices)

        # Apply strategy adjustment
        strategy_adj = POSITIONING_STRATEGIES.get(strategy, 0.0)
        adjusted_price = base_price * (1 + strategy_adj)

        # Log competitive position
        position_pct = (adjusted_price - comp_avg) / comp_avg * 100
        logger.info(
            f"Competition: avg={comp_avg:,.0f}, strategy={strategy}, "
            f"position={position_pct:+.1f}% vs market"
        )

        return adjusted_price

    def _get_regional_average_price(self, region: str, unit_type: str) -> float:
        """Get average price for region and unit type"""
        avg_result = (
            self.session.query(BasePricing.market_price_avg)
            .join(Project)
            .filter(Project.region == region)
            .filter(BasePricing.unit_type == unit_type)
            .filter(BasePricing.market_price_avg > 0)
            .all()
        )

        if avg_result:
            prices = [r[0] for r in avg_result if r[0]]
            return np.mean(prices) if prices else 500000  # fallback

        return 500000  # default fallback

    def _calculate_confidence_score(
        self, project_id: str, unit_type: str, base_pricing: BasePricing
    ) -> float:
        """
        Calculate confidence score for the recommendation (0-1)

        Based on:
        - Data availability
        - Market data quality
        - Sales history depth
        """
        score = 0.5  # base score

        # Boost for having market price data
        if base_pricing.market_price_avg and base_pricing.market_price_avg > 0:
            score += 0.15

        # Boost for recent sales data
        recent_sales = (
            self.session.query(DynamicSignal)
            .filter_by(project_id=project_id)
            .filter(DynamicSignal.unit_type_name == unit_type)
            .filter(DynamicSignal.date >= datetime.now().date() - timedelta(days=90))
            .count()
        )
        if recent_sales >= 10:
            score += 0.20
        elif recent_sales >= 5:
            score += 0.10

        # Boost for competitor data
        comp_count = (
            self.session.query(CompetitorData)
            .filter_by(unit_type=unit_type)
            .filter(CompetitorData.avg_price_m2 > 0)
            .count()
        )
        if comp_count >= 5:
            score += 0.15

        return min(score, 1.0)

    def _generate_rationale(
        self,
        cost_based: Dict,
        market_price: float,
        demand_adjusted: float,
        competition_adjusted: float,
        final_price: float,
        confidence: float,
    ) -> str:
        """Generate human-readable explanation of pricing recommendation"""
        rationale_parts = []

        # Cost floor
        rationale_parts.append(
            f"Минимальная цена (себестоимость + маржа {cost_based['margin_pct']*100:.0f}%): "
            f"{cost_based['price_m2']:,.0f} тг/м²"
        )

        # Market comparison
        diff_from_market = (final_price - market_price) / market_price * 100
        rationale_parts.append(
            f"Рыночная цена: {market_price:,.0f} тг/м² "
            f"(рекомендация {diff_from_market:+.1f}% от рынка)"
        )

        # Demand signal
        if abs(demand_adjusted - market_price) > 0.01 * market_price:
            direction = "выше" if demand_adjusted > market_price else "ниже"
            rationale_parts.append(
                f"Анализ спроса предлагает цену {direction}: {demand_adjusted:,.0f} тг/м²"
            )

        # Competition
        diff_from_demand = (competition_adjusted - demand_adjusted) / demand_adjusted * 100
        if abs(diff_from_demand) > 1:
            rationale_parts.append(
                f"Конкурентная корректировка: {diff_from_demand:+.1f}%"
            )

        # Confidence
        conf_level = "высокая" if confidence > 0.7 else "средняя" if confidence > 0.5 else "низкая"
        rationale_parts.append(f"Уверенность: {conf_level} ({confidence:.1%})")

        return " | ".join(rationale_parts)

    def generate_recommendations_for_project(
        self, project_id: str, strategy: str = "balanced"
    ) -> List[Dict]:
        """
        Generate price recommendations for all unit types in a project

        Args:
            project_id: Project identifier
            strategy: Pricing strategy to apply

        Returns:
            List of recommendation dictionaries
        """
        # Get all unit types for this project
        unit_types = (
            self.session.query(BasePricing.unit_type)
            .filter_by(project_id=project_id)
            .distinct()
            .all()
        )

        recommendations = []
        for (unit_type,) in unit_types:
            try:
                rec = self.calculate_recommended_price(project_id, unit_type, strategy)
                recommendations.append(rec)
            except Exception as e:
                logger.error(f"Failed to generate recommendation for {unit_type}: {e}")

        return recommendations

    def save_recommendations_to_db(
        self, recommendations: List[Dict], user_role: str = "analyst"
    ) -> int:
        """
        Save price recommendations to database for approval workflow

        Returns:
            Number of recommendations saved
        """
        saved_count = 0

        for rec in recommendations:
            try:
                # Create price recommendation record
                price_rec = PriceRecommendation(
                    project_id=rec["project_id"],
                    unit_type=rec["unit_type"],
                    recommendation_date=datetime.now().date(),
                    current_price_m2=rec["current_price_m2"],
                    recommended_price_m2=rec["recommended_price_m2"],
                    price_change_pct=rec["price_change_pct"],
                    price_change_amount=rec["price_change_amount"],
                    demand_score=rec.get("confidence_score", 0.5),
                    velocity_score=0.0,  # TODO: calculate
                    competition_score=0.0,  # TODO: calculate
                    profit_score=0.0,  # TODO: calculate
                    confidence_score=rec["confidence_score"],
                    rationale_text=rec["rationale"],
                    status="pending",
                )

                self.session.add(price_rec)
                saved_count += 1

            except Exception as e:
                logger.error(f"Failed to save recommendation: {e}")
                self.session.rollback()
                continue

        self.session.commit()
        logger.info(f"✓ Saved {saved_count} recommendations to database")

        return saved_count

    def __del__(self):
        """Cleanup: close database session"""
        if hasattr(self, "session"):
            self.session.close()


# ============================================================================
# STANDALONE SCRIPT EXECUTION
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate pricing recommendations")
    parser.add_argument("project_id", help="Project ID to generate recommendations for")
    parser.add_argument(
        "--strategy",
        choices=["aggressive", "neutral", "balanced", "premium"],
        default="balanced",
        help="Pricing strategy",
    )
    parser.add_argument(
        "--save", action="store_true", help="Save recommendations to database"
    )

    args = parser.parse_args()

    engine = PricingEngine()

    print(f"\n🎯 Generating recommendations for project: {args.project_id}")
    print(f"Strategy: {args.strategy}\n")
    print("=" * 80)

    recommendations = engine.generate_recommendations_for_project(
        args.project_id, args.strategy
    )

    for rec in recommendations:
        print(f"\n📋 {rec['unit_type']}")
        print(f"   Current Price: {rec['current_price_m2']:,.0f} тг/м²")
        print(f"   Recommended:   {rec['recommended_price_m2']:,.0f} тг/м²")
        print(f"   Change:        {rec['price_change_pct']:+.1%} ({rec['price_change_amount']:+,.0f} тг/м²)")
        print(f"   Cost Floor:    {rec['cost_floor_price']:,.0f} тг/м²")
        print(f"   Margin:        {rec['margin_pct']:.1%}")
        print(f"   Confidence:    {rec['confidence_score']:.1%}")
        print(f"   {rec['rationale']}")

    print("\n" + "=" * 80)

    if args.save:
        saved = engine.save_recommendations_to_db(recommendations)
        print(f"\n✅ Saved {saved} recommendations to database for approval")
