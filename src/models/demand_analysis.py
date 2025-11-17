"""
Demand Curve Analysis Module
Generates demand curve visualizations and analysis for UI display
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DEMAND_MODEL_CONFIG, TARGET_VELOCITY
from src.database.models import DynamicSignal, BasePricing, get_session


class DemandCurveAnalyzer:
    """Analyzes and visualizes demand curves for pricing decisions"""

    def __init__(self):
        self.session = get_session()

    def get_demand_curve_data(
        self,
        project_id: str,
        unit_type: str,
        lookback_days: int = None
    ) -> Dict:
        """
        Get demand curve data and analysis for a specific unit type

        Args:
            project_id: Project identifier
            unit_type: Unit type (e.g., "1-комнатная")
            lookback_days: Days to look back (default from config)

        Returns:
            Dictionary with demand curve data, model, and visualization
        """
        if lookback_days is None:
            lookback_days = DEMAND_MODEL_CONFIG["lookback_period_days"]

        cutoff_date = datetime.now().date() - timedelta(days=lookback_days)

        # Get sales data
        sales_data = (
            self.session.query(DynamicSignal)
            .filter_by(project_id=project_id)
            .filter(DynamicSignal.unit_type_name == unit_type)
            .filter(DynamicSignal.date >= cutoff_date)
            .order_by(DynamicSignal.date)
            .all()
        )

        if len(sales_data) < DEMAND_MODEL_CONFIG["min_data_points"]:
            return {
                "status": "insufficient_data",
                "message": f"Insufficient data: {len(sales_data)} points (need {DEMAND_MODEL_CONFIG['min_data_points']})",
                "data_points": len(sales_data),
            }

        # Extract price and units sold
        dates = []
        prices = []
        units_sold = []
        velocities = []

        for signal in sales_data:
            if signal.avg_price_m2 and signal.avg_price_m2 > 0:
                dates.append(signal.date)
                prices.append(signal.avg_price_m2)
                units_sold.append(signal.units_sold if signal.units_sold else 0)
                velocities.append(signal.sales_velocity if signal.sales_velocity else 0)

        if len(prices) < 3:
            return {
                "status": "insufficient_data",
                "message": f"Insufficient price points: {len(prices)} (need 3+)",
                "data_points": len(prices),
            }

        # Fit linear regression: units_sold = α × price + β
        X = np.array(prices).reshape(-1, 1)
        y = np.array(units_sold)

        model = LinearRegression()
        model.fit(X, y)

        # Calculate R² and predictions
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)

        # Get target velocity for this unit type
        target = TARGET_VELOCITY.get(unit_type, 10)

        # Calculate optimal price from model
        # target = α × price + β  =>  price = (target - β) / α
        optimal_price = None
        if model.coef_[0] != 0:
            optimal_price = (target - model.intercept_) / model.coef_[0]

        # Calculate price range for visualization
        price_min = min(prices) * 0.9
        price_max = max(prices) * 1.1
        price_range = np.linspace(price_min, price_max, 100)
        demand_curve = model.predict(price_range.reshape(-1, 1))

        # Get current price
        current_price = (
            self.session.query(BasePricing.base_price_m2)
            .filter_by(project_id=project_id, unit_type=unit_type)
            .first()
        )
        current_price = current_price[0] if current_price else None

        return {
            "status": "success",
            "data_points": len(prices),
            "dates": dates,
            "prices": prices,
            "units_sold": units_sold,
            "velocities": velocities,
            "model": {
                "slope": float(model.coef_[0]),
                "intercept": float(model.intercept_),
                "r2": float(r2),
                "equation": f"Units Sold = {model.coef_[0]:.6f} × Price + {model.intercept_:.2f}",
            },
            "predictions": {
                "price_range": price_range.tolist(),
                "demand_curve": demand_curve.tolist(),
            },
            "targets": {
                "target_velocity": target,
                "optimal_price": float(optimal_price) if optimal_price else None,
                "current_price": float(current_price) if current_price else None,
            },
            "confidence": {
                "r2": float(r2),
                "is_reliable": r2 >= DEMAND_MODEL_CONFIG["confidence_threshold"],
                "level": "High" if r2 >= 0.8 else "Medium" if r2 >= 0.6 else "Low",
            }
        }

    def create_demand_curve_plot(
        self,
        project_id: str,
        unit_type: str,
        project_name: str = None
    ) -> go.Figure:
        """
        Create interactive Plotly demand curve visualization

        Args:
            project_id: Project identifier
            unit_type: Unit type
            project_name: Project name for title

        Returns:
            Plotly Figure object
        """
        data = self.get_demand_curve_data(project_id, unit_type)

        if data["status"] != "success":
            # Create error plot
            fig = go.Figure()
            fig.add_annotation(
                text=data["message"],
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="red")
            )
            fig.update_layout(
                title=f"Demand Curve - {project_name or project_id} - {unit_type}",
                xaxis_title="Price (тг/м²)",
                yaxis_title="Units Sold",
                height=500
            )
            return fig

        # Create interactive plot
        fig = go.Figure()

        # Add actual data points
        fig.add_trace(go.Scatter(
            x=data["prices"],
            y=data["units_sold"],
            mode='markers',
            name='Actual Sales',
            marker=dict(
                size=10,
                color=data["velocities"],
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title="Sales<br>Velocity"),
                line=dict(width=1, color='DarkSlateGrey')
            ),
            text=[f"Date: {d}<br>Price: {p:,.0f}<br>Units: {u}<br>Velocity: {v}"
                  for d, p, u, v in zip(data["dates"], data["prices"], data["units_sold"], data["velocities"])],
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))

        # Add demand curve (regression line)
        fig.add_trace(go.Scatter(
            x=data["predictions"]["price_range"],
            y=data["predictions"]["demand_curve"],
            mode='lines',
            name=f'Demand Curve (R²={data["model"]["r2"]:.3f})',
            line=dict(color='red', width=2, dash='dash'),
            hovertemplate='Price: %{x:,.0f} тг/м²<br>Predicted Units: %{y:.1f}<extra></extra>'
        ))

        # Add target velocity line
        target_velocity = data["targets"]["target_velocity"]
        price_range = data["predictions"]["price_range"]
        fig.add_trace(go.Scatter(
            x=[min(price_range), max(price_range)],
            y=[target_velocity, target_velocity],
            mode='lines',
            name=f'Target Velocity ({target_velocity} units/month)',
            line=dict(color='green', width=2, dash='dot'),
            hovertemplate=f'Target: {target_velocity} units/month<extra></extra>'
        ))

        # Add current price marker
        if data["targets"]["current_price"]:
            current_price = data["targets"]["current_price"]
            # Predict units at current price
            current_units = (
                data["model"]["slope"] * current_price + data["model"]["intercept"]
            )
            fig.add_trace(go.Scatter(
                x=[current_price],
                y=[current_units],
                mode='markers',
                name='Current Price',
                marker=dict(size=15, color='orange', symbol='star', line=dict(width=2, color='black')),
                hovertemplate=f'<b>Current Price</b><br>Price: {current_price:,.0f} тг/м²<br>Expected Units: {current_units:.1f}<extra></extra>'
            ))

        # Add optimal price marker
        if data["targets"]["optimal_price"]:
            optimal_price = data["targets"]["optimal_price"]
            if min(price_range) <= optimal_price <= max(price_range):
                fig.add_trace(go.Scatter(
                    x=[optimal_price],
                    y=[target_velocity],
                    mode='markers',
                    name='Optimal Price',
                    marker=dict(size=15, color='lime', symbol='diamond', line=dict(width=2, color='darkgreen')),
                    hovertemplate=f'<b>Optimal Price</b><br>Price: {optimal_price:,.0f} тг/м²<br>Target Units: {target_velocity}<extra></extra>'
                ))

        # Update layout
        title = f"Demand Curve Analysis - {project_name or project_id} - {unit_type}"
        if data["confidence"]["is_reliable"]:
            title += f" ✓ ({data['confidence']['level']} Confidence)"
        else:
            title += f" ⚠ ({data['confidence']['level']} Confidence - Use with caution)"

        fig.update_layout(
            title=title,
            xaxis_title="Price per m² (KZT)",
            yaxis_title="Units Sold per Period",
            hovermode='closest',
            height=600,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        # Add annotation with model equation
        fig.add_annotation(
            text=f"<b>Model:</b> {data['model']['equation']}<br><b>R²:</b> {data['model']['r2']:.3f} ({data['confidence']['level']})",
            xref="paper", yref="paper",
            x=0.98, y=0.02,
            showarrow=False,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=10),
            align="right",
            xanchor="right",
            yanchor="bottom"
        )

        return fig

    def create_price_elasticity_plot(
        self,
        project_id: str,
        unit_type: str
    ) -> go.Figure:
        """
        Create price elasticity visualization showing revenue sensitivity

        Args:
            project_id: Project identifier
            unit_type: Unit type

        Returns:
            Plotly Figure showing revenue vs price
        """
        data = self.get_demand_curve_data(project_id, unit_type)

        if data["status"] != "success":
            fig = go.Figure()
            fig.add_annotation(
                text=data["message"],
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="red")
            )
            return fig

        # Calculate revenue at each price point
        price_range = np.array(data["predictions"]["price_range"])
        demand = np.array(data["predictions"]["demand_curve"])
        revenue = price_range * demand

        # Find revenue-maximizing price
        max_revenue_idx = np.argmax(revenue)
        optimal_revenue_price = price_range[max_revenue_idx]
        max_revenue = revenue[max_revenue_idx]

        fig = go.Figure()

        # Revenue curve
        fig.add_trace(go.Scatter(
            x=price_range,
            y=revenue,
            mode='lines',
            name='Total Revenue',
            line=dict(color='green', width=3),
            fill='tozeroy',
            fillcolor='rgba(0,255,0,0.1)',
            hovertemplate='Price: %{x:,.0f} тг/м²<br>Revenue: %{y:,.0f} KZT<extra></extra>'
        ))

        # Mark revenue-maximizing point
        fig.add_trace(go.Scatter(
            x=[optimal_revenue_price],
            y=[max_revenue],
            mode='markers',
            name='Revenue-Maximizing Price',
            marker=dict(size=15, color='gold', symbol='star', line=dict(width=2, color='black')),
            hovertemplate=f'<b>Max Revenue Point</b><br>Price: {optimal_revenue_price:,.0f} тг/м²<br>Revenue: {max_revenue:,.0f} KZT<extra></extra>'
        ))

        # Add current price if available
        if data["targets"]["current_price"]:
            current_price = data["targets"]["current_price"]
            current_demand = data["model"]["slope"] * current_price + data["model"]["intercept"]
            current_revenue = current_price * current_demand

            fig.add_trace(go.Scatter(
                x=[current_price],
                y=[current_revenue],
                mode='markers',
                name='Current Price Point',
                marker=dict(size=12, color='orange', symbol='diamond'),
                hovertemplate=f'<b>Current</b><br>Price: {current_price:,.0f} тг/м²<br>Revenue: {current_revenue:,.0f} KZT<extra></extra>'
            ))

        fig.update_layout(
            title=f"Revenue Optimization Analysis - {unit_type}",
            xaxis_title="Price per m² (KZT)",
            yaxis_title="Total Revenue (KZT)",
            hovermode='closest',
            height=500
        )

        return fig

    def get_demand_summary_stats(
        self,
        project_id: str,
        unit_type: str
    ) -> Dict:
        """
        Get summary statistics for demand analysis

        Returns:
            Dictionary with key metrics
        """
        data = self.get_demand_curve_data(project_id, unit_type)

        if data["status"] != "success":
            return {"status": "error", "message": data["message"]}

        # Calculate price elasticity at current price
        if data["targets"]["current_price"]:
            current_price = data["targets"]["current_price"]
            slope = data["model"]["slope"]
            intercept = data["model"]["intercept"]
            current_demand = slope * current_price + intercept

            # Elasticity = (dQ/dP) × (P/Q)
            elasticity = slope * (current_price / current_demand) if current_demand != 0 else None
        else:
            elasticity = None

        return {
            "status": "success",
            "data_quality": {
                "points": data["data_points"],
                "r2": data["model"]["r2"],
                "confidence": data["confidence"]["level"],
                "reliable": data["confidence"]["is_reliable"],
            },
            "pricing": {
                "current_price": data["targets"]["current_price"],
                "optimal_price_velocity": data["targets"]["optimal_price"],
                "price_range_min": min(data["prices"]),
                "price_range_max": max(data["prices"]),
            },
            "demand": {
                "avg_units_sold": np.mean(data["units_sold"]),
                "max_units_sold": max(data["units_sold"]),
                "target_velocity": data["targets"]["target_velocity"],
                "slope": data["model"]["slope"],
                "elasticity": elasticity,
            }
        }

    def __del__(self):
        """Cleanup: close database session"""
        if hasattr(self, "session"):
            self.session.close()
