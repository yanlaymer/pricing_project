# Dynamic Pricing Analytics System

**Sensata Real Estate** - Version 1.0.0

A desktop Python application providing semi-automatic dynamic pricing for residential real estate projects with monthly Excel data updates and full price recommendations with approval workflow.

---

## Features

- **Multi-Role Interface**: Tailored dashboards for Analysts, Sales Managers, Executives, and Marketing
- **Automated Price Recommendations**: AI-driven pricing based on market demand, sales velocity, and competition
- **Excel Integration**: Easy monthly data imports and CRM-ready exports
- **Sales Analytics**: Real-time tracking of sales velocity and conversion rates
- **Competitive Intelligence**: Monitor and analyze competitor pricing strategies
- **Profit Optimization**: Multi-criteria optimization considering financing costs and margins

---

## Quick Start

### Installation

1. **Install Python 3.11+** (if not already installed)
   - Download from [python.org](https://www.python.org/downloads/)

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

---

## Project Structure

```
pricing_project/
├── app.py                      # Main application entry point
├── requirements.txt            # Python dependencies
├── config/                     # Configuration files
│   ├── __init__.py
│   └── config.py              # Application settings
├── src/                       # Source code
│   ├── database/              # Database models and operations
│   ├── models/                # Pricing algorithms and analytics
│   ├── utils/                 # Utility functions
│   └── ui/                    # UI components
├── data/                      # Data directory
│   ├── raw/                   # Original Excel files
│   ├── processed/             # Cleaned data
│   ├── exports/               # Generated exports
│   └── pricing_system.db      # SQLite database
├── tests/                     # Unit tests
├── docs/                      # Documentation
└── logs/                      # Application logs
```

---

## User Roles

### 📈 Analyst
- Upload monthly Excel data
- Validate data quality
- Adjust model parameters
- Generate comprehensive reports

### 💼 Sales Manager
- View price recommendations
- Compare units side-by-side
- Approve/override prices
- Track sales performance

### 👔 Executive
- Strategic pricing dashboard
- Profit analysis and projections
- Market overview
- High-level KPIs

### 📣 Marketing
- Competitor price tracking
- Market positioning analysis
- Price comparison benchmarks
- Market intelligence insights

---

## Monthly Update Workflow

1. **Collect Data**: Sales team updates Excel file with monthly sales, inventory, competitor data
2. **Import**: Analyst uploads Excel via UI
3. **Validate**: System checks data quality and flags issues
4. **Calculate**: Engine recalculates price recommendations
5. **Review**: Sales Manager/Executive reviews and approves
6. **Export**: Generate CRM-ready Excel for price updates

---

## Data Requirements

### Excel File Format

The system expects an Excel file with the following sheets:

1. **residential_projects**: Project metadata (location, class, status)
2. **pricing_start_base**: Base pricing with cost breakdown
3. **pricing_dynamic_signals**: Sales performance time series
4. **competitor_market_data**: Competitor pricing data
5. **Справочник**: Reference data

See `data/pricing_data.xlsx` for the expected format.

---

## Configuration

Key settings can be adjusted in `config/config.py`:

- **Pricing coefficients**: Location, floor, view, finish adjustments
- **Margin ranges**: By housing class (Комфорт, Бизнес, Премиум)
- **Target velocity**: Sales targets by unit type
- **Demand model**: Regression parameters
- **Competitive positioning**: Strategy weights

---

## Technical Stack

- **Python 3.11+**
- **Streamlit** - Modern web UI framework
- **Pandas** - Data processing
- **SQLite** - Embedded database
- **Scikit-learn** - Machine learning models
- **Plotly** - Interactive visualizations
- **Openpyxl** - Excel file handling

---

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/
flake8 src/
```

### Building Standalone Executable

```bash
pyinstaller --onefile --windowed --name="PricingAnalytics" app.py
```

---

## Support

For issues or questions, contact the development team.

---

## License

Proprietary - Sensata Real Estate © 2025
