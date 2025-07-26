# Advanced Data Analysis Suite

A comprehensive, state-of-the-art data analysis toolkit built for automated exploratory data analysis (EDA), machine learning model selection, advanced visualizations, and intelligent reporting.

## 🚀 Features

### Core Capabilities
- **Automated Exploratory Data Analysis (EDA)**: Comprehensive statistical analysis with distribution analysis, correlation detection, and data quality assessment
- **Advanced Visualizations**: Interactive plots, correlation heatmaps, distribution plots, pair plots, and feature importance charts
- **Automated Machine Learning**: Intelligent model selection for both classification and regression tasks with hyperparameter optimization
- **Intelligent Reporting**: Generate detailed HTML reports with actionable insights and recommendations
- **Data Quality Assessment**: Automated detection of missing values, outliers, and data inconsistencies
- **Statistical Testing**: Normality tests, correlation analysis, and statistical significance testing

### Supported Analysis Types
- **Classification**: Logistic Regression, Random Forest, Gradient Boosting, SVM
- **Regression**: Linear Regression, Random Forest, Gradient Boosting, SVR
- **Time Series**: Trend analysis, seasonality detection, forecasting capabilities
- **Statistical Analysis**: Descriptive statistics, hypothesis testing, distribution fitting

## 📦 Installation

### Core Dependencies
Install the essential packages for the advanced data analysis suite:

```bash
pip install -r requirements_core.txt
```

### Full Dependencies (Optional)
For the complete feature set including advanced ML libraries:

```bash
pip install -r requirements.txt
```

### Key Libraries
- **Data Processing**: pandas, numpy, polars
- **Visualization**: matplotlib, seaborn, plotly, bokeh
- **Machine Learning**: scikit-learn, xgboost, lightgbm
- **Statistical Analysis**: scipy, statsmodels
- **Reporting**: jinja2, ydata-profiling
- **CLI Interface**: typer, rich

## 🛠️ Usage

### Command Line Interface

The suite provides a powerful CLI for quick analysis:

```bash
# Basic analysis
python advanced_data_suite.py analyze data.csv

# Classification analysis with target variable
python advanced_data_suite.py analyze data.csv --target target_column

# Regression analysis
python advanced_data_suite.py analyze data.csv --target numeric_target --task-type regression

# Generate sample data for testing
python examples/generate_example_data.py
```

### Programmatic Usage

```python
from advanced_data_suite import AdvancedDataSuite
import pandas as pd

# Initialize the suite
suite = AdvancedDataSuite()

# Load your data
data = pd.read_csv('your_data.csv')

# Perform comprehensive analysis
results = suite.comprehensive_analysis(
    data=data,
    target_column='your_target',
    task_type='classification'  # or 'regression'
)

# Generate HTML report
suite.generate_report(
    data=data,
    results=results,
    output_path='analysis_report.html'
)
```

### Advanced Features

#### Custom Model Configuration
```python
# Configure custom models for analysis
custom_models = {
    'classification': {
        'Random Forest': RandomForestClassifier(n_estimators=200),
        'XGBoost': XGBClassifier(n_estimators=100)
    },
    'regression': {
        'Random Forest': RandomForestRegressor(n_estimators=200),
        'XGBoost': XGBRegressor(n_estimators=100)
    }
}

suite = AdvancedDataSuite(custom_models=custom_models)
```

#### Automated Feature Engineering
```python
# The suite automatically handles:
# - Missing value imputation
# - Categorical encoding (Label/One-hot)
# - Feature scaling and normalization
# - Outlier detection and handling
```

## 📊 Output Examples

### Generated Reports Include:
1. **Data Overview**: Shape, types, missing values summary
2. **Statistical Summary**: Descriptive statistics for all features
3. **Correlation Analysis**: Heatmaps and correlation matrices
4. **Distribution Analysis**: Histograms, box plots, and normality tests
5. **Model Performance**: Cross-validation scores, feature importance
6. **Actionable Insights**: Data quality recommendations and next steps

### Visualization Gallery:
- Interactive correlation heatmaps
- Feature distribution plots
- Pair plot matrices
- Model performance comparisons
- Feature importance rankings

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set custom plotting backend
export MPLBACKEND=Agg  # For headless environments
```

### Custom Styling
The suite supports custom CSS styling for reports by modifying the HTML template in the `generate_report` method.

## 📁 Project Structure

```
data-analysis-suite/
├── README.md                    # This documentation
├── requirements.txt             # Full dependencies
├── requirements_core.txt        # Essential dependencies
├── advanced_data_suite.py       # Main analysis suite
├── data_analysis_suite.py       # Legacy analysis tools
├── data_analyzer.py            # Simple analysis utilities
└── examples/
    ├── generate_example_data.py # Sample data generator
    └── example_data.csv        # Generated sample dataset
```

## 🎯 Use Cases

### Business Analytics
- Customer segmentation analysis
- Sales forecasting and trend analysis
- Marketing campaign effectiveness
- Product recommendation systems

### Research & Science
- Experimental data analysis
- Statistical hypothesis testing
- Feature discovery and selection
- Predictive modeling validation

### Quality Assurance
- Data quality assessment
- Anomaly detection
- Performance monitoring
- Automated reporting pipelines

## 🚀 Performance Features

- **Efficient Processing**: Optimized pandas operations with chunking for large datasets
- **Memory Management**: Smart memory usage with garbage collection
- **Progress Tracking**: Rich progress bars for long-running operations
- **Parallel Processing**: Multi-core support for model training
- **Caching**: Intelligent caching of computed results

## 🧪 Testing

Generate sample data and test the suite:

```bash
# Generate test data
python examples/generate_example_data.py

# Run analysis on test data
python advanced_data_suite.py analyze example_data.csv --target weather_type
```

## 📈 Advanced Analytics

### Statistical Tests Available:
- Normality testing (Shapiro-Wilk, Jarque-Bera)
- Correlation significance testing
- Feature importance analysis
- Cross-validation with multiple metrics

### Model Evaluation Metrics:
- **Classification**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Regression**: MSE, MAE, R², MAPE

## 🤝 Contributing

This suite is part of the `alex-automation-tools` repository. To contribute:

1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is part of the alex-automation-tools suite. Please refer to the main repository for licensing information.

## 🔗 Related Tools

Check out other tools in the `alex-automation-tools` repository:
- Web scraping utilities
- API automation tools
- Database management scripts
- Workflow automation pipelines

---

**Made with ❤️ for data scientists, analysts, and automation enthusiasts**

For questions, issues, or feature requests, please open an issue in the main repository.
