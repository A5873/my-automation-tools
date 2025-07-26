#!/usr/bin/env python3
"""
Data Analysis Suite
A comprehensive toolkit for data analysis, visualization, and reporting.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy import stats
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
    import statsmodels.api as sm
    from statsmodels.tsa.seasonal import seasonal_decompose
    from jinja2 import Template
    from fpdf import FPDF
    from tqdm import tqdm
    from colorama import init, Fore, Style
    from tabulate import tabulate
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with: pip install -r requirements.txt")
    sys.exit(1)

# Initialize colorama
init(autoreset=True)

class DataAnalysisSuite:
    """Comprehensive data analysis and visualization toolkit."""
    
    def __init__(self):
        self.data = None
        self.original_data = None
        self.analysis_results = {}
        self.output_dir = Path("analysis_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def print_banner(self):
        """Display welcome banner."""
        print(f"\n{Fore.CYAN}{'='*70}")
        print(f"{Fore.CYAN}📊 DATA ANALYSIS SUITE 📊")
        print(f"{Fore.CYAN}{'='*70}")
        print(f"{Fore.YELLOW}Your comprehensive data analysis and visualization toolkit!")
        print(f"{Fore.GREEN}Output directory: {self.output_dir.absolute()}")
        print(f"{Fore.CYAN}{'='*70}\n")
    
    def load_data(self, file_path: str, **kwargs) -> bool:
        """Load data from various file formats."""
        try:
            file_path = Path(file_path)
            print(f"{Fore.CYAN}📂 Loading data from: {file_path.name}")
            
            if file_path.suffix.lower() == '.csv':
                self.data = pd.read_csv(file_path, **kwargs)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                self.data = pd.read_excel(file_path, **kwargs)
            elif file_path.suffix.lower() == '.json':
                self.data = pd.read_json(file_path, **kwargs)
            elif file_path.suffix.lower() == '.parquet':
                self.data = pd.read_parquet(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            self.original_data = self.data.copy()
            
            print(f"{Fore.GREEN}✅ Data loaded successfully!")
            print(f"{Fore.YELLOW}📊 Shape: {self.data.shape[0]:,} rows × {self.data.shape[1]} columns")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error loading data: {e}")
            return False
    
    def get_data_overview(self) -> Dict:
        """Get comprehensive overview of the dataset."""
        if self.data is None:
            return {"error": "No data loaded"}
        
        overview = {
            'basic_info': {
                'shape': self.data.shape,
                'columns': list(self.data.columns),
                'dtypes': self.data.dtypes.to_dict(),
                'memory_usage': f"{self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
            },
            'missing_data': {
                'total_missing': self.data.isnull().sum().sum(),
                'missing_by_column': self.data.isnull().sum().to_dict(),
                'missing_percentage': (self.data.isnull().sum() / len(self.data) * 100).to_dict()
            },
            'data_quality': {
                'duplicate_rows': self.data.duplicated().sum(),
                'unique_values_per_column': self.data.nunique().to_dict()
            }
        }
        
        return overview
    
    def clean_data(self, options: Dict = None) -> bool:
        """Clean and preprocess the data."""
        if self.data is None:
            print(f"{Fore.RED}❌ No data loaded for cleaning")
            return False
        
        if options is None:
            options = {
                'drop_duplicates': True,
                'handle_missing': 'drop_rows',  # 'drop_rows', 'drop_columns', 'fill_mean', 'fill_median'
                'outlier_method': 'iqr',  # 'iqr', 'zscore', 'none'
                'outlier_threshold': 3
            }
        
        print(f"{Fore.CYAN}🧹 Cleaning data...")
        original_shape = self.data.shape
        
        # Handle duplicates
        if options.get('drop_duplicates', True):
            duplicates_before = self.data.duplicated().sum()
            self.data = self.data.drop_duplicates()
            print(f"{Fore.YELLOW}   Removed {duplicates_before} duplicate rows")
        
        # Handle missing values
        missing_method = options.get('handle_missing', 'drop_rows')
        if missing_method == 'drop_rows':
            self.data = self.data.dropna()
        elif missing_method == 'drop_columns':
            self.data = self.data.dropna(axis=1)
        elif missing_method == 'fill_mean':
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns
            self.data[numeric_columns] = self.data[numeric_columns].fillna(self.data[numeric_columns].mean())
        elif missing_method == 'fill_median':
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns
            self.data[numeric_columns] = self.data[numeric_columns].fillna(self.data[numeric_columns].median())
        
        # Handle outliers
        outlier_method = options.get('outlier_method', 'none')
        if outlier_method != 'none':
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns
            
            for column in numeric_columns:
                if outlier_method == 'iqr':
                    Q1 = self.data[column].quantile(0.25)
                    Q3 = self.data[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    self.data = self.data[(self.data[column] >= lower_bound) & (self.data[column] <= upper_bound)]
                
                elif outlier_method == 'zscore':
                    z_scores = np.abs(stats.zscore(self.data[column]))
                    threshold = options.get('outlier_threshold', 3)
                    self.data = self.data[z_scores < threshold]
        
        new_shape = self.data.shape
        print(f"{Fore.GREEN}✅ Data cleaning completed!")
        print(f"{Fore.YELLOW}   Shape changed from {original_shape} to {new_shape}")
        
        return True
    
    def descriptive_statistics(self) -> Dict:
        """Generate comprehensive descriptive statistics."""
        if self.data is None:
            return {"error": "No data loaded"}
        
        print(f"{Fore.CYAN}📈 Generating descriptive statistics...")
        
        # Basic statistics
        basic_stats = self.data.describe(include='all').to_dict()
        
        # Additional statistics for numeric columns
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        additional_stats = {}
        
        for column in numeric_columns:
            series = self.data[column]
            additional_stats[column] = {
                'skewness': series.skew(),
                'kurtosis': series.kurtosis(),
                'variance': series.var(),
                'range': series.max() - series.min(),
                'iqr': series.quantile(0.75) - series.quantile(0.25),
                'coefficient_of_variation': series.std() / series.mean() if series.mean() != 0 else 0
            }
        
        # Correlation matrix for numeric data
        correlation_matrix = self.data[numeric_columns].corr().to_dict()
        
        results = {
            'basic_statistics': basic_stats,
            'additional_statistics': additional_stats,
            'correlation_matrix': correlation_matrix
        }
        
        self.analysis_results['descriptive_statistics'] = results
        
        # Save to file
        stats_file = self.output_dir / f"descriptive_statistics_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        with open(stats_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"{Fore.GREEN}✅ Descriptive statistics saved to: {stats_file}")
        return results
    
    def create_visualizations(self, chart_types: List[str] = None) -> List[str]:
        """Create various visualizations."""
        if self.data is None:
            print(f"{Fore.RED}❌ No data loaded for visualization")
            return []
        
        if chart_types is None:
            chart_types = ['distribution', 'correlation', 'boxplot', 'scatter_matrix']
        
        print(f"{Fore.CYAN}📊 Creating visualizations...")
        created_files = []
        
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns
        
        # Distribution plots
        if 'distribution' in chart_types and len(numeric_columns) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.ravel()
            
            for i, column in enumerate(numeric_columns[:4]):
                if i < 4:
                    axes[i].hist(self.data[column].dropna(), bins=30, alpha=0.7, edgecolor='black')
                    axes[i].set_title(f'Distribution of {column}')
                    axes[i].set_xlabel(column)
                    axes[i].set_ylabel('Frequency')
            
            plt.tight_layout()
            dist_file = self.output_dir / f"distributions_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
            plt.savefig(dist_file, dpi=300, bbox_inches='tight')
            plt.close()
            created_files.append(str(dist_file))
            print(f"{Fore.GREEN}   ✅ Distribution plots saved")
        
        # Correlation heatmap
        if 'correlation' in chart_types and len(numeric_columns) > 1:
            plt.figure(figsize=(12, 10))
            correlation_matrix = self.data[numeric_columns].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5)
            plt.title('Correlation Matrix')
            plt.tight_layout()
            
            corr_file = self.output_dir / f"correlation_matrix_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
            plt.savefig(corr_file, dpi=300, bbox_inches='tight')
            plt.close()
            created_files.append(str(corr_file))
            print(f"{Fore.GREEN}   ✅ Correlation matrix saved")
        
        # Box plots
        if 'boxplot' in chart_types and len(numeric_columns) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.ravel()
            
            for i, column in enumerate(numeric_columns[:4]):
                if i < 4:
                    axes[i].boxplot(self.data[column].dropna())
                    axes[i].set_title(f'Box Plot of {column}')
                    axes[i].set_ylabel(column)
            
            plt.tight_layout()
            box_file = self.output_dir / f"boxplots_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
            plt.savefig(box_file, dpi=300, bbox_inches='tight')
            plt.close()
            created_files.append(str(box_file))
            print(f"{Fore.GREEN}   ✅ Box plots saved")
        
        # Scatter matrix (if enough numeric columns)
        if 'scatter_matrix' in chart_types and len(numeric_columns) >= 2:
            sample_data = self.data[numeric_columns[:4]].sample(min(1000, len(self.data)))
            pd.plotting.scatter_matrix(sample_data, figsize=(15, 15), alpha=0.6, diagonal='hist')
            plt.suptitle('Scatter Matrix')
            
            scatter_file = self.output_dir / f"scatter_matrix_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
            plt.savefig(scatter_file, dpi=300, bbox_inches='tight')
            plt.close()
            created_files.append(str(scatter_file))
            print(f"{Fore.GREEN}   ✅ Scatter matrix saved")
        
        return created_files
    
    def time_series_analysis(self, date_column: str, value_columns: List[str] = None) -> Dict:
        """Perform time series analysis."""
        if self.data is None:
            return {"error": "No data loaded"}
        
        try:
            print(f"{Fore.CYAN}📅 Performing time series analysis...")
            
            # Convert date column
            self.data[date_column] = pd.to_datetime(self.data[date_column])
            self.data = self.data.sort_values(date_column)
            
            if value_columns is None:
                numeric_columns = self.data.select_dtypes(include=[np.number]).columns
                value_columns = list(numeric_columns[:3])  # Take first 3 numeric columns
            
            results = {}
            
            for column in value_columns:
                if column in self.data.columns:
                    # Create time series
                    ts_data = self.data.set_index(date_column)[column].resample('D').mean()
                    
                    # Seasonal decomposition
                    if len(ts_data) > 24:  # Need at least 2 seasons
                        decomposition = seasonal_decompose(ts_data.dropna(), model='additive', period=7)
                        
                        # Plot decomposition
                        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
                        decomposition.observed.plot(ax=axes[0], title='Original')
                        decomposition.trend.plot(ax=axes[1], title='Trend')
                        decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
                        decomposition.resid.plot(ax=axes[3], title='Residual')
                        
                        plt.tight_layout()
                        ts_file = self.output_dir / f"time_series_{column}_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
                        plt.savefig(ts_file, dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        results[column] = {
                            'trend_strength': float(np.var(decomposition.trend.dropna())),
                            'seasonal_strength': float(np.var(decomposition.seasonal.dropna())),
                            'plot_file': str(ts_file)
                        }
            
            print(f"{Fore.GREEN}✅ Time series analysis completed")
            return results
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error in time series analysis: {e}")
            return {"error": str(e)}
    
    def predictive_modeling(self, target_column: str, feature_columns: List[str] = None, 
                          model_type: str = 'auto') -> Dict:
        """Build predictive models."""
        if self.data is None:
            return {"error": "No data loaded"}
        
        try:
            print(f"{Fore.CYAN}🤖 Building predictive models...")
            
            # Prepare features
            if feature_columns is None:
                numeric_columns = self.data.select_dtypes(include=[np.number]).columns
                feature_columns = [col for col in numeric_columns if col != target_column]
            
            X = self.data[feature_columns]
            y = self.data[target_column]
            
            # Handle missing values
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Determine problem type
            is_classification = len(y.unique()) < 20 and y.dtype == 'object' or len(y.unique()) <= 10
            
            results = {}
            
            if is_classification or model_type == 'classification':
                # Classification models
                if y.dtype == 'object':
                    le = LabelEncoder()
                    y_train_encoded = le.fit_transform(y_train)
                    y_test_encoded = le.transform(y_test)
                else:
                    y_train_encoded, y_test_encoded = y_train, y_test
                
                # Logistic Regression
                log_model = LogisticRegression(random_state=42, max_iter=1000)
                log_model.fit(X_train, y_train_encoded)
                log_pred = log_model.predict(X_test)
                log_accuracy = accuracy_score(y_test_encoded, log_pred)
                
                # Random Forest
                rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
                rf_model.fit(X_train, y_train_encoded)
                rf_pred = rf_model.predict(X_test)
                rf_accuracy = accuracy_score(y_test_encoded, rf_pred)
                
                results = {
                    'problem_type': 'classification',
                    'models': {
                        'logistic_regression': {
                            'accuracy': float(log_accuracy),
                            'feature_importance': dict(zip(feature_columns, log_model.coef_[0] if hasattr(log_model, 'coef_') else []))
                        },
                        'random_forest': {
                            'accuracy': float(rf_accuracy),
                            'feature_importance': dict(zip(feature_columns, rf_model.feature_importances_))
                        }
                    }
                }
                
            else:
                # Regression models
                # Linear Regression
                lin_model = LinearRegression()
                lin_model.fit(X_train, y_train)
                lin_pred = lin_model.predict(X_test)
                lin_mse = mean_squared_error(y_test, lin_pred)
                lin_r2 = lin_model.score(X_test, y_test)
                
                # Random Forest Regression
                rf_model = RandomForestRegressor(random_state=42, n_estimators=100)
                rf_model.fit(X_train, y_train)
                rf_pred = rf_model.predict(X_test)
                rf_mse = mean_squared_error(y_test, rf_pred)
                rf_r2 = rf_model.score(X_test, y_test)
                
                results = {
                    'problem_type': 'regression',
                    'models': {
                        'linear_regression': {
                            'mse': float(lin_mse),
                            'r2_score': float(lin_r2),
                            'coefficients': dict(zip(feature_columns, lin_model.coef_))
                        },
                        'random_forest': {
                            'mse': float(rf_mse),
                            'r2_score': float(rf_r2),
                            'feature_importance': dict(zip(feature_columns, rf_model.feature_importances_))
                        }
                    }
                }
            
            self.analysis_results['predictive_modeling'] = results
            print(f"{Fore.GREEN}✅ Predictive modeling completed")
            return results
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error in predictive modeling: {e}")
            return {"error": str(e)}
    
    def generate_report(self, report_format: str = 'html') -> str:
        """Generate comprehensive analysis report."""
        if not self.analysis_results:
            print(f"{Fore.RED}❌ No analysis results available for report generation")
            return ""
        
        print(f"{Fore.CYAN}📄 Generating {report_format.upper()} report...")
        
        # Prepare report data
        report_data = {
            'generated_at': datetime.now().isoformat(),
            'data_shape': self.data.shape if self.data is not None else (0, 0),
            'analysis_results': self.analysis_results,
            'data_overview': self.get_data_overview() if self.data is not None else {}
        }
        
        if report_format.lower() == 'html':
            return self._generate_html_report(report_data)
        elif report_format.lower() == 'pdf':
            return self._generate_pdf_report(report_data)
        else:
            # JSON fallback
            report_file = self.output_dir / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            print(f"{Fore.GREEN}✅ JSON report saved to: {report_file}")
            return str(report_file)
    
    def _generate_html_report(self, report_data: Dict) -> str:
        """Generate HTML report."""
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Data Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        .header { background-color: #f4f4f4; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
        .section { margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .stat-item { background-color: #f9f9f9; padding: 15px; border-radius: 3px; text-align: center; }
        .stat-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
        .stat-label { color: #7f8c8d; font-size: 14px; }
        table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .error { color: #e74c3c; background-color: #fdf2f2; padding: 10px; border-radius: 5px; }
        h1, h2, h3 { color: #2c3e50; }
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Data Analysis Report</h1>
        <p><strong>Generated:</strong> {{ generated_at }}</p>
        <p><strong>Dataset:</strong> {{ data_shape[0]:,}} rows × {{ data_shape[1] }} columns</p>
    </div>

    {% if data_overview.basic_info %}
    <div class="section">
        <h2>📈 Data Overview</h2>
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-value">{{ data_shape[0]:,}}</div>
                <div class="stat-label">Total Rows</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{{ data_shape[1] }}</div>
                <div class="stat-label">Columns</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{{ data_overview.missing_data.total_missing }}</div>
                <div class="stat-label">Missing Values</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{{ data_overview.data_quality.duplicate_rows }}</div>
                <div class="stat-label">Duplicate Rows</div>
            </div>
        </div>
    </div>
    {% endif %}

    {% if analysis_results.descriptive_statistics %}
    <div class="section">
        <h2>📊 Descriptive Statistics</h2>
        <p>Statistical summary of numeric columns in the dataset.</p>
        <!-- Additional statistics would be formatted here -->
    </div>
    {% endif %}

    {% if analysis_results.predictive_modeling %}
    <div class="section">
        <h2>🤖 Predictive Modeling Results</h2>
        <p><strong>Problem Type:</strong> {{ analysis_results.predictive_modeling.problem_type|title }}</p>
        
        {% for model_name, model_results in analysis_results.predictive_modeling.models.items() %}
        <h3>{{ model_name|replace('_', ' ')|title }}</h3>
        {% if model_results.accuracy %}
            <p><strong>Accuracy:</strong> {{ "%.3f"|format(model_results.accuracy) }}</p>
        {% endif %}
        {% if model_results.r2_score %}
            <p><strong>R² Score:</strong> {{ "%.3f"|format(model_results.r2_score) }}</p>
        {% endif %}
        {% if model_results.mse %}
            <p><strong>Mean Squared Error:</strong> {{ "%.3f"|format(model_results.mse) }}</p>
        {% endif %}
        {% endfor %}
    </div>
    {% endif %}

    <div class="section">
        <h2>📄 Summary</h2>
        <p>This report was automatically generated by the Data Analysis Suite. 
           It provides comprehensive insights into your dataset including descriptive statistics, 
           data quality assessment, and predictive modeling results where applicable.</p>
    </div>
</body>
</html>
        """
        
        template = Template(html_template)
        html_content = template.render(**report_data)
        
        report_file = self.output_dir / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"{Fore.GREEN}✅ HTML report saved to: {report_file}")
        return str(report_file)
    
    def _generate_pdf_report(self, report_data: Dict) -> str:
        """Generate PDF report using FPDF."""
        class PDF(FPDF):
            def header(self):
                self.set_font('Arial', 'B', 15)
                self.cell(0, 10, 'Data Analysis Report', 0, 1, 'C')
                self.ln(10)
            
            def footer(self):
                self.set_y(-15)
                self.set_font('Arial', 'I', 8)
                self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        
        pdf = PDF()
        pdf.add_page()
        pdf.set_font('Arial', '', 12)
        
        # Add content
        pdf.cell(0, 10, f"Generated: {report_data['generated_at']}", 0, 1)
        pdf.cell(0, 10, f"Dataset: {report_data['data_shape'][0]:,} rows × {report_data['data_shape'][1]} columns", 0, 1)
        pdf.ln(10)
        
        # Add analysis results summary
        if report_data['analysis_results']:
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, 'Analysis Summary', 0, 1)
            pdf.set_font('Arial', '', 12)
            
            for analysis_type, results in report_data['analysis_results'].items():
                pdf.cell(0, 10, f"- {analysis_type.replace('_', ' ').title()}: Completed", 0, 1)
        
        report_file = self.output_dir / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        pdf.output(str(report_file))
        
        print(f"{Fore.GREEN}✅ PDF report saved to: {report_file}")
        return str(report_file)

def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description='Data Analysis Suite')
    parser.add_argument('--file', '-f', help='Input data file (CSV, Excel, JSON)', required=True)
    parser.add_argument('--clean', '-c', action='store_true', help='Clean data before analysis')
    parser.add_argument('--stats', '-s', action='store_true', help='Generate descriptive statistics')
    parser.add_argument('--visualize', '-v', action='store_true', help='Create visualizations')
    parser.add_argument('--predict', '-p', help='Target column for predictive modeling')
    parser.add_argument('--timeseries', '-t', help='Date column for time series analysis')
    parser.add_argument('--report', '-r', choices=['html', 'pdf', 'json'], default='html', 
                       help='Generate analysis report')
    parser.add_argument('--output', '-o', help='Output directory', default='analysis_output')
    
    args = parser.parse_args()
    
    # Initialize suite
    suite = DataAnalysisSuite()
    suite.print_banner()
    
    # Set output directory
    if args.output:
        suite.output_dir = Path(args.output)
        suite.output_dir.mkdir(exist_ok=True)
    
    # Load data
    if not suite.load_data(args.file):
        return
    
    # Clean data if requested
    if args.clean:
        suite.clean_data()
    
    # Generate descriptive statistics
    if args.stats:
        suite.descriptive_statistics()
    
    # Create visualizations
    if args.visualize:
        suite.create_visualizations()
    
    # Time series analysis
    if args.timeseries:
        suite.time_series_analysis(args.timeseries)
    
    # Predictive modeling
    if args.predict:
        suite.predictive_modeling(args.predict)
    
    # Generate report
    if args.report:
        suite.generate_report(args.report)
    
    print(f"\n{Fore.CYAN}🎉 Analysis completed! Check the output directory for results.")

if __name__ == "__main__":
    main()
