#!/usr/bin/env python3
"""
Advanced Data Analysis Suite
The most comprehensive data analysis toolkit with AI-powered insights.
"""

import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple, Union
import asyncio
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings('ignore')

# Core imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Scientific computing
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, MinMaxScaler
from sklearn.model_selection import (train_test_split, cross_val_score, 
                                   GridSearchCV, RandomizedSearchCV, StratifiedKFold)
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                             GradientBoostingRegressor, GradientBoostingClassifier,
                             ExtraTreesRegressor, ExtraTreesClassifier)
from sklearn.linear_model import (LinearRegression, LogisticRegression, Ridge, 
                                Lasso, ElasticNet, BayesianRidge)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.manifold import TSNE, MDS
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                           accuracy_score, precision_score, recall_score, f1_score,
                           classification_report, confusion_matrix, roc_auc_score,
                           silhouette_score, calinski_harabasz_score)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
from sklearn.impute import SimpleImputer, KNNImputer

# Statistical analysis
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.stattools import durbin_watson, jarque_bera

# Utilities
from jinja2 import Template
from colorama import init, Fore, Style
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.text import Text
from tqdm import tqdm
import typer

# Initialize
init(autoreset=True)
console = Console()

class AdvancedDataAnalysisSuite:
    """The most advanced data analysis and machine learning toolkit."""
    
    def __init__(self, config: Dict = None):
        self.data = None
        self.original_data = None
        self.processed_data = None
        self.analysis_results = {}
        self.model_results = {}
        self.insights = []
        
        # Configuration
        self.config = config or {
            'output_dir': 'analysis_output',
            'theme': 'modern',
            'auto_insights': True,
            'parallel_processing': True,
            'cache_results': True
        }
        
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(exist_ok=True)
        
        # Advanced styling
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        self.models = {
            'regression': {
                'linear': LinearRegression(),
                'ridge': Ridge(alpha=1.0),
                'lasso': Lasso(alpha=1.0),
                'elastic_net': ElasticNet(alpha=1.0, l1_ratio=0.5),
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(random_state=42),
                'svr': SVR(kernel='rbf'),
                'knn': KNeighborsRegressor(n_neighbors=5),
                'decision_tree': DecisionTreeRegressor(random_state=42),
                'extra_trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
                'mlp': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
            },
            'classification': {
                'logistic': LogisticRegression(random_state=42, max_iter=1000),
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingClassifier(random_state=42),
                'svc': SVC(probability=True, random_state=42),
                'knn': KNeighborsClassifier(n_neighbors=5),
                'decision_tree': DecisionTreeClassifier(random_state=42),
                'naive_bayes': GaussianNB(),
                'extra_trees': ExtraTreesClassifier(n_estimators=100, random_state=42),
                'mlp': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
            }
        }
    
    def display_banner(self):
        """Display an impressive welcome banner."""
        banner_text = """
╔══════════════════════════════════════════════════════════════╗
║                 🚀 ADVANCED DATA ANALYSIS SUITE 🚀           ║
║                                                              ║
║  ┌─ AI-Powered Analytics      ┌─ Advanced Visualizations   ║
║  ├─ AutoML & Model Selection  ├─ Statistical Testing       ║
║  ├─ Real-time Insights        ├─ Time Series Analysis      ║
║  ├─ Interactive Dashboards    ├─ Anomaly Detection         ║
║  └─ Performance Optimization  └─ Automated Reporting       ║
║                                                              ║
║              Your gateway to data science excellence         ║
╚══════════════════════════════════════════════════════════════╝
        """
        
        console.print(Panel(
            banner_text,
            title="[bold blue]Welcome[/bold blue]",
            border_style="bright_blue",
            expand=False
        ))
    
    def load_data(self, file_path: str, **kwargs) -> bool:
        """Advanced data loading with automatic format detection and optimization."""
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading data...", total=None)
            
            try:
                file_path = Path(file_path)
                
                # Auto-detect format and optimize loading
                if file_path.suffix.lower() == '.csv':
                    # Optimize CSV loading
                    self.data = pd.read_csv(file_path, **kwargs)
                elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                    self.data = pd.read_excel(file_path, **kwargs)
                elif file_path.suffix.lower() == '.json':
                    self.data = pd.read_json(file_path, **kwargs)
                elif file_path.suffix.lower() == '.parquet':
                    self.data = pd.read_parquet(file_path, **kwargs)
                elif file_path.suffix.lower() == '.feather':
                    self.data = pd.read_feather(file_path, **kwargs)
                else:
                    raise ValueError(f"Unsupported file format: {file_path.suffix}")
                
                self.original_data = self.data.copy()
                
                # Display success message with data info
                info_table = Table(title="Data Loading Summary")
                info_table.add_column("Metric", style="cyan")
                info_table.add_column("Value", style="green")
                
                info_table.add_row("File", str(file_path.name))
                info_table.add_row("Rows", f"{self.data.shape[0]:,}")
                info_table.add_row("Columns", str(self.data.shape[1]))
                info_table.add_row("Memory Usage", f"{self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                info_table.add_row("Data Types", str(len(self.data.dtypes.unique())))
                
                console.print(info_table)
                
                if self.config['auto_insights']:
                    self._generate_loading_insights()
                
                return True
                
            except Exception as e:
                console.print(f"[red]❌ Error loading data: {e}[/red]")
                return False
    
    def _generate_loading_insights(self):
        """Generate automatic insights upon data loading."""
        insights = []
        
        # Data shape insights
        if self.data.shape[0] > 100000:
            insights.append("🔍 Large dataset detected - consider sampling for initial exploration")
        
        # Missing data insights
        missing_pct = (self.data.isnull().sum().sum() / (self.data.shape[0] * self.data.shape[1])) * 100
        if missing_pct > 10:
            insights.append(f"⚠️  High missing data percentage: {missing_pct:.1f}%")
        
        # Data type insights
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        
        if len(numeric_cols) == 0:
            insights.append("📊 No numeric columns found - consider feature engineering")
        if len(categorical_cols) > len(numeric_cols):
            insights.append("🏷️  More categorical than numeric features - encoding may be needed")
        
        self.insights.extend(insights)
        
        if insights:
            console.print("\n[yellow]💡 Automatic Insights:[/yellow]")
            for insight in insights:
                console.print(f"   {insight}")
    
    def comprehensive_eda(self) -> Dict:
        """Perform comprehensive exploratory data analysis."""
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            task = progress.add_task("Analyzing data structure...", total=7)
            
            results = {}
            
            # 1. Basic Information
            progress.update(task, description="Computing basic statistics...")
            results['basic_info'] = self._get_basic_info()
            progress.advance(task)
            
            # 2. Data Quality Assessment
            progress.update(task, description="Assessing data quality...")
            results['data_quality'] = self._assess_data_quality()
            progress.advance(task)
            
            # 3. Statistical Summary
            progress.update(task, description="Computing statistical summary...")
            results['statistical_summary'] = self._get_statistical_summary()
            progress.advance(task)
            
            # 4. Correlation Analysis
            progress.update(task, description="Analyzing correlations...")
            results['correlation_analysis'] = self._analyze_correlations()
            progress.advance(task)
            
            # 5. Distribution Analysis
            progress.update(task, description="Analyzing distributions...")
            results['distribution_analysis'] = self._analyze_distributions()
            progress.advance(task)
            
            # 6. Outlier Detection
            progress.update(task, description="Detecting outliers...")
            results['outlier_analysis'] = self._detect_outliers()
            progress.advance(task)
            
            # 7. Feature Analysis
            progress.update(task, description="Analyzing features...")
            results['feature_analysis'] = self._analyze_features()
            progress.advance(task)
        
        self.analysis_results['eda'] = results
        
        # Generate insights
        self._generate_eda_insights(results)
        
        # Save results
        self._save_analysis_results('eda', results)
        
        return results
    
    def _get_basic_info(self) -> Dict:
        """Get comprehensive basic information about the dataset."""
        return {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).to_dict(),
            'total_memory_mb': self.data.memory_usage(deep=True).sum() / 1024**2,
            'numeric_columns': list(self.data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.data.select_dtypes(include=['object', 'category']).columns),
            'datetime_columns': list(self.data.select_dtypes(include=['datetime64']).columns)
        }
    
    def _assess_data_quality(self) -> Dict:
        """Comprehensive data quality assessment."""
        quality_metrics = {}
        
        # Missing data analysis
        missing_counts = self.data.isnull().sum()
        missing_percentages = (missing_counts / len(self.data)) * 100
        
        quality_metrics['missing_data'] = {
            'total_missing': missing_counts.sum(),
            'missing_by_column': missing_counts.to_dict(),
            'missing_percentage_by_column': missing_percentages.to_dict(),
            'columns_with_missing': missing_counts[missing_counts > 0].index.tolist(),
            'complete_rows': len(self.data) - self.data.isnull().any(axis=1).sum()
        }
        
        # Duplicate analysis
        duplicates = self.data.duplicated()
        quality_metrics['duplicates'] = {
            'total_duplicates': duplicates.sum(),
            'duplicate_percentage': (duplicates.sum() / len(self.data)) * 100,
            'unique_rows': len(self.data) - duplicates.sum()
        }
        
        # Data consistency checks
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        quality_metrics['consistency'] = {}
        
        for col in numeric_cols:
            if len(numeric_cols) > 0:
                quality_metrics['consistency'][col] = {
                    'infinite_values': np.isinf(self.data[col]).sum(),
                    'zero_values': (self.data[col] == 0).sum(),
                    'negative_values': (self.data[col] < 0).sum() if self.data[col].dtype in ['int64', 'float64'] else 0
                }
        
        return quality_metrics
    
    def _get_statistical_summary(self) -> Dict:
        """Enhanced statistical summary with advanced metrics."""
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return {'error': 'No numeric columns found'}
        
        summary = {}
        
        # Basic statistics
        summary['basic_stats'] = numeric_data.describe().to_dict()
        
        # Advanced statistics
        summary['advanced_stats'] = {}
        for col in numeric_data.columns:
            series = numeric_data[col].dropna()
            summary['advanced_stats'][col] = {
                'skewness': float(series.skew()),
                'kurtosis': float(series.kurtosis()),
                'variance': float(series.var()),
                'coefficient_of_variation': float(series.std() / series.mean()) if series.mean() != 0 else 0,
                'median_absolute_deviation': float((series - series.median()).abs().median()),
                'interquartile_range': float(series.quantile(0.75) - series.quantile(0.25)),
                'range': float(series.max() - series.min()),
                'unique_values': int(series.nunique()),
                'mode': float(series.mode().iloc[0]) if not series.mode().empty else None
            }
        
        return summary
    
    def _analyze_correlations(self) -> Dict:
        """Advanced correlation analysis."""
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        if numeric_data.shape[1] < 2:
            return {'error': 'Need at least 2 numeric columns for correlation analysis'}
        
        correlations = {}
        
        # Pearson correlation
        correlations['pearson'] = numeric_data.corr().to_dict()
        
        # Spearman correlation
        correlations['spearman'] = numeric_data.corr(method='spearman').to_dict()
        
        # Find highly correlated pairs
        corr_matrix = numeric_data.corr().abs()
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.8:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': float(corr_matrix.iloc[i, j])
                    })
        
        correlations['high_correlations'] = high_corr_pairs
        
        return correlations
    
    def _analyze_distributions(self) -> Dict:
        """Analyze distributions of numeric variables."""
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        distributions = {}
        
        for col in numeric_data.columns:
            series = numeric_data[col].dropna()
            
            # Normality tests
            shapiro_stat, shapiro_p = stats.shapiro(series.sample(min(5000, len(series))))
            
            distributions[col] = {
                'distribution_type': self._identify_distribution(series),
                'normality_test': {
                    'shapiro_wilk_statistic': float(shapiro_stat),
                    'shapiro_wilk_p_value': float(shapiro_p),
                    'is_normal': shapiro_p > 0.05
                },
                'histogram_bins': 'auto'
            }
        
        return distributions
    
    def _identify_distribution(self, series: pd.Series) -> str:
        """Identify the likely distribution type."""
        # Simple heuristics for distribution identification
        skewness = series.skew()
        kurtosis = series.kurtosis()
        
        if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
            return 'normal'
        elif skewness > 1:
            return 'right_skewed'
        elif skewness < -1:
            return 'left_skewed'
        elif kurtosis > 3:
            return 'heavy_tailed'
        else:
            return 'unknown'
    
    def _detect_outliers(self) -> Dict:
        """Advanced outlier detection using multiple methods."""
        numeric_data = self.data.select_dtypes(include=[np.number])
        outliers = {}
        
        for col in numeric_data.columns:
            series = numeric_data[col].dropna()
            
            # IQR method
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = series[(series < lower_bound) | (series > upper_bound)]
            
            # Z-score method
            z_scores = np.abs(stats.zscore(series))
            z_outliers = series[z_scores > 3]
            
            # Modified Z-score method
            median = series.median()
            mad = (series - median).abs().median()
            modified_z_scores = 0.6745 * (series - median) / mad
            modified_z_outliers = series[np.abs(modified_z_scores) > 3.5]
            
            outliers[col] = {
                'iqr_outliers': {
                    'count': len(iqr_outliers),
                    'percentage': (len(iqr_outliers) / len(series)) * 100,
                    'bounds': {'lower': float(lower_bound), 'upper': float(upper_bound)}
                },
                'z_score_outliers': {
                    'count': len(z_outliers),
                    'percentage': (len(z_outliers) / len(series)) * 100
                },
                'modified_z_score_outliers': {
                    'count': len(modified_z_outliers),
                    'percentage': (len(modified_z_outliers) / len(series)) * 100
                }
            }
        
        return outliers
    
    def _analyze_features(self) -> Dict:
        """Advanced feature analysis."""
        features = {}
        
        # Numeric features
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        features['numeric_features'] = {
            'count': len(numeric_cols),
            'columns': list(numeric_cols),
            'high_cardinality': [col for col in numeric_cols if self.data[col].nunique() > 0.9 * len(self.data)],
            'low_variance': [col for col in numeric_cols if self.data[col].var() < 0.01]
        }
        
        # Categorical features
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        features['categorical_features'] = {
            'count': len(categorical_cols),
            'columns': list(categorical_cols),
            'high_cardinality': [col for col in categorical_cols if self.data[col].nunique() > 50],
            'binary_features': [col for col in categorical_cols if self.data[col].nunique() == 2]
        }
        
        return features
    
    def _generate_eda_insights(self, results: Dict):
        """Generate actionable insights from EDA results."""
        insights = []
        
        # Data quality insights
        quality = results['data_quality']
        if quality['missing_data']['total_missing'] > 0:
            insights.append(f"📊 Found {quality['missing_data']['total_missing']} missing values across dataset")
        
        if quality['duplicates']['total_duplicates'] > 0:
            insights.append(f"🔄 {quality['duplicates']['total_duplicates']} duplicate rows detected")
        
        # Correlation insights
        if 'correlation_analysis' in results and 'high_correlations' in results['correlation_analysis']:
            high_corr = results['correlation_analysis']['high_correlations']
            if high_corr:
                insights.append(f"🔗 {len(high_corr)} highly correlated feature pairs found")
        
        # Outlier insights
        outlier_results = results['outlier_analysis']
        total_outliers = sum([data['iqr_outliers']['count'] for data in outlier_results.values()])
        if total_outliers > 0:
            insights.append(f"⚠️  {total_outliers} outliers detected using IQR method")
        
        self.insights.extend(insights)
    
    def automl_model_selection(self, target_column: str, problem_type: str = 'auto', 
                             test_size: float = 0.2, cv_folds: int = 5) -> Dict:
        """Automated machine learning model selection and evaluation."""
        
        if self.data is None:
            return {'error': 'No data loaded'}
        
        if target_column not in self.data.columns:
            return {'error': f'Target column {target_column} not found'}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            task = progress.add_task("Preparing data for ML...", total=6)
            
            # Prepare features and target
            X = self.data.drop(columns=[target_column])
            y = self.data[target_column]
            
            # Handle missing values
            X = self._preprocess_features(X)
            y = y.dropna()
            X = X.loc[y.index]  # Align indices
            
            progress.advance(task)
            
            # Determine problem type
            if problem_type == 'auto':
                if y.dtype == 'object' or y.nunique() < 20:
                    problem_type = 'classification'
                else:
                    problem_type = 'regression'
            
            progress.update(task, description=f"Running {problem_type} models...")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, 
                stratify=y if problem_type == 'classification' and y.nunique() > 1 else None
            )
            
            progress.advance(task)
            
            # Select models based on problem type
            models_to_test = self.models[problem_type]
            results = {}
            
            progress.update(task, description="Training and evaluating models...")
            
            for model_name, model in models_to_test.items():
                try:
                    # Cross-validation
                    cv_scores = cross_val_score(
                        model, X_train, y_train, cv=cv_folds,
                        scoring='accuracy' if problem_type == 'classification' else 'r2'
                    )
                    
                    # Fit model
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    if problem_type == 'classification':
                        metrics = {
                            'accuracy': accuracy_score(y_test, y_pred),
                            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
                        }
                        
                        if len(np.unique(y)) == 2:  # Binary classification
                            try:
                                y_pred_proba = model.predict_proba(X_test)[:, 1]
                                metrics['auc_roc'] = roc_auc_score(y_test, y_pred_proba)
                            except:
                                metrics['auc_roc'] = None
                    
                    else:  # Regression
                        metrics = {
                            'r2_score': r2_score(y_test, y_pred),
                            'mse': mean_squared_error(y_test, y_pred),
                            'mae': mean_absolute_error(y_test, y_pred),
                            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
                        }
                    
                    # Feature importance
                    feature_importance = None
                    if hasattr(model, 'feature_importances_'):
                        feature_importance = dict(zip(X.columns, model.feature_importances_))
                    elif hasattr(model, 'coef_'):
                        feature_importance = dict(zip(X.columns, 
                                                    model.coef_ if len(model.coef_.shape) == 1 else model.coef_[0]))
                    
                    results[model_name] = {
                        'metrics': metrics,
                        'cv_scores': {
                            'mean': float(cv_scores.mean()),
                            'std': float(cv_scores.std()),
                            'scores': cv_scores.tolist()
                        },
                        'feature_importance': feature_importance,
                        'model_object': model
                    }
                    
                except Exception as e:
                    results[model_name] = {'error': str(e)}
                
                progress.advance(task, advance=0.7)
            
            progress.update(task, description="Selecting best model...")
            
            # Find best model
            best_model = self._select_best_model(results, problem_type)
            
            final_results = {
                'problem_type': problem_type,
                'target_column': target_column,
                'models': results,
                'best_model': best_model,
                'data_shape': {'train': X_train.shape, 'test': X_test.shape},
                'preprocessing_info': self._get_preprocessing_info(X)
            }
            
            progress.advance(task)
            
            self.model_results[target_column] = final_results
            self._save_analysis_results('automl', final_results)
            
            return final_results
    
    def _preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Advanced feature preprocessing."""
        X_processed = X.copy()
        
        # Handle numeric columns
        numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Fill missing values with median
            imputer = SimpleImputer(strategy='median')
            X_processed[numeric_cols] = imputer.fit_transform(X_processed[numeric_cols])
        
        # Handle categorical columns
        categorical_cols = X_processed.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            # Fill missing values with mode
            mode_value = X_processed[col].mode().iloc[0] if not X_processed[col].mode().empty else 'Unknown'
            X_processed[col] = X_processed[col].fillna(mode_value)
            
            # Label encoding for now (could be enhanced with one-hot encoding)
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))
        
        return X_processed
    
    def _select_best_model(self, results: Dict, problem_type: str) -> Dict:
        """Select the best performing model."""
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            return {'error': 'No valid model results'}
        
        if problem_type == 'classification':
            # Sort by accuracy
            best_model_name = max(valid_results.keys(), 
                                key=lambda x: valid_results[x]['metrics']['accuracy'])
        else:
            # Sort by R² score
            best_model_name = max(valid_results.keys(), 
                                key=lambda x: valid_results[x]['metrics']['r2_score'])
        
        return {
            'name': best_model_name,
            'results': valid_results[best_model_name]
        }
    
    def _get_preprocessing_info(self, X: pd.DataFrame) -> Dict:
        """Get information about preprocessing steps."""
        return {
            'numeric_columns': list(X.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(X.select_dtypes(include=['object', 'category']).columns),
            'total_features': X.shape[1],
            'preprocessing_steps': ['missing_value_imputation', 'label_encoding']
        }
    
    def create_advanced_visualizations(self, viz_types: List[str] = None) -> List[str]:
        """Create advanced, publication-ready visualizations."""
        
        if self.data is None:
            console.print("[red]❌ No data loaded for visualization[/red]")
            return []
        
        if viz_types is None:
            viz_types = ['correlation_heatmap', 'distribution_plots', 'pairplot', 
                        'feature_importance', 'interactive_scatter']
        
        created_files = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            task = progress.add_task("Creating visualizations...", total=len(viz_types))
            
            for viz_type in viz_types:
                progress.update(task, description=f"Creating {viz_type}...")
                
                try:
                    if viz_type == 'correlation_heatmap':
                        file_path = self._create_correlation_heatmap()
                        if file_path:
                            created_files.append(file_path)
                    
                    elif viz_type == 'distribution_plots':
                        file_path = self._create_distribution_plots()
                        if file_path:
                            created_files.append(file_path)
                    
                    elif viz_type == 'pairplot':
                        file_path = self._create_advanced_pairplot()
                        if file_path:
                            created_files.append(file_path)
                    
                    elif viz_type == 'interactive_scatter':
                        file_path = self._create_interactive_scatter()
                        if file_path:
                            created_files.append(file_path)
                
                except Exception as e:
                    console.print(f"[yellow]⚠️  Could not create {viz_type}: {e}[/yellow]")
                
                progress.advance(task)
        
        console.print(f"[green]✅ Created {len(created_files)} visualizations[/green]")
        return created_files
    
    def _create_correlation_heatmap(self) -> Optional[str]:
        """Create an advanced correlation heatmap."""
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        if numeric_data.shape[1] < 2:
            return None
        
        # Create correlation matrix
        corr_matrix = numeric_data.corr()
        
        # Create the plot
        plt.figure(figsize=(14, 10))
        
        # Custom colormap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, fmt='.2f')
        
        plt.title('Advanced Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        file_path = self.output_dir / f"correlation_heatmap_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(file_path)
    
    def _create_distribution_plots(self) -> Optional[str]:
        """Create advanced distribution plots."""
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return None
        
        n_cols = min(4, len(numeric_data.columns))
        n_rows = (len(numeric_data.columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, column in enumerate(numeric_data.columns):
            if i < len(axes):
                # Histogram with KDE
                sns.histplot(data=numeric_data, x=column, kde=True, ax=axes[i], alpha=0.7)
                axes[i].set_title(f'Distribution of {column}', fontweight='bold')
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(numeric_data.columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Feature Distributions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        file_path = self.output_dir / f"distributions_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(file_path)
    
    def _create_advanced_pairplot(self) -> Optional[str]:
        """Create an advanced pairplot with statistical information."""
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        if numeric_data.shape[1] < 2:
            return None
        
        # Sample data if too large
        if len(numeric_data) > 1000:
            numeric_data = numeric_data.sample(1000, random_state=42)
        
        # Create pairplot
        plt.figure(figsize=(12, 10))
        
        # Select top 5 numeric columns by variance
        top_cols = numeric_data.var().nlargest(5).index.tolist()
        subset_data = numeric_data[top_cols]
        
        pairplot = sns.pairplot(subset_data, diag_kind='kde', plot_kws={'alpha': 0.6})
        pairplot.fig.suptitle('Advanced Pairplot Analysis', y=1.02, fontsize=16, fontweight='bold')
        
        file_path = self.output_dir / f"pairplot_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(file_path)
    
    def _create_interactive_scatter(self) -> Optional[str]:
        """Create an interactive scatter plot using Plotly."""
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        if numeric_data.shape[1] < 2:
            return None
        
        # Select two columns with highest variance
        top_cols = numeric_data.var().nlargest(2).index.tolist()
        
        fig = px.scatter(
            self.data, 
            x=top_cols[0], 
            y=top_cols[1],
            title=f'Interactive Scatter Plot: {top_cols[0]} vs {top_cols[1]}',
            hover_data=numeric_data.columns[:3].tolist(),
            opacity=0.7
        )
        
        fig.update_layout(
            title_font_size=16,
            showlegend=True,
            height=600
        )
        
        file_path = self.output_dir / f"interactive_scatter_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
        fig.write_html(file_path)
        
        return str(file_path)
    
    def _save_analysis_results(self, analysis_type: str, results: Dict):
        """Save analysis results to JSON file."""
        filename = f"{analysis_type}_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        file_path = self.output_dir / filename
        
        # Convert numpy types to Python types for JSON serialization
        json_results = self._convert_for_json(results)
        
        with open(file_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        console.print(f"[green]💾 Results saved to {filename}[/green]")
    
    def _convert_for_json(self, obj):
        """Convert numpy types to JSON-serializable types."""
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                             np.int16, np.int32, np.int64, np.uint8,
                             np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    def generate_insights_report(self) -> str:
        """Generate a comprehensive insights report."""
        
        console.print("[cyan]📊 Generating comprehensive insights report...[/cyan]")
        
        # HTML template for the report
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Advanced Data Analysis Report</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f8f9fa; }
        .container { max-width: 1200px; margin: 0 auto; background: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px 10px 0 0; }
        .header h1 { margin: 0; font-size: 2.5em; font-weight: 300; }
        .header p { margin: 10px 0 0 0; opacity: 0.9; }
        .content { padding: 30px; }
        .section { margin-bottom: 40px; }
        .section h2 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        .insights-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }
        .insight-card { background: #f8f9fa; border-left: 4px solid #3498db; padding: 20px; border-radius: 5px; }
        .insight-card h3 { margin-top: 0; color: #2c3e50; }
        .metric { display: inline-block; background: #e3f2fd; padding: 8px 16px; margin: 5px; border-radius: 20px; font-weight: bold; }
        .warning { border-left-color: #f39c12; background: #fef9e7; }
        .success { border-left-color: #27ae60; background: #eafaf1; }
        .info { border-left-color: #3498db; background: #ebf3fd; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; font-weight: 600; }
        .summary-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .stat-box { background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; }
        .stat-value { font-size: 2em; font-weight: bold; margin-bottom: 5px; }
        .stat-label { opacity: 0.9; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Advanced Data Analysis Report</h1>
            <p>Generated on {{ timestamp }}</p>
            <p>Dataset: {{ "{:,}".format(data_shape[0]) }} rows × {{ data_shape[1] }} columns</p>
        </div>
        
        <div class="content">
            <div class="section">
                <h2>📈 Executive Summary</h2>
                <div class="summary-stats">
                    <div class="stat-box">
                        <div class="stat-value">{{ "{:,}".format(data_shape[0]) }}</div>
                        <div class="stat-label">Total Records</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{{ data_shape[1] }}</div>
                        <div class="stat-label">Features</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{{ insights_count }}</div>
                        <div class="stat-label">Key Insights</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{{ analyses_completed }}</div>
                        <div class="stat-label">Analyses Completed</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>💡 Key Insights</h2>
                <div class="insights-grid">
                    {% for insight in insights %}
                    <div class="insight-card {{ insight.type }}">
                        <h3>{{ insight.title }}</h3>
                        <p>{{ insight.description }}</p>
                    </div>
                    {% endfor %}
                </div>
            </div>
            
            {% if model_results %}
            <div class="section">
                <h2>🤖 Machine Learning Results</h2>
                {% for target, results in model_results.items() %}
                <div class="insight-card success">
                    <h3>Target: {{ target }}</h3>
                    <p><strong>Problem Type:</strong> {{ results.problem_type|title }}</p>
                    <p><strong>Best Model:</strong> {{ results.best_model.name|title }}</p>
                    {% if results.problem_type == 'classification' %}
                    <p><strong>Accuracy:</strong> {{ "%.3f"|format(results.best_model.results.metrics.accuracy) }}</p>
                    {% else %}
                    <p><strong>R² Score:</strong> {{ "%.3f"|format(results.best_model.results.metrics.r2_score) }}</p>
                    {% endif %}
                </div>
                {% endfor %}
            </div>
            {% endif %}
            
            <div class="section">
                <h2>📊 Data Quality Assessment</h2>
                <div class="insights-grid" id="quality-metrics">
                    <!-- Quality metrics will be populated here -->
                </div>
            </div>
            
            <div class="section">
                <h2>🔍 Recommendations</h2>
                <div class="insight-card info">
                    <h3>Next Steps</h3>
                    <ul>
                        {% for recommendation in recommendations %}
                        <li>{{ recommendation }}</li>
                        {% endfor %}
                    </ul>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
        """
        
        # Prepare data for the template
        report_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_shape': self.data.shape if self.data is not None else (0, 0),
            'insights_count': len(self.insights),
            'analyses_completed': len(self.analysis_results),
            'insights': self._format_insights_for_report(),
            'model_results': self.model_results,
            'recommendations': self._generate_recommendations()
        }
        
        # Render template
        template = Template(html_template)
        html_content = template.render(**report_data)
        
        # Save report
        report_path = self.output_dir / f"insights_report_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        console.print(f"[green]📄 Comprehensive report saved to: {report_path}[/green]")
        return str(report_path)
    
    def _format_insights_for_report(self) -> List[Dict]:
        """Format insights for HTML report."""
        formatted_insights = []
        
        for insight in self.insights:
            insight_type = 'info'
            if '⚠️' in insight or 'warning' in insight.lower():
                insight_type = 'warning'
            elif '✅' in insight or 'success' in insight.lower():
                insight_type = 'success'
            
            formatted_insights.append({
                'title': insight.split(' ', 1)[0],  # First emoji/word as title
                'description': ' '.join(insight.split(' ')[1:]),  # Rest as description
                'type': insight_type
            })
        
        return formatted_insights
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        if self.data is not None:
            # Data quality recommendations
            missing_pct = (self.data.isnull().sum().sum() / (self.data.shape[0] * self.data.shape[1])) * 100
            if missing_pct > 5:
                recommendations.append("Consider advanced imputation techniques for missing data")
            
            # Feature engineering recommendations
            numeric_cols = len(self.data.select_dtypes(include=[np.number]).columns)
            categorical_cols = len(self.data.select_dtypes(include=['object']).columns)
            
            if categorical_cols > numeric_cols:
                recommendations.append("Explore feature encoding techniques for categorical variables")
            
            if numeric_cols > 10:
                recommendations.append("Consider dimensionality reduction techniques (PCA, t-SNE)")
            
            # Model recommendations
            if self.model_results:
                recommendations.append("Fine-tune hyperparameters of the best performing models")
                recommendations.append("Explore ensemble methods to improve model performance")
            
            # General recommendations
            recommendations.extend([
                "Validate findings with domain experts",
                "Consider collecting additional relevant features",
                "Implement model monitoring in production environment"
            ])
        
        return recommendations

def main():
    """Main CLI interface."""
    app = typer.Typer(help="Advanced Data Analysis Suite - The most comprehensive data analysis toolkit")
    
    @app.command()
    def analyze(
            file_path: str = typer.Argument(..., help="Path to data file"),
            target: str = typer.Option(None, "--target", "-t", help="Target column for ML analysis"),
            output_dir: str = typer.Option("analysis_output", "--output", "-o", help="Output directory"),
            auto_insights: bool = typer.Option(True, "--insights/--no-insights", help="Generate automatic insights")
    ):
        """Run comprehensive data analysis."""
        
        # Initialize suite
        suite = AdvancedDataAnalysisSuite({
            'output_dir': output_dir,
            'auto_insights': auto_insights
        })
        
        # Display banner
        suite.display_banner()
        
        # Load data
        if not suite.load_data(file_path):
            console.print("[red]❌ Failed to load data. Exiting.[/red]")
            return
        
        # Comprehensive EDA
        console.print("\n[cyan]🔍 Starting comprehensive exploratory data analysis...[/cyan]")
        eda_results = suite.comprehensive_eda()
        
        # Create visualizations
        console.print("\n[cyan]📊 Creating advanced visualizations...[/cyan]")
        viz_files = suite.create_advanced_visualizations()
        
        # AutoML if target specified
        if target:
            console.print(f"\n[cyan]🤖 Running AutoML for target: {target}[/cyan]")
            ml_results = suite.automl_model_selection(target)
            
            if 'error' not in ml_results:
                best_model = ml_results['best_model']
                console.print(f"[green]🏆 Best model: {best_model['name']} [/green]")
        
        # Generate comprehensive report
        console.print("\n[cyan]📄 Generating insights report...[/cyan]")
        report_path = suite.generate_insights_report()
        
        # Final summary
        console.print("\n[green]🎉 Analysis completed successfully![/green]")
        console.print(f"[blue]📁 Results saved in: {suite.output_dir}[/blue]")
        console.print(f"[blue]📄 View report: {report_path}[/blue]")
    
    app()

if __name__ == "__main__":
    main()
