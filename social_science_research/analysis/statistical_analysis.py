# -*- coding: utf-8 -*-
"""
Advanced Statistical Analysis for Social Science Research
Comprehensive statistical methods with AI-enhanced capabilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import scipy.stats as stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import warnings
warnings.filterwarnings('ignore')

class AnalysisType(Enum):
    """Types of statistical analyses"""
    DESCRIPTIVE = "descriptive"
    INFERENTIAL = "inferential"
    MULTIVARIATE = "multivariate"
    TIME_SERIES = "time_series"
    NETWORK_ANALYSIS = "network_analysis"
    MACHINE_LEARNING = "machine_learning"

class EffectSizeType(Enum):
    """Types of effect size measures"""
    COHENS_D = "cohens_d"
    ETA_SQUARED = "eta_squared"
    CRAMERS_V = "cramers_v"
    PEARSON_R = "pearson_r"
    ODDS_RATIO = "odds_ratio"

@dataclass
class StatisticalResult:
    """Structure for statistical analysis results"""
    analysis_type: AnalysisType
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    effect_size_type: EffectSizeType
    confidence_interval: Tuple[float, float]
    interpretation: str
    assumptions_met: Dict[str, bool]
    sample_size: int
    power: Optional[float] = None

@dataclass
class DescriptiveStats:
    """Comprehensive descriptive statistics"""
    variable: str
    n: int
    mean: float
    median: float
    mode: Any
    std: float
    variance: float
    skewness: float
    kurtosis: float
    min_val: float
    max_val: float
    q1: float
    q3: float
    iqr: float
    outliers: List[Any]

class AdvancedStatisticalAnalyzer:
    """
    Advanced statistical analysis system for social science research
    Integrates traditional statistics with AI-enhanced capabilities
    """
    
    def __init__(self):
        """Initialize the statistical analyzer"""
        self.data = None
        self.results = {}
        self.assumptions_cache = {}
        
    def load_data(self, data: pd.DataFrame) -> None:
        """Load data for analysis"""
        if data.empty:
            raise ValueError("Data cannot be empty")
        
        self.data = data.copy()
        self._validate_data()
        
    def _validate_data(self) -> None:
        """Validate loaded data"""
        if self.data is None:
            raise ValueError("No data loaded")
        
        # Check for basic data quality issues
        missing_data = self.data.isnull().sum()
        if missing_data.sum() > 0:
            print(f"Warning: Missing data detected in {missing_data[missing_data > 0].index.tolist()}")
    
    def descriptive_analysis(self, variables: List[str] = None) -> Dict[str, DescriptiveStats]:
        """Comprehensive descriptive analysis"""
        if variables is None:
            variables = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        results = {}
        
        for var in variables:
            if var not in self.data.columns:
                continue
                
            data_series = self.data[var].dropna()
            
            # Calculate outliers using IQR method
            q1 = data_series.quantile(0.25)
            q3 = data_series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = data_series[(data_series < lower_bound) | (data_series > upper_bound)].tolist()
            
            # Mode calculation
            mode_result = stats.mode(data_series, keepdims=True)
            mode_val = mode_result.mode[0] if len(mode_result.mode) > 0 else np.nan
            
            desc_stats = DescriptiveStats(
                variable=var,
                n=len(data_series),
                mean=data_series.mean(),
                median=data_series.median(),
                mode=mode_val,
                std=data_series.std(),
                variance=data_series.var(),
                skewness=stats.skew(data_series),
                kurtosis=stats.kurtosis(data_series),
                min_val=data_series.min(),
                max_val=data_series.max(),
                q1=q1,
                q3=q3,
                iqr=iqr,
                outliers=outliers
            )
            
            results[var] = desc_stats
        
        self.results['descriptive'] = results
        return results
    
    def correlation_analysis(self, variables: List[str] = None, 
                           method: str = 'pearson') -> Dict[str, Any]:
        """Comprehensive correlation analysis"""
        if variables is None:
            variables = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        data_subset = self.data[variables].dropna()
        
        if method == 'pearson':
            corr_matrix = data_subset.corr(method='pearson')
            # Calculate p-values
            p_values = np.zeros((len(variables), len(variables)))
            for i, var1 in enumerate(variables):
                for j, var2 in enumerate(variables):
                    if i != j:
                        _, p_val = pearsonr(data_subset[var1], data_subset[var2])
                        p_values[i, j] = p_val
        elif method == 'spearman':
            corr_matrix = data_subset.corr(method='spearman')
            # Calculate p-values for Spearman
            p_values = np.zeros((len(variables), len(variables)))
            for i, var1 in enumerate(variables):
                for j, var2 in enumerate(variables):
                    if i != j:
                        _, p_val = spearmanr(data_subset[var1], data_subset[var2])
                        p_values[i, j] = p_val
        
        p_values_df = pd.DataFrame(p_values, index=variables, columns=variables)
        
        # Identify significant correlations
        significant_corrs = []
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i < j:  # Avoid duplicates
                    corr_val = corr_matrix.loc[var1, var2]
                    p_val = p_values_df.loc[var1, var2]
                    if p_val < 0.05:
                        significant_corrs.append({
                            'var1': var1,
                            'var2': var2,
                            'correlation': corr_val,
                            'p_value': p_val,
                            'strength': self._interpret_correlation_strength(abs(corr_val))
                        })
        
        results = {
            'correlation_matrix': corr_matrix,
            'p_values': p_values_df,
            'significant_correlations': significant_corrs,
            'method': method,
            'sample_size': len(data_subset)
        }
        
        self.results['correlation'] = results
        return results
    
    def t_test_analysis(self, dependent_var: str, independent_var: str = None,
                       test_type: str = 'independent') -> StatisticalResult:
        """Comprehensive t-test analysis"""
        
        if test_type == 'one_sample':
            # One-sample t-test
            data_series = self.data[dependent_var].dropna()
            t_stat, p_val = stats.ttest_1samp(data_series, 0)
            
            # Effect size (Cohen's d)
            effect_size = data_series.mean() / data_series.std()
            
            # Confidence interval
            ci = stats.t.interval(0.95, len(data_series)-1, 
                                loc=data_series.mean(), 
                                scale=stats.sem(data_series))
            
        elif test_type == 'independent':
            # Independent samples t-test
            if independent_var is None:
                raise ValueError("Independent variable required for independent t-test")
            
            groups = self.data[independent_var].unique()
            if len(groups) != 2:
                raise ValueError("Independent variable must have exactly 2 groups")
            
            group1_data = self.data[self.data[independent_var] == groups[0]][dependent_var].dropna()
            group2_data = self.data[self.data[independent_var] == groups[1]][dependent_var].dropna()
            
            # Levene's test for equal variances
            levene_stat, levene_p = stats.levene(group1_data, group2_data)
            equal_var = levene_p > 0.05
            
            t_stat, p_val = stats.ttest_ind(group1_data, group2_data, equal_var=equal_var)
            
            # Cohen's d effect size
            pooled_std = np.sqrt(((len(group1_data) - 1) * group1_data.var() + 
                                (len(group2_data) - 1) * group2_data.var()) / 
                               (len(group1_data) + len(group2_data) - 2))
            effect_size = (group1_data.mean() - group2_data.mean()) / pooled_std
            
            # Confidence interval for difference
            diff_mean = group1_data.mean() - group2_data.mean()
            se_diff = pooled_std * np.sqrt(1/len(group1_data) + 1/len(group2_data))
            df = len(group1_data) + len(group2_data) - 2
            t_critical = stats.t.ppf(0.975, df)
            ci = (diff_mean - t_critical * se_diff, diff_mean + t_critical * se_diff)
        
        # Check assumptions
        assumptions = self._check_t_test_assumptions(dependent_var, independent_var, test_type)
        
        # Interpretation
        interpretation = self._interpret_t_test(t_stat, p_val, effect_size)
        
        result = StatisticalResult(
            analysis_type=AnalysisType.INFERENTIAL,
            test_name=f"{test_type}_t_test",
            statistic=t_stat,
            p_value=p_val,
            effect_size=abs(effect_size),
            effect_size_type=EffectSizeType.COHENS_D,
            confidence_interval=ci,
            interpretation=interpretation,
            assumptions_met=assumptions,
            sample_size=len(self.data[dependent_var].dropna())
        )
        
        self.results[f't_test_{test_type}'] = result
        return result
    
    def anova_analysis(self, dependent_var: str, independent_var: str) -> StatisticalResult:
        """One-way ANOVA analysis"""
        
        # Prepare data
        clean_data = self.data[[dependent_var, independent_var]].dropna()
        groups = clean_data[independent_var].unique()
        
        if len(groups) < 2:
            raise ValueError("Independent variable must have at least 2 groups")
        
        # Perform ANOVA
        group_data = [clean_data[clean_data[independent_var] == group][dependent_var].values 
                     for group in groups]
        f_stat, p_val = stats.f_oneway(*group_data)
        
        # Effect size (eta-squared)
        ss_between = sum([len(group) * (np.mean(group) - np.mean(clean_data[dependent_var]))**2 
                         for group in group_data])
        ss_total = sum([(x - np.mean(clean_data[dependent_var]))**2 
                       for x in clean_data[dependent_var]])
        eta_squared = ss_between / ss_total
        
        # Confidence interval (approximate)
        ci = (0, eta_squared * 1.2)  # Rough approximation
        
        # Check assumptions
        assumptions = self._check_anova_assumptions(dependent_var, independent_var)
        
        # Interpretation
        interpretation = self._interpret_anova(f_stat, p_val, eta_squared)
        
        result = StatisticalResult(
            analysis_type=AnalysisType.INFERENTIAL,
            test_name="one_way_anova",
            statistic=f_stat,
            p_value=p_val,
            effect_size=eta_squared,
            effect_size_type=EffectSizeType.ETA_SQUARED,
            confidence_interval=ci,
            interpretation=interpretation,
            assumptions_met=assumptions,
            sample_size=len(clean_data)
        )
        
        self.results['anova'] = result
        return result
    
    def regression_analysis(self, dependent_var: str, independent_vars: List[str],
                          regression_type: str = 'linear') -> Dict[str, Any]:
        """Comprehensive regression analysis"""
        
        # Prepare data
        all_vars = [dependent_var] + independent_vars
        clean_data = self.data[all_vars].dropna()
        
        X = clean_data[independent_vars]
        y = clean_data[dependent_var]
        
        if regression_type == 'linear':
            model = LinearRegression()
            model.fit(X, y)
            
            # Predictions and residuals
            y_pred = model.predict(X)
            residuals = y - y_pred
            
            # R-squared
            r_squared = model.score(X, y)
            
            # Adjusted R-squared
            n = len(y)
            p = len(independent_vars)
            adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
            
            # Coefficients and p-values (using statsmodels for p-values)
            import statsmodels.api as sm
            X_with_const = sm.add_constant(X)
            sm_model = sm.OLS(y, X_with_const).fit()
            
            coefficients = {
                'intercept': model.intercept_,
                'slopes': dict(zip(independent_vars, model.coef_)),
                'p_values': dict(zip(['const'] + independent_vars, sm_model.pvalues)),
                'confidence_intervals': dict(zip(['const'] + independent_vars, 
                                                sm_model.conf_int().values))
            }
            
        elif regression_type == 'logistic':
            model = LogisticRegression()
            model.fit(X, y)
            
            y_pred = model.predict_proba(X)[:, 1]
            residuals = y - y_pred
            
            # Pseudo R-squared (McFadden's)
            from sklearn.metrics import log_loss
            null_model_loss = log_loss(y, [y.mean()] * len(y))
            model_loss = log_loss(y, y_pred)
            r_squared = 1 - (model_loss / null_model_loss)
            adj_r_squared = r_squared  # Simplified for logistic
            
            coefficients = {
                'intercept': model.intercept_[0],
                'slopes': dict(zip(independent_vars, model.coef_[0])),
                'odds_ratios': dict(zip(independent_vars, np.exp(model.coef_[0])))
            }
        
        # Check assumptions
        assumptions = self._check_regression_assumptions(residuals, y_pred, regression_type)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5)
        
        results = {
            'model_type': regression_type,
            'r_squared': r_squared,
            'adjusted_r_squared': adj_r_squared,
            'coefficients': coefficients,
            'residuals': residuals,
            'predictions': y_pred,
            'assumptions_met': assumptions,
            'cross_validation_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'sample_size': len(clean_data)
        }
        
        self.results[f'regression_{regression_type}'] = results
        return results
    
    def factor_analysis(self, variables: List[str], n_factors: int = None) -> Dict[str, Any]:
        """Exploratory Factor Analysis"""
        
        data_subset = self.data[variables].dropna()
        
        # Standardize data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_subset)
        
        # Determine number of factors if not specified
        if n_factors is None:
            # Use Kaiser criterion (eigenvalues > 1)
            pca_temp = PCA()
            pca_temp.fit(data_scaled)
            eigenvalues = pca_temp.explained_variance_
            n_factors = sum(eigenvalues > 1)
        
        # Perform Factor Analysis
        fa = FactorAnalysis(n_components=n_factors, random_state=42)
        fa.fit(data_scaled)
        
        # Factor loadings
        loadings = pd.DataFrame(
            fa.components_.T,
            columns=[f'Factor_{i+1}' for i in range(n_factors)],
            index=variables
        )
        
        # Communalities
        communalities = pd.Series(
            np.sum(loadings**2, axis=1),
            index=variables,
            name='Communalities'
        )
        
        # Variance explained
        eigenvalues = np.var(fa.transform(data_scaled), axis=0)
        variance_explained = eigenvalues / len(variables)
        cumulative_variance = np.cumsum(variance_explained)
        
        results = {
            'n_factors': n_factors,
            'factor_loadings': loadings,
            'communalities': communalities,
            'eigenvalues': eigenvalues,
            'variance_explained': variance_explained,
            'cumulative_variance': cumulative_variance,
            'sample_size': len(data_subset)
        }
        
        self.results['factor_analysis'] = results
        return results
    
    def cluster_analysis(self, variables: List[str], n_clusters: int = None) -> Dict[str, Any]:
        """K-means cluster analysis"""
        
        data_subset = self.data[variables].dropna()
        
        # Standardize data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_subset)
        
        # Determine optimal number of clusters if not specified
        if n_clusters is None:
            # Use elbow method
            inertias = []
            k_range = range(2, min(11, len(data_subset)//2))
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(data_scaled)
                inertias.append(kmeans.inertia_)
            
            # Simple elbow detection (largest decrease)
            decreases = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
            n_clusters = k_range[np.argmax(decreases)]
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(data_scaled)
        
        # Add cluster labels to original data
        clustered_data = data_subset.copy()
        clustered_data['Cluster'] = cluster_labels
        
        # Cluster centers in original scale
        cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
        centers_df = pd.DataFrame(
            cluster_centers,
            columns=variables,
            index=[f'Cluster_{i}' for i in range(n_clusters)]
        )
        
        # Cluster statistics
        cluster_stats = clustered_data.groupby('Cluster')[variables].agg(['mean', 'std', 'count'])
        
        results = {
            'n_clusters': n_clusters,
            'cluster_labels': cluster_labels,
            'cluster_centers': centers_df,
            'cluster_statistics': cluster_stats,
            'inertia': kmeans.inertia_,
            'silhouette_score': self._calculate_silhouette_score(data_scaled, cluster_labels),
            'sample_size': len(data_subset)
        }
        
        self.results['cluster_analysis'] = results
        return results
    
    def network_analysis(self, correlation_threshold: float = 0.3) -> Dict[str, Any]:
        """Social network analysis of variable relationships"""
        
        # Get correlation matrix
        if 'correlation' not in self.results:
            self.correlation_analysis()
        
        corr_matrix = self.results['correlation']['correlation_matrix']
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes (variables)
        variables = corr_matrix.columns.tolist()
        G.add_nodes_from(variables)
        
        # Add edges (significant correlations)
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i < j:  # Avoid duplicates
                    corr_val = corr_matrix.loc[var1, var2]
                    if abs(corr_val) >= correlation_threshold:
                        G.add_edge(var1, var2, weight=abs(corr_val), correlation=corr_val)
        
        # Network metrics
        centrality_measures = {
            'degree_centrality': nx.degree_centrality(G),
            'betweenness_centrality': nx.betweenness_centrality(G),
            'closeness_centrality': nx.closeness_centrality(G),
            'eigenvector_centrality': nx.eigenvector_centrality(G) if len(G.edges()) > 0 else {}
        }
        
        # Network statistics
        network_stats = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
            'average_clustering': nx.average_clustering(G),
            'connected_components': nx.number_connected_components(G)
        }
        
        results = {
            'graph': G,
            'centrality_measures': centrality_measures,
            'network_statistics': network_stats,
            'correlation_threshold': correlation_threshold
        }
        
        self.results['network_analysis'] = results
        return results
    
    def _interpret_correlation_strength(self, correlation: float) -> str:
        """Interpret correlation strength"""
        if correlation < 0.1:
            return "negligible"
        elif correlation < 0.3:
            return "weak"
        elif correlation < 0.5:
            return "moderate"
        elif correlation < 0.7:
            return "strong"
        else:
            return "very strong"
    
    def _check_t_test_assumptions(self, dependent_var: str, independent_var: str = None,
                                 test_type: str = 'independent') -> Dict[str, bool]:
        """Check t-test assumptions"""
        assumptions = {}
        
        if test_type == 'independent' and independent_var:
            groups = self.data[independent_var].unique()
            group1_data = self.data[self.data[independent_var] == groups[0]][dependent_var].dropna()
            group2_data = self.data[self.data[independent_var] == groups[1]][dependent_var].dropna()
            
            # Normality (Shapiro-Wilk test)
            _, p1 = stats.shapiro(group1_data)
            _, p2 = stats.shapiro(group2_data)
            assumptions['normality'] = p1 > 0.05 and p2 > 0.05
            
            # Equal variances (Levene's test)
            _, p_levene = stats.levene(group1_data, group2_data)
            assumptions['equal_variances'] = p_levene > 0.05
            
        else:
            data_series = self.data[dependent_var].dropna()
            _, p_norm = stats.shapiro(data_series)
            assumptions['normality'] = p_norm > 0.05
        
        return assumptions
    
    def _check_anova_assumptions(self, dependent_var: str, independent_var: str) -> Dict[str, bool]:
        """Check ANOVA assumptions"""
        assumptions = {}
        
        clean_data = self.data[[dependent_var, independent_var]].dropna()
        groups = clean_data[independent_var].unique()
        
        # Normality within groups
        normality_tests = []
        for group in groups:
            group_data = clean_data[clean_data[independent_var] == group][dependent_var]
            if len(group_data) >= 3:  # Minimum for Shapiro-Wilk
                _, p_val = stats.shapiro(group_data)
                normality_tests.append(p_val > 0.05)
        
        assumptions['normality'] = all(normality_tests) if normality_tests else False
        
        # Homogeneity of variances (Levene's test)
        group_data = [clean_data[clean_data[independent_var] == group][dependent_var].values 
                     for group in groups]
        _, p_levene = stats.levene(*group_data)
        assumptions['homogeneity_of_variances'] = p_levene > 0.05
        
        return assumptions
    
    def _check_regression_assumptions(self, residuals: np.ndarray, predictions: np.ndarray,
                                    regression_type: str) -> Dict[str, bool]:
        """Check regression assumptions"""
        assumptions = {}
        
        if regression_type == 'linear':
            # Linearity (correlation between residuals and predictions should be near 0)
            corr_resid_pred, p_val = pearsonr(residuals, predictions)
            assumptions['linearity'] = abs(corr_resid_pred) < 0.1
            
            # Normality of residuals
            _, p_norm = stats.shapiro(residuals)
            assumptions['normality_of_residuals'] = p_norm > 0.05
            
            # Homoscedasticity (Breusch-Pagan test approximation)
            # Simple check: correlation between absolute residuals and predictions
            abs_residuals = np.abs(residuals)
            corr_abs_resid, p_val = pearsonr(abs_residuals, predictions)
            assumptions['homoscedasticity'] = abs(corr_abs_resid) < 0.1
        
        return assumptions
    
    def _interpret_t_test(self, t_stat: float, p_val: float, effect_size: float) -> str:
        """Interpret t-test results"""
        significance = "significant" if p_val < 0.05 else "not significant"
        effect_interpretation = self._interpret_cohens_d(abs(effect_size))
        
        return f"The t-test result is {significance} (p = {p_val:.4f}) with a {effect_interpretation} effect size (d = {abs(effect_size):.3f})"
    
    def _interpret_anova(self, f_stat: float, p_val: float, eta_squared: float) -> str:
        """Interpret ANOVA results"""
        significance = "significant" if p_val < 0.05 else "not significant"
        effect_interpretation = self._interpret_eta_squared(eta_squared)
        
        return f"The ANOVA result is {significance} (F = {f_stat:.3f}, p = {p_val:.4f}) with a {effect_interpretation} effect size (η² = {eta_squared:.3f})"
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _interpret_eta_squared(self, eta_squared: float) -> str:
        """Interpret eta-squared effect size"""
        if eta_squared < 0.01:
            return "negligible"
        elif eta_squared < 0.06:
            return "small"
        elif eta_squared < 0.14:
            return "medium"
        else:
            return "large"
    
    def _calculate_silhouette_score(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Calculate silhouette score for clustering"""
        from sklearn.metrics import silhouette_score
        if len(np.unique(labels)) > 1:
            return silhouette_score(data, labels)
        else:
            return 0.0
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive statistical analysis report"""
        
        report = {
            'summary': {
                'total_analyses': len(self.results),
                'data_shape': self.data.shape if self.data is not None else None,
                'analyses_performed': list(self.results.keys())
            },
            'descriptive_summary': self._summarize_descriptive_results(),
            'inferential_summary': self._summarize_inferential_results(),
            'multivariate_summary': self._summarize_multivariate_results(),
            'recommendations': self._generate_analysis_recommendations()
        }
        
        return report
    
    def _summarize_descriptive_results(self) -> Dict[str, Any]:
        """Summarize descriptive analysis results"""
        if 'descriptive' not in self.results:
            return {}
        
        desc_results = self.results['descriptive']
        summary = {
            'variables_analyzed': len(desc_results),
            'variables_with_outliers': sum(1 for stats in desc_results.values() if len(stats.outliers) > 0),
            'highly_skewed_variables': [var for var, stats in desc_results.items() if abs(stats.skewness) > 2],
            'variables_with_high_kurtosis': [var for var, stats in desc_results.items() if abs(stats.kurtosis) > 2]
        }
        
        return summary
    
    def _summarize_inferential_results(self) -> Dict[str, Any]:
        """Summarize inferential analysis results"""
        inferential_tests = [key for key in self.results.keys() 
                           if any(test in key for test in ['t_test', 'anova', 'correlation'])]
        
        significant_results = []
        for test_name in inferential_tests:
            result = self.results[test_name]
            if hasattr(result, 'p_value') and result.p_value < 0.05:
                significant_results.append(test_name)
            elif isinstance(result, dict) and 'significant_correlations' in result:
                if len(result['significant_correlations']) > 0:
                    significant_results.append(test_name)
        
        return {
            'tests_performed': len(inferential_tests),
            'significant_results': significant_results,
            'proportion_significant': len(significant_results) / len(inferential_tests) if inferential_tests else 0
        }
    
    def _summarize_multivariate_results(self) -> Dict[str, Any]:
        """Summarize multivariate analysis results"""
        multivariate_analyses = [key for key in self.results.keys() 
                               if any(analysis in key for analysis in ['factor_analysis', 'cluster_analysis', 'regression'])]
        
        summary = {
            'analyses_performed': multivariate_analyses,
            'total_analyses': len(multivariate_analyses)
        }
        
        # Add specific summaries for each type
        if 'factor_analysis' in self.results:
            fa_result = self.results['factor_analysis']
            summary['factor_analysis'] = {
                'n_factors_extracted': fa_result['n_factors'],
                'total_variance_explained': fa_result['cumulative_variance'][-1]
            }
        
        if 'cluster_analysis' in self.results:
            cluster_result = self.results['cluster_analysis']
            summary['cluster_analysis'] = {
                'n_clusters': cluster_result['n_clusters'],
                'silhouette_score': cluster_result['silhouette_score']
            }
        
        return summary
    
    def _generate_analysis_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis results"""
        recommendations = []
        
        # Check for common issues and provide recommendations
        if 'descriptive' in self.results:
            desc_results = self.results['descriptive']
            
            # Check for outliers
            vars_with_outliers = [var for var, stats in desc_results.items() if len(stats.outliers) > 0]
            if vars_with_outliers:
                recommendations.append(f"Consider investigating outliers in variables: {', '.join(vars_with_outliers)}")
            
            # Check for skewness
            skewed_vars = [var for var, stats in desc_results.items() if abs(stats.skewness) > 2]
            if skewed_vars:
                recommendations.append(f"Consider data transformation for highly skewed variables: {', '.join(skewed_vars)}")
        
        # Check assumption violations
        assumption_violations = []
        for test_name, result in self.results.items():
            if hasattr(result, 'assumptions_met'):
                violated_assumptions = [assumption for assumption, met in result.assumptions_met.items() if not met]
                if violated_assumptions:
                    assumption_violations.extend(violated_assumptions)
        
        if assumption_violations:
            recommendations.append(f"Address assumption violations: {', '.join(set(assumption_violations))}")
        
        # General recommendations
        recommendations.extend([
            "Consider effect sizes alongside statistical significance",
            "Validate findings with independent samples when possible",
            "Report confidence intervals for key estimates",
            "Consider multiple comparison corrections for multiple tests"
        ])
        
        return recommendations
