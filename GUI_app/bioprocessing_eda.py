import os
import warnings
# ==========================================
# 🌟 NEW: GEORGIEV FEATURES DICTIONARY
# ==========================================
# 19-Dimensional Principal Components (Georgiev 2008)
GEORGIEV_DICT = {
    'A': [0.57, 3.37, -3.66, 2.34, -1.07, -0.4, 1.23, -2.32, -2.01, 1.31, -1.14, 0.19, 1.66, 4.39, 0.18, -2.6, 1.49, 0.46, -4.22],
    'C': [2.66, -1.52, -3.29, -3.77, 2.96, -2.23, 0.44, -3.49, 2.22, -3.78, 1.98, -0.43, -1.03, 0.93, 1.43, 1.45, -1.15, -1.64, -1.05],
    'D': [-2.46, -0.66, -0.57, 0.14, 0.75, 0.24, -5.15, -1.17, 0.73, 1.5, 1.51, 5.61, -3.85, 1.28, -1.98, 0.05, 0.9, 1.38, -0.03],
    'E': [-3.08, 3.45, 0.05, 0.62, -0.49, 0, -5.66, -0.11, 1.49, -2.26, -1.62, -3.97, 2.3, -0.06, -0.35, 1.51, -2.29, -1.47, 0.15],
    'F': [3.12, 0.68, 2.4, -0.35, -0.88, 1.62, -0.15, -0.41, 4.2, 0.73, -0.56, 3.54, 5.25, 1.73, 2.14, 1.1, 0.68, 1.46, 2.33],
    'G': [0.15, -3.49, -2.97, 2.06, 0.7, 7.47, 0.41, 1.62, -0.47, -2.9, -0.98, -0.62, -0.11, 0.15, -0.53, 0.35, 0.3, 0.32, 0.05],
    'H': [-0.39, 1, -0.63, -3.49, 0.05, 0.41, 1.61, -0.6, 3.55, 1.52, -2.28, -3.12, -1.45, -0.77, -4.18, -2.91, 3.37, 1.87, 2.17],
    'I': [3.1, 0.37, 0.26, 1.04, -0.05, -1.18, -0.21, 3.45, 0.86, 1.98, 0.89, -1.67, -1.02, -1.21, -1.78, 5.71, 1.54, 2.11, -4.18],
    'K': [-3.89, 1.47, 1.95, 1.17, 0.53, 0.1, 4.01, -0.01, -0.26, -1.66, 5.86, -0.06, 1.38, 1.78, -2.71, 1.62, 0.96, -1.09, 1.36],
    'L': [2.72, 1.88, 1.92, 5.33, 0.08, 0.09, 0.27, -4.06, 0.43, -1.2, 0.67, -0.29, -2.47, -4.79, 0.8, -1.43, 0.63, -0.24, 1.01],
    'M': [1.89, 3.88, -1.57, -3.58, -2.55, 2.07, 0.84, 1.85, -2.05, 0.78, 1.53, 2.44, -0.26, -3.09, -1.39, -1.02, -4.32, -1.34, 0.09],
    'N': [-2.02, -1.92, 0.04, -0.65, 1.61, 2.08, 0.4, -2.47, -0.07, 7.02, 1.32, -2.44, 0.37, -0.89, 3.13, 0.79, -1.54, -1.71, -0.25],
    'P': [-0.58, -4.33, -0.02, -0.21, -8.31, -1.82, -0.12, -1.18, 0, -0.66, 0.64, -0.92, -0.37, 0.17, 0.36, 0.08, 0.16, -0.34, 0.04],
    'Q': [-2.54, 1.82, -0.82, -1.85, 0.09, 0.6, 0.25, 2.11, -1.92, -1.67, 0.7, -0.27, -0.99, -1.56, 6.22, -0.18, 2.72, 4.35, 0.92],
    'R': [-2.8, 0.31, 2.84, 0.25, 0.2, -0.37, 3.81, 0.98, 2.43, -0.99, -4.9, 2.09, -3.08, 0.82, 1.32, 0.69, -2.62, -1.49, -2.57],
    'S': [-1.1, -2.05, -2.19, 1.36, 1.78, -3.36, 1.39, -1.21, -2.83, 0.39, -2.92, 1.27, 2.86, -1.88, -2.42, 1.75, -2.77, 3.36, 2.67],
    'T': [-0.65, -1.6, -1.39, 0.63, 1.35, -2.45, -0.65, 3.43, 0.34, 0.24, -0.53, 1.91, 2.66, -3.07, 0.2, -2.2, 3.73, -5.46, -0.73],
    'V': [2.64, 0.03, -0.67, 2.34, 0.64, -2.01, -0.33, 3.93, -0.21, 1.27, 0.43, -1.71, -2.93, 4.22, 1.06, -1.31, -1.97, -1.21, 4.77],
    'W': [1.89, -0.09, 4.21, -2.77, 0.72, 0.86, -1.07, -1.66, -5.87, -0.66, -2.49, -0.3, -0.5, 1.64, -0.72, 1.75, 2.73, -2.2, 0.9],
    'Y': [0.79, -2.62, 4.11, -0.63, 1.89, -0.53, -1.3, 1.31, -0.56, -0.95, 1.91, -1.26, 1.57, 0.2, -0.76, -5.19, -2.56, 2.87, -3.43]
}
# --- NUCLEAR OPTION FOR WARNINGS ---
# Scikit-learn has an internal bug where it warns itself about parallel processing. 
# Because joblib workers often bypass standard filters, we physically disable the warn function.
def completely_silence_warnings(*args, **kwargs):
    pass
warnings.warn = completely_silence_warnings

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools
import urllib.request
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedKFold, train_test_split, cross_val_predict, LeaveOneOut, cross_validate, LearningCurveDisplay, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score
from scipy.stats import spearmanr
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PowerTransformer
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from scipy.optimize import minimize_scalar
from scipy.stats import yeojohnson
from matplotlib.colors import ListedColormap
import math
# --- REMOVED HEAVY IMPORTS FROM HERE ---
# (They have been moved inside the feature extraction function for faster loading)
ESM_AVAILABLE = True

# --- NEW: PROPERMAB INTEGRATION ---
#from propermab import defaults
#from propermab.features import feature_utils
#defaults.system_config.update_from_json('./default_config.json') # Often needed depending on user setup
PROPERMAB_AVAILABLE = False

# Set visualization style
sns.set_theme(style="whitegrid")

def custom_spearman(y_true, y_pred):
    """Custom scoring function to handle NaN spearman correlations."""
    sc, _ = spearmanr(y_true, y_pred)
    return sc if not np.isnan(sc) else 0

# --- NEW: CUSTOM WEIGHTED TRANSFORMER MATH & META-ESTIMATOR ---
def calculate_weighted_yeojohnson_lambda(y, weights):
    """Finds the optimal Yeo-Johnson lambda that maximizes the weighted log-likelihood."""
    y_flat = np.asarray(y).flatten()
    w = np.asarray(weights).flatten()
    
    # Normalize weights to sum to N to stabilize variance calculations
    w = (w / np.sum(w)) * len(w)
    
    def weighted_nll(lmbda):
        # 1. Transform data
        y_trans = yeojohnson(y_flat, lmbda)
        
        # 2. Weighted Variance
        w_mean = np.average(y_trans, weights=w)
        w_var = np.average((y_trans - w_mean)**2, weights=w)
        if w_var <= 0: return np.inf
        
        # 3. Jacobian (Log-derivative of the transformation)
        jacobian = np.sum(w * np.sign(y_flat) * np.log1p(np.abs(y_flat)))
        
        # 4. Negative Log-Likelihood
        nll = (len(w) / 2.0) * np.log(w_var) - (lmbda - 1) * jacobian
        return nll
        
    res = minimize_scalar(weighted_nll, bounds=(-3.0, 3.0), method='bounded')
    return res.x

class CustomYeoJohnsonTransformer(BaseEstimator, TransformerMixin):
    """A Scikit-Learn compatible transformer that uses a pre-calculated optimal lambda."""
    def __init__(self, lmbda=1.0):
        self.lmbda = lmbda
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_flat = np.asarray(X).flatten()
        X_trans = yeojohnson(X_flat, self.lmbda)
        return X_trans.reshape(-1, 1)
        
    def inverse_transform(self, X_trans):
        X_trans_flat = np.asarray(X_trans).flatten()
        X_inv = np.zeros_like(X_trans_flat)
        
        pos = X_trans_flat >= 0
        neg = ~pos
        
        # Inverse mapping for y >= 0
        if abs(self.lmbda) < 1e-10:
            X_inv[pos] = np.exp(X_trans_flat[pos]) - 1
        else:
            X_inv[pos] = np.maximum(X_trans_flat[pos] * self.lmbda + 1, 0) ** (1 / self.lmbda) - 1
            
        # Inverse mapping for y < 0
        if abs(self.lmbda - 2.0) < 1e-10:
            X_inv[neg] = 1 - np.exp(-X_trans_flat[neg])
        else:
            X_inv[neg] = 1 - np.maximum(1 - (2 - self.lmbda) * X_trans_flat[neg], 0) ** (1 / (2 - self.lmbda))
            
        return X_inv.reshape(-1, 1)

class SelfContainedTargetTransformRegressor(BaseEstimator, RegressorMixin):
    """
    A custom meta-estimator that perfectly handles target transformations (including weighted) 
    without leaking data across CV folds by unpacking hidden weights from X during fit.
    """
    def __init__(self, regressor, transform_type=None, weight_col=None):
        self.regressor = regressor
        self.transform_type = transform_type
        self.weight_col = weight_col
        
    def fit(self, X, y):
        self.regressor_ = clone(self.regressor)
        
        # Unpack weights and clean X so the ML model never sees the weights as a feature
        if self.weight_col:
            if isinstance(X, pd.DataFrame):
                weights = X[self.weight_col].values
                X_clean = X.drop(columns=[self.weight_col])
            else:
                # If a CV tool converted X to a numpy array, the weight column is appended at the very end
                weights = X[:, -1]
                X_clean = X[:, :-1]
        else:
            X_clean = X.copy() if isinstance(X, pd.DataFrame) else np.copy(X)
            weights = None
            
        # Transform Target securely on this specific CV fold
        if self.transform_type == 'log1p':
            y_trans = np.log1p(y)
        elif self.transform_type in ['box-cox', 'yeo-johnson']:
            self.pt_ = PowerTransformer(method=self.transform_type)
            y_trans = self.pt_.fit_transform(np.asarray(y).reshape(-1, 1)).flatten()
        elif self.transform_type == 'weighted-yeo-johnson':
            if weights is None:
                weights = np.ones_like(y)
            self.lmbda_ = calculate_weighted_yeojohnson_lambda(y, weights)
            self.pt_ = CustomYeoJohnsonTransformer(lmbda=self.lmbda_)
            y_trans = self.pt_.transform(np.asarray(y).reshape(-1, 1)).flatten()
        else:
            y_trans = y
            
        self.regressor_.fit(X_clean, y_trans)
        return self
        
    def predict(self, X):
        # Unpack weights and clean X before predicting
        if self.weight_col:
            if isinstance(X, pd.DataFrame):
                X_clean = X.drop(columns=[self.weight_col])
            else:
                X_clean = X[:, :-1]
        else:
            X_clean = X
            
        y_pred_trans = self.regressor_.predict(X_clean)
        
        # Inverse Transform
        if self.transform_type == 'log1p':
            return np.expm1(y_pred_trans)
        elif self.transform_type in ['box-cox', 'yeo-johnson']:
            return self.pt_.inverse_transform(y_pred_trans.reshape(-1, 1)).flatten()
        elif self.transform_type == 'weighted-yeo-johnson':
            return self.pt_.inverse_transform(y_pred_trans.reshape(-1, 1)).flatten()
        else:
            return np.asarray(y_pred_trans).flatten()


def load_and_clean_data(filepath, remove_outlier=False):
    """Loads the dataset and optionally removes known outliers."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Strip whitespace from column names just in case
    df.columns = df.columns.str.strip()
    
    if remove_outlier:
        # Check if the column exists to avoid key errors
        if 'Samples' in df.columns:
            # Strip string values to ensure a perfect match, then filter
            initial_len = len(df)
            
            # 🌟 NEW: Define a list of all known catastrophic failures
            outliers_to_remove = ['H57L46', 'H57L38']
            
            # Use ~ (NOT) and .isin() to drop any row matching the list
            df = df[~df['Samples'].str.strip().isin(outliers_to_remove)]
            
            # Reset the index (CRITICAL for Leave-One-Out CV to work properly later)
            df = df.reset_index(drop=True)
            
            rows_dropped = initial_len - len(df)
            if rows_dropped > 0:
                print(f"  -> SUCCESS: Outliers removed. ({rows_dropped} rows dropped)")
            else:
                print("  -> WARNING: Toggle is ON, but specified outliers were not found in the dataset.")
        else:
            print("  -> ERROR: 'Samples' column not found. Cannot remove outlier.")
            
    return df

def determine_preferred_media(df):
    """Calculates average performance to determine the preferred media."""
    print("--- Media Performance Summary ---")
    if 'ProA_Monomer_Excell' in df.columns and 'ProA_Monomer_ActiPro' in df.columns:
        avg_monomer_ex = df['ProA_Monomer_Excell'].mean()
        avg_monomer_act = df['ProA_Monomer_ActiPro'].mean()
        print(f"Average Monomer (Excell):  {avg_monomer_ex:.2f}%")
        print(f"Average Monomer (ActiPro): {avg_monomer_act:.2f}%")
        if avg_monomer_act > avg_monomer_ex: print("Winner for Purity (Monomer): ActiPro")
        else: print("Winner for Purity (Monomer): Excell")
            
    if 'ProA_HMW_Excell' in df.columns and 'ProA_HMW_ActiPro' in df.columns:
        avg_hmw_ex = df['ProA_HMW_Excell'].mean()
        avg_hmw_act = df['ProA_HMW_ActiPro'].mean()
        print(f"\nAverage Aggregates HMW (Excell):  {avg_hmw_ex:.2f}%")
        print(f"Average Aggregates HMW (ActiPro): {avg_hmw_act:.2f}%")
        if avg_hmw_act < avg_hmw_ex: print("Winner for Lowest Aggregation: ActiPro")
        else: print("Winner for Lowest Aggregation: Excell")
    print("---------------------------------\n")

def plot_media_comparison(df):
    """Compares key metrics between Excell and ActiPro media conditions."""
    print("Generating Media Comparison Plots (Excell vs ActiPro)...")
    metrics = [
        ('ProA_Monomer_Excell', 'ProA_Monomer_ActiPro', 'Protein A Monomer %'),
        ('ProA_HMW_Excell', 'ProA_HMW_ActiPro', 'Protein A HMW (Aggregates) %'),
        ('AC-SINS_Excell', 'AC-SINS_ActiPro', 'AC-SINS Score')
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (ex_col, acti_col, title) in enumerate(metrics):
        if ex_col in df.columns and acti_col in df.columns:
            sns.scatterplot(data=df, x=ex_col, y=acti_col, ax=axes[i], alpha=0.7)
            min_val = min(df[ex_col].min(), df[acti_col].min())
            max_val = max(df[ex_col].max(), df[acti_col].max())
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x (Equal Perf.)')
            axes[i].set_title(title)
            axes[i].set_xlabel(f"{title} - Excell")
            axes[i].set_ylabel(f"{title} - ActiPro")
            axes[i].legend()
    plt.tight_layout()
    plt.savefig('media_comparison.png', dpi=300)
    plt.close()

def filter_redundant_sequences(df, seq_cols):
    """Finds the most granular sequence columns by checking if they are substrings of others."""
    parents = set()
    for col_a in seq_cols:
        for col_b in seq_cols:
            if col_a == col_b: continue
            
            val_a = df[col_a].dropna()
            val_b = df[col_b].dropna()
            if val_a.empty or val_b.empty: continue
            
            str_a = ''.join(str(val_a.iloc[0]).split()).replace(',', '').upper()
            str_b = ''.join(str(val_b.iloc[0]).split()).replace(',', '').upper()
            
            if len(str_a) > 5 and str_a in str_b and len(str_a) < len(str_b):
                parents.add(col_b)
                
    granular_cols = [c for c in seq_cols if c not in parents]
    return granular_cols, list(parents)

def load_aaindex():
    """Downloads and parses the AAindex1 database of 566 physicochemical properties."""
    filepath = 'aaindex1.txt'
    aaindex_dict = {}
    aaindex_desc = {}
    if not os.path.exists(filepath):
        print("Downloading AAindex1 database (~566 properties)...")
        url = "https://www.genome.jp/ftp/db/community/aaindex/aaindex1"
        try:
            urllib.request.urlretrieve(url, filepath)
        except Exception as e:
            print(f"Failed to download AAindex1: {e}")
            return aaindex_dict, aaindex_desc
            
    print("Parsing AAindex1 properties and descriptions...")
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        current_id = None
        current_desc = None
        for i in range(len(lines)):
            if lines[i].startswith('H '):
                current_id = lines[i].split()[1]
            elif lines[i].startswith('D '):
                current_desc = lines[i][2:].strip()
            elif lines[i].startswith('I '):
                vals1 = lines[i+1].strip().split()
                vals2 = lines[i+2].strip().split()
                
                def parse_val(x):
                    try: return float(x)
                    except ValueError: return np.nan
                        
                vals1 = [parse_val(x) for x in vals1]
                vals2 = [parse_val(x) for x in vals2]
                
                if len(vals1) == 10 and len(vals2) == 10:
                    aa_keys = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                               'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
                    prop_map = dict(zip(aa_keys, vals1 + vals2))
                    aaindex_dict[current_id] = prop_map
                    aaindex_desc[current_id] = current_desc
    except Exception as e:
        print(f"Error parsing AAindex1: {e}")
        
    return aaindex_dict, aaindex_desc

def calculate_and_plot_variance(filepath, output_filename='subregion_sequence_variance.png'):
    """
    Loads antibody sequence data, constructs global chains, calculates 
    the sequence diversity (unique sequences) for each subregion, 
    and generates a color-coded horizontal bar chart.
    """
    print(f"Loading data from '{filepath}'...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: Could not find '{filepath}'. Please check the path.")
        return

    df.columns = df.columns.str.strip()
    total_samples = len(df)
    
    # Define the exact columns that make up the Heavy and Light chains
    vh_ordered_cols = ['seq_frh1', 'seq_cdrh1', 'seq_frh2', 'seq_cdrh2', 'seq_frh3', 'seq_cdrh3', 'seq_frh4'] 
    vl_ordered_cols = ['seq_frl1', 'seq_cdrl1', 'seq_frl2', 'seq_cdrl2', 'seq_frl3', 'seq_cdrl3', 'seq_frl4']
    
    # All subregions we want to analyze
    subregion_cols = vh_ordered_cols + vl_ordered_cols
    
    # Helper to build the stitched chains
    def build_full_seq(row, cols):
        parts = []
        for c in cols:
            if c in df.columns and pd.notna(row[c]):
                val = str(row[c]).strip().upper()
                if val != 'NAN' and val != 'NONE':
                    parts.append(val)
        return "".join(parts) if parts else 'NAN'

    # Stitch the global chains
    df['Global_VH'] = df.apply(lambda r: build_full_seq(r, vh_ordered_cols), axis=1)
    df['Global_VL'] = df.apply(lambda r: build_full_seq(r, vl_ordered_cols), axis=1)
    df['Global_Fv'] = df.apply(lambda r: r['Global_VH'] + r['Global_VL'] if r['Global_VH'] != 'NAN' and r['Global_VL'] != 'NAN' else 'NAN', axis=1)

    global_cols = ['Global_VH', 'Global_VL', 'Global_Fv']
    all_seq_cols = subregion_cols + global_cols

    variance_data = []
    
    for col in all_seq_cols:
        if col in df.columns:
            # Clean the sequences just like the main pipeline does
            valid_seqs = df[col].astype(str).str.replace(r'\s+|,', '', regex=True).str.upper()
            
            # Remove empty strings or string-literals of NAN/NONE
            valid_seqs = valid_seqs[~valid_seqs.isin(['NAN', 'NONE', ''])]
            
            # Count the absolute number of unique sequences in this region
            num_unique = valid_seqs.nunique()
            
            # Determine the Region Type for color coding the plot
            if col in global_cols:
                region_type = 'Global Chain (VH, VL, Fv)'
            elif 'cdr' in col.lower():
                region_type = 'CDR Loop'
            elif 'fr' in col.lower():
                region_type = 'Framework Region'
            else:
                region_type = 'Other'
                
            variance_data.append({
                'Subregion': col,
                'Unique Sequences': num_unique,
                'Region Type': region_type
            })

    # Convert to DataFrame and sort from highest variance to lowest
    var_df = pd.DataFrame(variance_data)
    var_df = var_df.sort_values(by='Unique Sequences', ascending=False)
    
    print("\nSequence Diversity (Unique Sequences out of Total Samples):")
    print(var_df.to_string(index=False))

    plt.figure(figsize=(12, 10))
    
    # Define a custom color palette for high readability
    custom_palette = {
        'Global Chain (VH, VL, Fv)': '#2ca02c', # Green
        'CDR Loop': '#1f77b4',                  # Blue
        'Framework Region': '#ff7f0e'           # Orange
    }
    
    # Create the horizontal bar plot
    ax = sns.barplot(
        data=var_df, 
        x='Unique Sequences', 
        y='Subregion', 
        hue='Region Type', 
        dodge=False, 
        palette=custom_palette,
        edgecolor='black'
    )
    
    # Add a vertical dotted line showing the theoretical maximum variance (Total Samples)
    plt.axvline(total_samples, color='red', linestyle='--', linewidth=2, 
                label=f'Max Possible Variance (N={total_samples})')
    
    # Labels and Titles
    plt.title('Sequence Diversity by Subregion\n(Higher Variance = More Learning Signal for ML Models)', fontsize=16, pad=20)
    plt.xlabel('Number of Unique Sequences in Library', fontsize=14)
    plt.ylabel('Antibody Region', fontsize=14)
    
    # Adjust legend
    plt.legend(title='Region Type', loc='lower right', fontsize=12, title_fontsize=12)
    
    # Add data labels to the end of each bar for easy reading
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        if not pd.isna(width) and width > 0:
            ax.text(width + 0.5, p.get_y() + p.get_height() / 2, f'{int(width)}', 
                    ha='left', va='center', fontsize=10, fontweight='bold')

    plt.xlim(0, total_samples + 5)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_filename, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Success! Variance plot saved as '{output_filename}'.")

def extract_sequence_features(df, is_inference=False, dataset_name="default_dataset", esm_model_name="facebook/esm2_t6_8M_UR50D", cache_tag=""):
    """Extracts physiochemical features from antibody subregions."""
    if not is_inference:
        print("Extracting sequence features from subregions...")
    df_feat = df.copy()
    
    standard_aas = list('ARNDCQEGHILKMFPSTWYV')
    
    seq_cols = []
    for col in df_feat.columns:
        valid_data = df_feat[col].dropna()
        if not valid_data.empty:
            sample = str(valid_data.iloc[0])
            clean_sample = ''.join(sample.split()).replace(',', '').upper()
            if len(clean_sample) > 5:
                aa_ratio = sum(c in standard_aas for c in clean_sample) / len(clean_sample)
                if aa_ratio > 0.8:
                    seq_cols.append(col)
                
    if not is_inference:
        print(f"Detected {len(seq_cols)} potential sequence columns.")
    
    if not is_inference:
        variable_seq_cols = []
        for col in seq_cols:
            valid_seqs = df_feat[col].dropna().astype(str).str.replace(r'\s+|,', '', regex=True).str.upper()
            num_unique = valid_seqs.nunique()
            
            if num_unique > 1:
                variable_seq_cols.append(col)
                if num_unique <= 5: 
                    counts_dict = valid_seqs.value_counts().to_dict()
                    counts_str = ", ".join([f"'{k}' ({v} samples)" for k, v in counts_dict.items()])
                    print(f"  -> Kept '{col}' (Low variance: {num_unique} unique seqs). Breakdown: {counts_str}")
            else:
                print(f"  -> Dropping '{col}' (constant sequence across all samples)")
                
        seq_cols = variable_seq_cols
        print(f"Retained {len(seq_cols)} sequence columns with variance.\n")
        
        granular_seq_cols, composite_cols = filter_redundant_sequences(df_feat, seq_cols)
        print(f"Filtered out {len(composite_cols)} composite/redundant regions.")
        seq_cols = granular_seq_cols
        
        if 'G4S Linker1_HCK' in seq_cols:
            print("  -> Dropping 'G4S Linker1_HCK' (manual override: near-zero variance).")
            seq_cols.remove('G4S Linker1_HCK')
        print(f"Proceeding with {len(seq_cols)} granular building blocks: {seq_cols}\n")
    
    # --- 🌟 NEW: CONSTRUCT GLOBAL CHAINS FOR CONTEXTUAL EXTRACTION ---
    vh_ordered_cols = ['seq_frh1', 'seq_cdrh1', 'seq_frh2', 'seq_cdrh2', 'seq_frh3', 'seq_cdrh3', 'seq_frh4'] 
    vl_ordered_cols = ['seq_frl1', 'seq_cdrl1', 'seq_frl2', 'seq_cdrl2', 'seq_frl3', 'seq_cdrl3', 'seq_frl4']

    def build_full_seq(row, cols):
        parts = []
        for c in cols:
            if c in df_feat.columns and pd.notna(row[c]):
                val = str(row[c]).strip().upper()
                if val != 'NAN' and val != 'NONE':
                    parts.append(val)
        return "".join(parts) if parts else 'NAN'

    if not is_inference: print("Stitching together global sequences (VH, VL, Fv)...")
    df_feat['Global_VH'] = df_feat.apply(lambda r: build_full_seq(r, vh_ordered_cols), axis=1)
    df_feat['Global_VL'] = df_feat.apply(lambda r: build_full_seq(r, vl_ordered_cols), axis=1)
    df_feat['Global_Fv'] = df_feat.apply(lambda r: r['Global_VH'] + r['Global_VL'] if r['Global_VH'] != 'NAN' and r['Global_VL'] != 'NAN' else 'NAN', axis=1)

    global_seq_cols = ['Global_VH', 'Global_VL', 'Global_Fv']
    all_seq_cols_to_process = seq_cols + global_seq_cols

    aaindex_db, aaindex_desc = load_aaindex()
    generated_features = []
    new_columns = {}
    
    # --- NEW: Dynamic Cache Folders ---
    # Appending the row count and custom tag prevents cache invalidation loops
    cache_folder = f"{dataset_name}_{len(df_feat)}samples"
    if cache_tag:
        cache_folder += f"_{cache_tag}"
        
    cache_dir = os.path.join("feature_cache", cache_folder)
    os.makedirs(cache_dir, exist_ok=True)
    if not is_inference: print(f"Using cache directory: '{cache_dir}'")
    
    def _load_valid_cache(cache_path, expected_len):
        if os.path.exists(cache_path):
            try:
                with np.load(cache_path) as cached_data:
                    if len(cached_data.files) > 0:
                        cached_len = len(cached_data[cached_data.files[0]])
                        
                        # The cache matches your current dataset size
                        if cached_len == expected_len:
                            return {k: cached_data[k] for k in cached_data.files}
                        
                        # 🌟 NEW: The cache exists, but the size changed!
                        else:
                            filename = os.path.basename(cache_path)
                            print(f"  -> ⚠️ Cache mismatch ({filename}): Cached={cached_len}, Dataset={expected_len}. Re-extracting...")
                            
            except Exception as e:
                print(f"  -> ⚠️ Cache corrupted or unreadable ({os.path.basename(cache_path)}). Re-extracting...")
        
        # File doesn't exist, or it failed the checks above
        return None
        
    expected_dataset_len = len(df_feat)
    
    # 🌟 NEW: Extract AAC, AAIndex, and ESM for BOTH subregions and the new global chains!
    for col in all_seq_cols_to_process:
        seqs = df_feat[col].astype(str).str.replace(r'\s+|,', '', regex=True).str.upper()

        # # 1. Length Feature Cache
        # len_cache_file = os.path.join(cache_dir, f"{col}_length_features.npz")
        # cached_dict = _load_valid_cache(len_cache_file, expected_dataset_len) if not is_inference else None
        
        # if cached_dict:
        #     for k, v in cached_dict.items():
        #         new_columns[k] = v
        #         generated_features.append(k)
        # else:
        #     len_features = {}
        #     len_col = f'{col}_Length'
        #     len_features[len_col] = seqs.apply(lambda x: len(x) if x != 'NAN' else 0).to_numpy()
        #     if not is_inference: np.savez(len_cache_file, **len_features)
        #     for k, v in len_features.items():
        #         new_columns[k] = v
        #         generated_features.append(k)
                
        # 2. Amino Acid Composition (AAC) Cache
        aac_cache_file = os.path.join(cache_dir, f"{col}_aac_features.npz")
        cached_dict = _load_valid_cache(aac_cache_file, expected_dataset_len) if not is_inference else None
        
        if cached_dict:
            for k, v in cached_dict.items():
                new_columns[k] = v
                generated_features.append(k)
        else:
            aac_features = {}
            for aa in standard_aas:
                aa_col = f'{col}_AAC_{aa}'
                aac_features[aa_col] = seqs.apply(lambda x: x.count(aa) if x != 'NAN' else 0).to_numpy()
            if not is_inference: np.savez(aac_cache_file, **aac_features)
            for k, v in aac_features.items():
                new_columns[k] = v
                generated_features.append(k)
                
        # 3. AAindex Physiological Properties Cache
        aaindex_cache_file = os.path.join(cache_dir, f"{col}_aaindex_features.npz")
        cached_dict = _load_valid_cache(aaindex_cache_file, expected_dataset_len) if not is_inference else None
        
        if cached_dict:
            for k, v in cached_dict.items():
                new_columns[k] = v
                generated_features.append(k)
        else:
            aaindex_features = {}
            if aaindex_db:
                for code, prop_map in aaindex_db.items():
                    col_name = f'{col}_AAindex_{code}'
                    
                    def calc_prop(seq, pmap=prop_map):
                        if seq == 'NAN' or not seq: return 0
                        vals = [pmap.get(aa) for aa in seq]
                        vals = [v for v in vals if v is not None and not np.isnan(v)]
                        return sum(vals)/len(vals) if vals else 0
                        
                    aaindex_features[col_name] = seqs.apply(calc_prop).to_numpy()
            
            if aaindex_features:
                if not is_inference: np.savez(aaindex_cache_file, **aaindex_features)
                for k, v in aaindex_features.items():
                    new_columns[k] = v
                    generated_features.append(k)
                    
        # 4. ESM-2 Protein Language Model (PLM) Cache
        if ESM_AVAILABLE:
            # Generate a distinct tag based on the chosen ESM model string to prevent cache collisions
            if "8M" in esm_model_name: esm_tag = "ESM_Small_8M"
            elif "35M" in esm_model_name: esm_tag = "ESM_Medium_35M"
            elif "150M" in esm_model_name: esm_tag = "ESM_Large_150M"
            elif "650M" in esm_model_name: esm_tag = "ESM_Big_650M"
            elif "3B" in esm_model_name: esm_tag = "ESM_Massive_3B"
            else: esm_tag = "ESM_Custom"
            
            esm_cache_file = os.path.join(cache_dir, f"{col}_{esm_tag}_features.npz")
            cached_dict = _load_valid_cache(esm_cache_file, expected_dataset_len) if not is_inference else None
            
            if cached_dict:
                for k, v in cached_dict.items():
                    new_columns[k] = v
                    generated_features.append(k)
            else:
                if is_inference:
                    print(f"  -> Extracting ESM-2 for modified sequence '{col}'...")
                else:
                    print(f"  -> Extracting ESM-2 Embeddings for '{col}' using {esm_model_name} (This may take a moment)...")
                
                import torch
                from transformers import EsmModel, EsmTokenizer
                
                tokenizer = EsmTokenizer.from_pretrained(esm_model_name)
                model = EsmModel.from_pretrained(esm_model_name)
                model.eval()
                
                esm_features_dict = {}
                embeddings = []
                
                for seq in seqs:
                    if seq == 'NAN' or not seq or len(seq) < 2:
                        embeddings.append(np.zeros(model.config.hidden_size))
                        continue
                        
                    inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        
                    hidden_states = outputs.last_hidden_state[0]
                    if hidden_states.shape[0] > 2:
                        repr_vector = hidden_states[1:-1].mean(dim=0).numpy()
                    else:
                        repr_vector = hidden_states.mean(dim=0).numpy()
                        
                    embeddings.append(repr_vector)
                    
                embeddings = np.array(embeddings)
                
                for i in range(embeddings.shape[1]):
                    feat_name = f'{col}_{esm_tag}_{i}'
                    esm_features_dict[feat_name] = embeddings[:, i]
                    
                if not is_inference: np.savez(esm_cache_file, **esm_features_dict)
                for k, v in esm_features_dict.items():
                    new_columns[k] = v
                    generated_features.append(k)
                    
        # 5. NEW: PROPERMAB 3D Structural Features Cache
        if PROPERMAB_AVAILABLE:
            # 🌟 NEW: Use the global sequences we stitched at the very top!
            hc_col = 'Global_VH'
            lc_col = 'Global_VL'
            
            if hc_col in df_feat.columns and lc_col in df_feat.columns:
                propermab_cache_file = os.path.join(cache_dir, f"propermab_features.npz")
                cached_dict = _load_valid_cache(propermab_cache_file, expected_dataset_len) if not is_inference else None
                
                if cached_dict:
                    for k, v in cached_dict.items():
                        new_columns[k] = v
                        generated_features.append(k)
                else:
                    if is_inference:
                        print(f"  -> Extracting 3D Propermab features...")
                    else:
                        print(f"  -> Extracting 3D Structural Features using PROPERMAB (This takes time due to ABodyBuilder2 folding)...")
                    
                    pm_features_dict = {}
                    all_pm_results = []
                    total_samples = len(df_feat)
                    
                    for i, (idx, row) in enumerate(df_feat.iterrows()):
                        print(f"     -> [Sample {i + 1}/{total_samples}] Predicting 3D structure and extracting features...")
                        
                        h_seq = str(row[hc_col]).strip()
                        l_seq = str(row[lc_col]).strip()
                        
                        if h_seq == 'NAN' or l_seq == 'NAN' or not h_seq or not l_seq:
                            print(f"        -> Skipped (Missing full Heavy or Light sequence)")
                            all_pm_results.append({})
                            continue
                            
                        try:
                            # get_all_mol_features runs ABodyBuilder2 and calculates features like hyd_asa, pos_ann_index, etc.
                            mol_feat = feature_utils.get_all_mol_features(h_seq, l_seq, num_runs=1)
                            # mol_feat is a dict of lists. Take the first run [0]
                            flat_feat = {f"Propermab_{k}": v[0] for k, v in mol_feat.items()}
                            all_pm_results.append(flat_feat)
                        except Exception as e:
                            print(f"        -> Warning: Propermab structural prediction failed on row {idx}: {e}")
                            all_pm_results.append({})
                            
                    # Convert list of dicts to a dict of arrays
                    if all_pm_results:
                        all_keys = set().union(*(d.keys() for d in all_pm_results))
                        for k in all_keys:
                            # Extract the feature for each row, defaulting to NaN if it failed
                            arr = np.array([d.get(k, np.nan) for d in all_pm_results], dtype=float)
                            # Fill NaNs with column mean for robustness
                            if np.isnan(arr).any():
                                arr[np.isnan(arr)] = np.nanmean(arr)
                            pm_features_dict[k] = arr
                    
                    if not is_inference and pm_features_dict:
                        np.savez(propermab_cache_file, **pm_features_dict)
                    for k, v in pm_features_dict.items():
                        new_columns[k] = v
                        generated_features.append(k)
            else:
                if not is_inference:
                    print(f"  -> PROPERMAB is installed, but sequence construction failed.")

        # # ==========================================
        # # 🌟 NEW: GEORGIEV FEATURES EXTRACTION
        # # ==========================================
        # georgiev_cache_file = os.path.join(cache_dir, f"{col}_georgiev_features.npz")
        # cached_dict = _load_valid_cache(georgiev_cache_file, expected_dataset_len)
        
        # if cached_dict:
        #     print(f"  -> Loading Georgiev from cache: {col}")
        #     for k, v in cached_dict.items():
        #         new_columns[k] = v
        #         if k not in generated_features:
        #             generated_features.append(k)
        # else:
        #     print(f"  -> Calculating Georgiev for: {col}")
        #     georgiev_features = []
        #     for seq in df_feat[col]:
        #         seq_str = str(seq).strip().upper()
        #         if seq_str == 'NAN' or seq_str == 'NONE' or len(seq_str) == 0:
        #             georgiev_features.append([0.0] * 19)
        #         else:
        #             seq_georgiev = []
        #             for aa in seq_str:
        #                 if aa in GEORGIEV_DICT:
        #                     seq_georgiev.append(GEORGIEV_DICT[aa])
                    
        #             if len(seq_georgiev) > 0:
        #                 # Average the 19 dimensions across the sequence length
        #                 avg_georgiev = np.mean(seq_georgiev, axis=0).tolist()
        #                 georgiev_features.append(avg_georgiev)
        #             else:
        #                 georgiev_features.append([0.0] * 19)
            
        #     georgiev_features = np.array(georgiev_features)
        #     new_features = {}
        #     for i in range(19):
        #         feat_name = f"{col}_Georgiev_PC{i+1}"
        #         new_features[feat_name] = georgiev_features[:, i]
                
        #     if not is_inference: np.savez(georgiev_cache_file, **new_features)
        #     for k, v in new_features.items():
        #         new_columns[k] = v
        #         if k not in generated_features:
        #             generated_features.append(k)

    if new_columns:
        new_features_df = pd.DataFrame(new_columns)
        df_feat = pd.concat([df_feat, new_features_df], axis=1)
        
    return df_feat, seq_cols, generated_features, aaindex_desc

def get_model(model_name, n_features, transform_type=None, weight_col=None):
    """Returns a scikit-learn model/pipeline based on the requested name."""
    scoring_dict = {
        'rmse': 'neg_root_mean_squared_error',
        'mae': 'neg_mean_absolute_error',
        'r2': 'r2',
        'spearman': make_scorer(custom_spearman)
    }
    
    if model_name == 'RandomForest':
        base_model = RandomForestRegressor(n_estimators=100, random_state=42)
        param_grid = {}
        
    elif model_name == 'PLSRegression':
        base_model = Pipeline([
            ('vt', VarianceThreshold()), 
            ('scaler', StandardScaler()),
            ('pls', PLSRegression())
        ])
        max_comp = min(10, n_features)
        if max_comp >= 2:
            param_grid = {'pls__n_components': range(2, max_comp + 1)}
        else:
            base_model.set_params(pls__n_components=1)
            param_grid = {}
            
    elif model_name == 'ElasticNet':
        base_model = Pipeline([
            ('vt', VarianceThreshold()), 
            ('scaler', StandardScaler()),
            ('enet', ElasticNet(max_iter=10000, random_state=42))
        ])
        param_grid = {
            'enet__alpha': [0.01, 0.1, 1.0, 10.0],
            'enet__l1_ratio': [0.1, 0.5, 0.9]
        }

    elif model_name == 'SVR':
        base_model = Pipeline([
            ('vt', VarianceThreshold()), 
            ('scaler', StandardScaler()),
            ('svr', SVR())
        ])
        param_grid = {
            'svr__kernel': ['linear', 'rbf'],
            'svr__C': [0.001, 0.01, 0.1, 1.0]
        }

    elif model_name == 'XGBoost':
        base_model = Pipeline([
            ('vt', VarianceThreshold()), 
            ('scaler', StandardScaler()),
            ('xgboost', XGBRegressor(random_state=42, objective='reg:squarederror'))
        ])
        # Use the same prefix ('model__') as your other algorithms
        param_grid = {
            'xgboost__max_depth': [2, 3],
            'xgboost__learning_rate': [0.05, 0.1],
            'xgboost__n_estimators': [50, 100],
            'xgboost__colsample_bytree': [0.3, 0.8]
        }   
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")

    # Wrap the base model using our new Custom Meta-Estimator!
    model = SelfContainedTargetTransformRegressor(
        regressor=base_model, 
        transform_type=transform_type, 
        weight_col=weight_col
    )
    
    # Update param grid to point to the nested regressor
    param_grid = {f'regressor__{k}': v for k, v in param_grid.items()}

    if param_grid:
        return GridSearchCV(model, param_grid, cv=3, scoring=scoring_dict, refit='spearman', n_jobs=1)
    else:
        return model

def plot_target_distribution(y, target_col, transform_type, output_dir, optimal_lambda=None, feature_tag=""):
    """Plots the target distribution with enhanced statistical overlays and percentiles."""
    transform_suffix = f"_{transform_type}" if transform_type else ""
    feat_suffix = f"_{feature_tag}" if feature_tag else ""
    
    if transform_type == 'log1p':
        y_transformed = np.log1p(y)
        transform_title = "Log1p"
    elif transform_type == 'weighted-yeo-johnson' and optimal_lambda is not None:
        pt = CustomYeoJohnsonTransformer(lmbda=optimal_lambda)
        y_transformed = pt.transform(y.values.reshape(-1, 1)).flatten()
        y_transformed = pd.Series(y_transformed) # Keep as series for easy pandas stats
        transform_title = f"Weighted Yeo-Johnson (λ={optimal_lambda:.2f})"
    elif transform_type in ['box-cox', 'yeo-johnson']:
        pt = PowerTransformer(method=transform_type)
        y_transformed = pt.fit_transform(y.values.reshape(-1, 1)).flatten()
        y_transformed = pd.Series(y_transformed)
        transform_title = transform_type.title()
    else:
        y_transformed = y

    fig, axes = plt.subplots(1, 2 if transform_type else 1, figsize=(16 if transform_type else 8, 7))
    
    ax_orig = axes[0] if transform_type else axes
    ax_trans = axes[1] if transform_type else None

    # --- Plot 1: Original Data ---
    sns.histplot(y, kde=True, ax=ax_orig, color='royalblue', bins=15, alpha=0.6)
    
    # Calculate key statistics
    mean_val = y.mean()
    median_val = y.median()
    p20 = np.percentile(y, 20)
    p80 = np.percentile(y, 80)
    
    # Overlay lines for Mean, Median, and our Enrichment Percentiles
    ax_orig.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax_orig.axvline(median_val, color='darkorange', linestyle='-', linewidth=2, label=f'Median: {median_val:.2f}')
    ax_orig.axvline(p20, color='gray', linestyle=':', linewidth=2, label=f'20th Pctl: {p20:.2f}')
    ax_orig.axvline(p80, color='gray', linestyle=':', linewidth=2, label=f'80th Pctl: {p80:.2f}')
    
    ax_orig.set_title(f'Original Distribution\n{target_col}', fontsize=14)
    ax_orig.set_xlabel(target_col, fontsize=12)
    ax_orig.set_ylabel('Frequency', fontsize=12)
    ax_orig.legend(loc='upper right')

    # Add a clean statistics text box
    stats_text = (
        f"N = {len(y)}\n"
        f"Std Dev = {y.std():.2f}\n"
        f"Skewness = {y.skew():.2f}\n"
        f"Min = {y.min():.2f}\n"
        f"Max = {y.max():.2f}"
    )
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
    ax_orig.text(0.05, 0.95, stats_text, transform=ax_orig.transAxes, 
                 fontsize=11, verticalalignment='top', bbox=props)

    # --- Plot 2: Transformed Data (If Applicable) ---
    if transform_type:
        sns.histplot(y_transformed, kde=True, ax=ax_trans, color='seagreen', bins=15, alpha=0.6)
        
        t_mean = y_transformed.mean()
        t_median = y_transformed.median()
        t_p20 = np.percentile(y_transformed, 20)
        t_p80 = np.percentile(y_transformed, 80)
        
        ax_trans.axvline(t_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {t_mean:.2f}')
        ax_trans.axvline(t_median, color='darkorange', linestyle='-', linewidth=2, label=f'Median: {t_median:.2f}')
        ax_trans.axvline(t_p20, color='gray', linestyle=':', linewidth=2, label=f'20th Pctl: {t_p20:.2f}')
        ax_trans.axvline(t_p80, color='gray', linestyle=':', linewidth=2, label=f'80th Pctl: {t_p80:.2f}')
        
        ax_trans.set_title(f'{transform_title} Transformed Distribution\n{target_col}', fontsize=14)
        ax_trans.set_xlabel(f'Transformed {target_col}', fontsize=12)
        ax_trans.set_ylabel('Frequency', fontsize=12)
        ax_trans.legend(loc='upper right')
        
        t_stats_text = (
            f"N = {len(y_transformed)}\n"
            f"Std Dev = {y_transformed.std():.2f}\n"
            f"Skewness = {pd.Series(y_transformed).skew():.2f}\n"
            f"Min = {y_transformed.min():.2f}\n"
            f"Max = {y_transformed.max():.2f}"
        )
        ax_trans.text(0.05, 0.95, t_stats_text, transform=ax_trans.transAxes, 
                      fontsize=11, verticalalignment='top', bbox=props)
                      
        fig.suptitle(f'Target Distribution Comparison: Original vs. {transform_title}', fontsize=16, y=0.98)
    else:
        fig.suptitle(f'Target Distribution Analysis: {target_col}', fontsize=16, y=0.98)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plot_filename = os.path.join(output_dir, f'target_dist_{target_col}{transform_suffix}{feat_suffix}.png')
    plt.savefig(plot_filename, dpi=300, facecolor='white')
    plt.close()
    print(f"Saved target distribution plot to '{plot_filename}'")

def plot_best_model_diagnostics(X, y, combo_name, model_name, target_col, output_dir, final_estimator, best_params=None, threshold=None, transform_type=None, feature_tag="", type_labels=None, type_col_name=None, prefix="", top_quantile=0.2):
    """Generates diagnostic and learning curve plots for the best model."""
    transform_suffix = f"_{transform_type}" if transform_type else ""
    feat_suffix = f"_{feature_tag}" if feature_tag else ""
    title_tag = f" ({transform_type.title()} Transformed)" if transform_type else ""
    
    print(f"\nGenerating diagnostic plots for the best model ({combo_name}) using {model_name} on {target_col}{title_tag}...")
    
    # 🌟 EXPANDED: 3x3 Grid (9 slots total)
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    fig.suptitle(f'Diagnostic Plots: Best Model ({combo_name} | {model_name} | {target_col}){title_tag}', fontsize=18, y=0.98)
    
    try:
        loo = LeaveOneOut()
        cv_preds = cross_val_predict(final_estimator, X, y, cv=loo, n_jobs=-1)
        residuals = y - cv_preds
        
        calc_r2 = r2_score(y, cv_preds)
        calc_rmse = np.sqrt(mean_squared_error(y, cv_preds))
        calc_mae = mean_absolute_error(y, cv_preds)
        calc_spearman = custom_spearman(y, cv_preds)
        
        if threshold is None:
            threshold = np.median(y)
            
        y_binary = (y >= threshold).astype(int)
        cv_preds_binary = (cv_preds >= threshold).astype(int)
        
        acc = accuracy_score(y_binary, cv_preds_binary)
        prec = precision_score(y_binary, cv_preds_binary, zero_division=0)
        rec = recall_score(y_binary, cv_preds_binary, zero_division=0)
        f1 = f1_score(y_binary, cv_preds_binary, zero_division=0)
        cm = confusion_matrix(y_binary, cv_preds_binary)
        
        try:
            roc_auc = roc_auc_score(y_binary, cv_preds)
            pr_auc = average_precision_score(y_binary, cv_preds)
            auc_text = f"ROC-AUC: {roc_auc:.2f} | PR-AUC: {pr_auc:.2f}\n"
        except ValueError:
            auc_text = "ROC/PR-AUC: N/A (Single Class)\n"
        
        metrics_text = (
            f"Spearman: {calc_spearman:.3f}\n"
            f"R²: {calc_r2:.3f}\n"
            f"RMSE: {calc_rmse:.2f}\n"
            f"MAE: {calc_mae:.2f}"
        )
        
        metrics_text += f"\n\nBinary Eval (Cutoff: {threshold:.2f}):\n{auc_text}Acc: {acc:.2f} | F1: {f1:.2f} | P: {prec:.2f} | R: {rec:.2f}"
        
        if best_params:
            params_str = "\n".join([f"{k.split('__')[-1]}: {v}" for k, v in best_params.items()])
            metrics_text += f"\n\nOptimal Params:\n{params_str}"
        
        # ==========================================
        # ROW 1: REGRESSION ACCURACY & ERROR
        # ==========================================
        
        classification_labels = []
        for actual, pred in zip(y, cv_preds):
            if actual >= threshold and pred >= threshold:
                classification_labels.append('True Positive (TP)')
            elif actual < threshold and pred < threshold:
                classification_labels.append('True Negative (TN)')
            elif actual < threshold and pred >= threshold:
                classification_labels.append('False Positive (FP)')
            else:
                classification_labels.append('False Negative (FN)')
                
        plot_df = pd.DataFrame({'Actual': y, 'Predicted': cv_preds, 'Classification': classification_labels})
        
        style_col = None
        if type_labels is not None:
            style_col = type_col_name if type_col_name else 'Antibody Type'
            plot_df[style_col] = type_labels
        
        custom_palette = {
            'True Negative (TN)': '#2ca02c', 'True Positive (TP)': '#d62728',
            'False Positive (FP)': '#ff7f0e', 'False Negative (FN)': '#1f77b4'
        }
        
        # --- Plot 1: Predicted vs Actual ---
        sns.scatterplot(data=plot_df, x='Actual', y='Predicted', hue='Classification', 
                        style=style_col, palette=custom_palette, ax=axes[0, 0], alpha=0.8, edgecolor='k', s=60)
        axes[0, 0].axvline(threshold, color='gray', linestyle=':', linewidth=1.5, label=f'Threshold ({threshold:.1f})')
        axes[0, 0].axhline(threshold, color='gray', linestyle=':', linewidth=1.5)
        min_val = min(y.min(), cv_preds.min())
        max_val = max(y.max(), cv_preds.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        axes[0, 0].set_title(f'Predicted vs Actual {target_col} (LOO CV)')
        axes[0, 0].set_xlabel(f'Actual {target_col}')
        axes[0, 0].set_ylabel(f'Predicted {target_col} (Out-of-Fold)')
        axes[0, 0].legend(loc='lower right', fontsize=9)
        
        # --- Plot 2: Residuals vs Predicted ---
        plot_df_res = pd.DataFrame({'Predicted': cv_preds, 'Residuals': residuals})
        if style_col:
            plot_df_res[style_col] = type_labels
        sns.scatterplot(data=plot_df_res, x='Predicted', y='Residuals', 
                        style=style_col, color='purple', ax=axes[0, 1], alpha=0.7, edgecolor='k', s=60)
        axes[0, 1].axhline(0, color='r', linestyle='--')
        axes[0, 1].set_title('Residuals vs Predicted')
        axes[0, 1].set_xlabel(f'Predicted {target_col}')
        axes[0, 1].set_ylabel('Residuals (Actual - Predicted)')
        if style_col:
            axes[0, 1].legend(loc='lower right', fontsize=9)
            
        # --- Plot 3: Distribution of Residuals ---
        sns.histplot(residuals, kde=True, ax=axes[0, 2], color='purple', bins=15)
        axes[0, 2].axvline(0, color='r', linestyle='--')
        axes[0, 2].set_title('Distribution of Residuals')
        axes[0, 2].set_xlabel(f'Residual Error {target_col}')
        axes[0, 2].set_ylabel('Frequency')

        # ==========================================
        # ROW 2: TRANSLATIONAL VALUE & RANKING
        # ==========================================

        # --- Plot 4: Confusion Matrix ---
        dummy_matrix = np.array([[0, 1], [2, 3]])
        cm_cmap = ListedColormap(['#2ca02c', '#ff7f0e', '#1f77b4', '#d62728'])
        sns.heatmap(dummy_matrix, annot=cm, fmt='d', cmap=cm_cmap, ax=axes[1, 0], cbar=False,
                    xticklabels=[f'<{threshold:.1f}', f'>={threshold:.1f}'],
                    yticklabels=[f'<{threshold:.1f}', f'>={threshold:.1f}'],
                    annot_kws={"size": 22, "weight": "bold", "color": "white"})
        axes[1, 0].set_title(f'Classification Confusion Matrix\n(Threshold = {threshold:.2f})')
        axes[1, 0].set_xlabel('Predicted Class')
        axes[1, 0].set_ylabel('Actual Class')

        # --- Plot 5: 🌟 CALLING THE ENRICHMENT HELPER FUNCTION ---
        # plot_enrichment_comparison(y, cv_preds, target_col, ax=axes[1, 1], top_quantile=top_quantile)

        # --- Plot 6: 🌟 CALLING THE HIT RATE HELPER FUNCTION ---
        # plot_hit_rate_curve(y, cv_preds, target_col, ax=axes[1, 1], top_quantile=top_quantile)
        plot_enrichment_comparison(y, cv_preds, target_col, ax=axes[1, 1], top_quantile=top_quantile)
        # ==========================================
        # ROW 3: SUMMARY & METRICS DASHBOARD
        # ==========================================
        
        # axes[2, 0].axis('off')
        # axes[2, 2].axis('off')
        
        axes[1, 2].axis('off')  
        props = dict(boxstyle='round,pad=1', facecolor='#f8f9fa', alpha=1.0, edgecolor='gray', linewidth=2)
        axes[1, 2].text(0.5, 0.5, metrics_text, transform=axes[1, 2].transAxes,
                        fontsize=14, verticalalignment='center', horizontalalignment='center', bbox=props)
        axes[1, 2].set_title("Model Performance Summary", fontsize=16, pad=20)
        
    except Exception as e:
        print(f"Warning: Could not generate LOO predictions due to mathematical failure: {e}")
        for row in range(2):
            for col in range(3):
                axes[row, col].set_title("Plot Failed")
                axes[row, col].text(0.5, 0.5, f"Error: {e}", ha='center', va='center', wrap=True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plot_filename = os.path.join(output_dir, f'best_{prefix}{model_name}_{target_col}{transform_suffix}{feat_suffix}_diagnostics.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved '{plot_filename}'")

    print("Generating comprehensive learning curves (this takes a moment)...")
    fig_lc, axes_lc = plt.subplots(2, 2, figsize=(16, 12))
    fig_lc.suptitle(f'Learning Curves: Best Model ({combo_name} | {model_name} | {target_col}){title_tag}', fontsize=16)
    
    metrics_to_plot = {
        'Negative RMSE': 'neg_root_mean_squared_error',
        'Negative MAE': 'neg_mean_absolute_error',
        'R2 Score': 'r2',
        'Spearman Correlation': make_scorer(custom_spearman)
    }
    
    try:
        for ax, (name, scorer) in zip(axes_lc.flatten(), metrics_to_plot.items()):
            LearningCurveDisplay.from_estimator(
                final_estimator, X, y, cv=5, n_jobs=-1,
                train_sizes=np.linspace(0.2, 1.0, 10),
                scoring=scorer,
                score_name=name,
                error_score=np.nan,
                ax=ax
            )
            ax.set_title(f'Learning Curve ({name})')
            ax.legend(loc='best')
    except Exception as e:
        print(f"    -> Warning: Learning curves failed due to math error: {e}")
        for ax in axes_lc.flatten():
            ax.set_title("Learning Curve Failed")
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    lc_filename = os.path.join(output_dir, f'best_{prefix}{model_name}_{target_col}{transform_suffix}{feat_suffix}_learning_curves.png')
    plt.savefig(lc_filename, dpi=300, facecolor='white')
    plt.close()
    print(f"Saved '{lc_filename}'")

def plot_best_model_diagnostics2(X, y, combo_name, model_name, target_col, output_dir, final_estimator, best_params=None, threshold=None, transform_type=None, feature_tag="", type_labels=None, type_col_name=None, prefix="", top_quantile=0.2):
    """Generates diagnostic and learning curve plots for the best model."""
    transform_suffix = f"_{transform_type}" if transform_type else ""
    feat_suffix = f"_{feature_tag}" if feature_tag else ""
    title_tag = f" ({transform_type.title()} Transformed)" if transform_type else ""
    
    print(f"\nGenerating diagnostic plots for the best model ({combo_name}) using {model_name} on {target_col}{title_tag}...")
    
    # 🌟 NEW: Clean 1x2 Grid with performance summary overlaid on the scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f'Diagnostic Plots: Best Model ({combo_name} | {model_name} | {target_col}){title_tag}', fontsize=18, y=1.05)
    
    try:
        loo = LeaveOneOut()
        cv_preds = cross_val_predict(final_estimator, X, y, cv=loo, n_jobs=-1)
        
        calc_r2 = r2_score(y, cv_preds)
        calc_rmse = np.sqrt(mean_squared_error(y, cv_preds))
        calc_mae = mean_absolute_error(y, cv_preds)
        calc_spearman = custom_spearman(y, cv_preds)
        
        metrics_text = (
            f"Spearman Correlation: {calc_spearman:.3f}\n"
            f"R² Score: {calc_r2:.3f}\n"
            f"RMSE: {calc_rmse:.2f}\n"
            f"MAE: {calc_mae:.2f}"
        )
        
        if best_params:
            params_str = "\n".join([f"{k.split('__')[-1]}: {v}" for k, v in best_params.items()])
            metrics_text += f"\n\nOptimal Hyperparameters:\n{params_str}"
        
        # ==========================================
        # REGRESSION ACCURACY, SUMMARY & ENRICHMENT
        # ==========================================
        
        # --- Plot 1: Predicted vs Actual (Color-Coded by CSV Column) ---
        plot_df = pd.DataFrame({'Actual': y, 'Predicted': cv_preds})
        
        hue_col = None
        if type_labels is not None and type_col_name is not None:
            hue_col = type_col_name
            plot_df[hue_col] = type_labels
            
        sns.scatterplot(data=plot_df, x='Actual', y='Predicted', hue=hue_col, 
                        palette='tab10' if hue_col else None, ax=axes[0], alpha=0.8, edgecolor='k', s=80)
        
        min_val = min(y.min(), cv_preds.min())
        max_val = max(y.max(), cv_preds.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label="Perfect Prediction (y=x)")
        axes[0].set_title(f'Predicted vs Actual {target_col} (LOO CV)')
        axes[0].set_xlabel(f'Actual {target_col}')
        axes[0].set_ylabel(f'Predicted {target_col} (Out-of-Fold)')
        
        # Overlay the performance summary text box inside the upper left of the scatter plot
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
        axes[0].text(0.05, 0.95, metrics_text, transform=axes[0].transAxes,
                     fontsize=10, verticalalignment='top', bbox=props)

        # Move legend to lower right to avoid colliding with the text box
        if hue_col:
            axes[0].legend(loc='lower right', fontsize=10, title=hue_col)
        else:
            axes[0].legend(loc='lower right', fontsize=10)

        # --- Plot 2: Enrichment Curve ---
        plot_enrichment_comparison(y, cv_preds, target_col, ax=axes[1], top_quantile=top_quantile)
        
    except Exception as e:
        print(f"Warning: Could not generate LOO predictions due to mathematical failure: {e}")
        for col in range(2):
            axes[col].set_title("Plot Failed")
            axes[col].text(0.5, 0.5, f"Error: {e}", ha='center', va='center', wrap=True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plot_filename = os.path.join(output_dir, f'best_{prefix}{model_name}_{target_col}{transform_suffix}{feat_suffix}_diagnostics.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved '{plot_filename}'")

    print("Generating comprehensive learning curves (this takes a moment)...")
    fig_lc, axes_lc = plt.subplots(2, 2, figsize=(16, 12))
    fig_lc.suptitle(f'Learning Curves: Best Model ({combo_name} | {model_name} | {target_col}){title_tag}', fontsize=16)
    
    metrics_to_plot = {
        'Negative RMSE': 'neg_root_mean_squared_error',
        'Negative MAE': 'neg_mean_absolute_error',
        'R2 Score': 'r2',
        'Spearman Correlation': make_scorer(custom_spearman)
    }
    
    try:
        for ax, (name, scorer) in zip(axes_lc.flatten(), metrics_to_plot.items()):
            LearningCurveDisplay.from_estimator(
                final_estimator, X, y, cv=5, n_jobs=-1,
                train_sizes=np.linspace(0.2, 1.0, 10),
                scoring=scorer,
                score_name=name,
                error_score=np.nan,
                ax=ax
            )
            ax.set_title(f'Learning Curve ({name})')
            ax.legend(loc='best')
    except Exception as e:
        print(f"    -> Warning: Learning curves failed due to math error: {e}")
        for ax in axes_lc.flatten():
            ax.set_title("Learning Curve Failed")
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    lc_filename = os.path.join(output_dir, f'best_{prefix}{model_name}_{target_col}{transform_suffix}{feat_suffix}_learning_curves.png')
    plt.savefig(lc_filename, dpi=300, facecolor='white')
    plt.close()
    print(f"Saved '{lc_filename}'")


def plot_enrichment_comparison(y_true, y_pred, target_col, ax=None, top_quantile=0.2):
    """Plots a cumulative gain / enrichment curve vs random baseline using absolute numbers."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        
    df_eval = pd.DataFrame({'true': y_true, 'pred': y_pred})
    target_str = target_col.lower()
    lower_is_better = any(term in target_str for term in ['hmw', 'agg', 'viscosity', 'lmw', 'hcp', 'clearance', 'polyreactivity'])
    
    if lower_is_better:
        df_eval = df_eval.sort_values(by='pred', ascending=True).reset_index(drop=True)
        top_quantile_threshold = np.percentile(y_true, 100 * top_quantile)
        df_eval['is_top_performer'] = df_eval['true'] <= top_quantile_threshold
        direction_label = f"Lowest {int(top_quantile*100)}%"
    else:
        df_eval = df_eval.sort_values(by='pred', ascending=False).reset_index(drop=True)
        top_quantile_threshold = np.percentile(y_true, 100 * (1 - top_quantile))
        df_eval['is_top_performer'] = df_eval['true'] >= top_quantile_threshold
        direction_label = f"Highest {int(top_quantile*100)}%"
        
    total_top_performers = df_eval['is_top_performer'].sum()
    total_samples = len(df_eval)
    
    if total_top_performers > 0:
        # Convert to absolute counts
        df_eval['hits_found'] = df_eval['is_top_performer'].cumsum()
        df_eval['samples_screened'] = df_eval.index + 1
        
        # Random baseline is a straight line from (0,0) to (Total Samples, Total Hits)
        ax.plot([0, total_samples], [0, total_top_performers], 'k--', label='Random Selection', alpha=0.6, linewidth=2)
        
        # Model performance curve
        ax.plot(df_eval['samples_screened'], df_eval['hits_found'], 'b-', label='ML Model Ranking', linewidth=2.5)
        
        # Shade the area where the model outperforms random guessing
        baseline_y = df_eval['samples_screened'] * (total_top_performers / total_samples)
        ax.fill_between(df_eval['samples_screened'], baseline_y, df_eval['hits_found'], 
                        where=(df_eval['hits_found'] > baseline_y), color='blue', alpha=0.1)
        
        ax.set_title(f'Enrichment: Finding the {direction_label}', fontsize=12)
        ax.set_xlabel('Total Antibodies Screened (Model Ranked)')
        ax.set_ylabel('Number of Hits Found(Expected)')
        ax.legend(loc='lower right')
        ax.grid(True, linestyle=':', alpha=0.6)
    else:
        ax.set_title('Virtual Screening Enrichment')
        ax.text(0.5, 0.5, 'Not enough variance to calculate enrichment', ha='center', va='center')
        
    return ax

def plot_hit_rate_curve(y_true, y_pred, target_col, ax=None, top_quantile=0.2):
    """Plots a Hit Rate (Precision) curve using absolute numbers on the X-axis."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        
    df_eval = pd.DataFrame({'true': y_true, 'pred': y_pred})
    target_str = target_col.lower()
    lower_is_better = any(term in target_str for term in ['hmw', 'agg', 'viscosity', 'lmw', 'hcp', 'clearance', 'polyreactivity'])
    
    if lower_is_better:
        df_eval = df_eval.sort_values(by='pred', ascending=True).reset_index(drop=True)
        threshold_val = np.percentile(y_true, 100 * top_quantile)
        df_eval['is_hit'] = df_eval['true'] <= threshold_val
    else:
        df_eval = df_eval.sort_values(by='pred', ascending=False).reset_index(drop=True)
        threshold_val = np.percentile(y_true, 100 * (1 - top_quantile))
        df_eval['is_hit'] = df_eval['true'] >= threshold_val

    total_hits = df_eval['is_hit'].sum()
    total_samples = len(df_eval)
    
    if total_hits > 0:
        # Keep hit rate as a percentage, but change X-axis to absolute count
        df_eval['hit_rate'] = df_eval['is_hit'].cumsum() / (df_eval.index + 1)
        df_eval['samples_screened'] = df_eval.index + 1
        baseline_hit_rate = df_eval['is_hit'].mean()
        
        ax.plot([0, total_samples], [baseline_hit_rate, baseline_hit_rate], 'k--', label=f'Random Base Rate ({baseline_hit_rate*100:.0f}%)', alpha=0.6, linewidth=2)
        ax.plot(df_eval['samples_screened'], df_eval['hit_rate'], 'g-', label='Model Hit Rate', linewidth=2.5)
        ax.fill_between(df_eval['samples_screened'], baseline_hit_rate, df_eval['hit_rate'], 
                        where=(df_eval['hit_rate'] > baseline_hit_rate), color='green', alpha=0.1, interpolate=True)
        
        ax.set_title('Hit Rate: Are the Top Ranks Actually Hits?', fontsize=12)
        ax.set_xlabel('Total Antibodies Screened (Model Ranked)')
        ax.set_ylabel('Hit Rate (% of screened that are True Hits)')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.6)
    else:
        ax.set_title('Hit Rate (Precision)')
        ax.text(0.5, 0.5, 'Not enough variance to calculate hit rate', ha='center', va='center')
        
    return ax

def generate_shap_analysis(final_model, X, feature_names, model_name, target_col, output_dir, prefix="", transform_suffix="", feat_suffix=""):
    """Generates and saves a SHAP summary plot for the final trained model."""
    shap_filename = os.path.join(output_dir, f'shap_{prefix}{model_name}_{target_col}{transform_suffix}{feat_suffix}.png')
    
    if os.path.exists(shap_filename):
        print(f"\n⏭️ SHAP plot '{os.path.basename(shap_filename)}' already exists. Skipping recalculation to save time.")
        return
        
    try:
        import shap
        print("\nGenerating SHAP Summary Plot for deep feature insights...")
        
        # Extract the core scikit-learn pipeline/estimator from our custom meta-estimator
        base_pipeline = final_model.regressor_ if hasattr(final_model, 'regressor_') else final_model
        
        # Preprocess the data exactly as the model sees it
        if isinstance(base_pipeline, Pipeline):
            # Transform data through VarianceThreshold and Scaler, stopping before the final predictor
            X_shap = base_pipeline[:-1].transform(X)
            predictor = base_pipeline[-1]
            
            # 🌟 Filter feature names based on VarianceThreshold so they align perfectly
            support = base_pipeline.named_steps['vt'].get_support()
            active_feature_names = pd.Index(feature_names)[support]
        else:
            X_shap = X.values if isinstance(X, pd.DataFrame) else X
            predictor = base_pipeline
            active_feature_names = feature_names
            
        # Select the right mathematical explainer
        if model_name in ['RandomForest', 'XGBoost']:
            explainer = shap.TreeExplainer(predictor)
            shap_values = explainer.shap_values(X_shap)
        else:
            # Use KernelExplainer for PLS, SVR, and ElasticNet. 
            # We use shap.sample to summarize the background dataset down to 20 samples for speed.
            # background = shap.sample(X_shap, min(20, len(X_shap)))
            # explainer = shap.KernelExplainer(predictor.predict, background)
            # # 🌟 FIXED: Added l1_reg parameter and nsamples to fix Big Data memory crash
            # shap_values = explainer.shap_values(X_shap, nsamples=200, l1_reg="num_features(20)", silent=True)
            background = shap.kmeans(X_shap, min(20, len(X_shap)))
            explainer = shap.KernelExplainer(predictor.predict, background)
            shap_values = explainer.shap_values(X_shap, silent=True)
        # Plot and save
        plt.figure(figsize=(12, 8))
        
        # SHAP returns a list of arrays for multi-class/binary classification. We just need one.
        if isinstance(shap_values, list):
            shap_vals_to_plot = shap_values[1]
        else:
            shap_vals_to_plot = shap_values
            
        shap.summary_plot(shap_vals_to_plot, X_shap, feature_names=active_feature_names, show=False)
        plt.title(f"SHAP Value Impact: {model_name} on {target_col}", fontsize=14)
        plt.tight_layout()
        
        plt.savefig(shap_filename, dpi=300, facecolor='white', bbox_inches='tight')
        plt.close()
        print(f"✅ SHAP Summary Plot successfully saved to '{shap_filename}'!")
        
    except ImportError:
        print("\n⚠️ SHAP library not found. Run 'pip install shap' in your terminal to enable advanced feature analysis.")
    except Exception as e:
        print(f"\n⚠️ SHAP analysis failed: {e}")

def evaluate_subregion_combinations(df, seq_cols, generated_features, aaindex_desc, target_col='ProA_HMW_Excell', model_name='PLSRegression', force_retrain=False,
                                    classification_threshold=None, transform_type=None, weight_target_col=None, feature_tag="", antibody_type_col=None, test_mode="exhaustive",
                                    fixed_combos=None, prefix="", top_quantile=0.2):
    """Iterates through combinations of subregions OR tests specific fixed combinations."""
    target = target_col
    
    if target not in df.columns:
        print(f"Skipping ML model: Missing target column '{target}'.")
        return
    
    output_dir = model_name
    os.makedirs(output_dir, exist_ok=True)
    
    transform_suffix = f"_{transform_type}" if transform_type else ""
    feat_suffix = f"_{feature_tag}" if feature_tag else ""
    # 🌟 NEW: Apply prefix to isolate results!
    results_filename = os.path.join(output_dir, f'{prefix}combinations_results_{model_name}_{target}{transform_suffix}{feat_suffix}.xlsx')
    
    # 🌟 FIXED: Identify true "External" features (like Propermab CSV). 
    # We must explicitly exclude the 'Global_' prefix so the stitched chains don't leak into the subregion combos!
    global_features = [
        f for f in generated_features 
        if not any(f.startswith(sc + '_') for sc in seq_cols) 
        and not f.startswith('Global_')
    ]
    print(f"\n--- Model Setup: {model_name} for target {target} ---")
    
    best_combo_name = None
    best_features = None
    
    # --- SMART RESUME LOGIC ---
    if not force_retrain and os.path.exists(results_filename):
        print(f"✅ Found existing CV results at '{results_filename}'.")
        print("⏭️ SKIPPING exhaustive Cross-Validation search and proceeding directly to final model training!")
        
        # Load the best result directly from the Excel file
        results_df = pd.read_excel(results_filename)
        best_run = results_df.iloc[0]
        best_combo_name = best_run['Subregions_Used']
        
        # Reconstruct the optimal feature list based on the saved best combination
        combo = best_combo_name.split(' + ')
        best_features = [f for f in generated_features if any(f.startswith(sc + '_') for sc in combo)]
        
        # 🌟 FIXED: Re-append the global CQA and Propermab features during smart resume!
        best_features.extend(global_features)
        
    else:
        # --- FALLBACK: RUN EXHAUSTIVE CROSS VALIDATION LOOP ---
        print(f"Total samples in full dataset: {len(df)}")
        results = []
        # 🌟 NEW: Support Fixed Combo Testing
        if test_mode == "exhaustive":
            max_r = len(seq_cols)
            import math
            total_combos = sum(math.comb(len(seq_cols), i) for i in range(1, max_r + 1))
            print(f"Testing all exhaustive combinations up to {max_r} subregions at once.")
            
            combos_to_test = []
            for r in range(1, max_r + 1):
                combos_to_test.extend(list(itertools.combinations(seq_cols, r)))
        else:
            total_combos = len(fixed_combos)
            combos_to_test = fixed_combos
            print(f"Testing {total_combos} fixed global combinations.")
            
        current_combo = 0
        print(f"Evaluation Strategy: Parallel Repeated 5-Fold Cross Validation (3 repeats = 15 models per combo)\n")
        
        rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
        
        scoring = {
            'rmse': 'neg_root_mean_squared_error',
            'mae': 'neg_mean_absolute_error',
            'r2': 'r2',
            'spearman': make_scorer(custom_spearman)
        }
        
        for combo in combos_to_test:
            r = len(combo)
            current_combo += 1
            combo_name = ' + '.join(combo)
            
            # Fetch subregion features AND append the global/external features to every model!
            combo_features = [f for f in generated_features if any(f.startswith(sc + '_') for sc in combo)]
            combo_features.extend(global_features)
            
            cols_to_check = [target] + combo_features

            if weight_target_col and weight_target_col in df.columns:
                cols_to_check.append(weight_target_col)
                
            model_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols_to_check)
            
            print(f"[{current_combo}/{total_combos}] Evaluating: {combo_name}")
            
            if len(model_df) < 15:
                print(f"    -> Skipped (Only {len(model_df)} valid samples; need >=15 for 5-Fold CV)\n")
                continue
                
            # Assemble X with the hidden weights column at the end!
            X_cols = list(combo_features)
            if weight_target_col and weight_target_col in df.columns:
                X_cols.append(weight_target_col)
                
            X = model_df[X_cols].reset_index(drop=True)
            y = model_df[target].reset_index(drop=True)
            
            model = get_model(model_name, len(combo_features), transform_type, weight_target_col)
            
            cv_results = cross_vFValidate(
                model, X, y, cv=rkf, scoring=scoring, 
                n_jobs=-1, error_score=np.nan, return_estimator=True
            )
            
            valid_folds = ~np.isnan(cv_results['test_r2'])
            failed_folds = len(valid_folds) - valid_folds.sum()
            
            if failed_folds == len(valid_folds):
                print("    -> Skipped (All CV folds failed due to zero variance or math errors)")
                continue
                
            if failed_folds > 0:
                print(f"    -> Warning: {failed_folds} fold(s) failed math checks and were skipped.")
            
            mean_rmse = -np.mean(cv_results['test_rmse'][valid_folds])
            var_rmse = np.var(cv_results['test_rmse'][valid_folds])
            
            mean_mae = -np.mean(cv_results['test_mae'][valid_folds])
            var_mae = np.var(cv_results['test_mae'][valid_folds])
            
            mean_r2 = np.mean(cv_results['test_r2'][valid_folds])
            var_r2 = np.var(cv_results['test_r2'][valid_folds])
            
            mean_spearman = np.mean(cv_results['test_spearman'][valid_folds])
            var_spearman = np.var(cv_results['test_spearman'][valid_folds])
            
            fold_hyperparams = []
            for est in np.array(cv_results['estimator'])[valid_folds]:
                if hasattr(est, 'best_params_'):
                    # 🌟 NEW: Detect if this is an XGBoost model by checking the parameter keys
                    is_xgb = any('xgb' in key.lower() or 'max_depth' in key.lower() for key in est.best_params_.keys())
                    
                    if is_xgb:
                        # For XGBoost, save the ENTIRE parameter dictionary as a string 
                        fold_hyperparams.append(str(est.best_params_))
                    else:
                        # For PLS, SVR, and ElasticNet, keep the original single-value logic
                        for key, val in est.best_params_.items():
                            if key.endswith('pls__n_components') or key.endswith('enet__alpha') or key.endswith('svr__C'):
                                fold_hyperparams.append(val)
                                break
            
            if fold_hyperparams:
                # 🌟 FIXED: Gracefully handle multiple hyperparameters for XGBoost
                try:
                    # Works for SVR/PLS (single numeric parameters)
                    optimal_hyperparam = float(pd.Series(fold_hyperparams).mode()[0])
                except (ValueError, TypeError):
                    # Works for XGBoost (dictionary of multiple parameters)
                    # Converts dicts to strings to find the most common combination safely
                    optimal_hyperparam = str(pd.Series([str(p) for p in fold_hyperparams]).mode()[0])
            else:
                optimal_hyperparam = np.nan
            
            # 🌟 FEATURE COUNTING LOGIC (Must happen BEFORE the print statement)
            num_pm = sum(1 for f in combo_features if f.startswith('Propermab_'))
            num_cqa = sum(1 for f in combo_features if f.startswith('CQA_'))
            num_aac = sum(1 for f in combo_features if '_AAC_' in f)
            num_aaindex = sum(1 for f in combo_features if '_AAindex_' in f)
            num_esm = sum(1 for f in combo_features if '_ESM_' in f)
            
            num_other = len(combo_features) - (num_pm + num_cqa + num_aac + num_aaindex + num_esm)
            
            feat_info = f"{len(combo_features)} total ({num_esm} ESM, {num_pm} 3D-PM, {num_cqa} CQA, {num_aaindex} AAidx, {num_aac} AAC, {num_other} Other)"           
            print(f"    -> Mean Spearman: {mean_spearman:.3f} | Mean MAE: {mean_mae:.3f} | Valid Samples: {len(model_df)} | Features: {feat_info}")
        
            results.append({
                'Subregions_Used': combo_name,
                'Num_Subregions': r,
                'Optimal_Hyperparam': optimal_hyperparam,
                'Mean_Spearman': mean_spearman,
                'Var_Spearman': var_spearman,
                'Mean_MAE': mean_mae,
                'Var_MAE': var_mae,
                'Mean_RMSE': mean_rmse,
                'Var_RMSE': var_rmse,
                'Mean_R2': mean_r2,
                'Var_R2': var_r2,
                'Features': combo_features
            })

        if not results:
            print("Not enough data to train models.")
            return

        results_df = pd.DataFrame(results).sort_values(by='Mean_Spearman', ascending=False)
        
        print(f"\nSaving complete results for all {len(results_df)} combinations to '{results_filename}'...")
        csv_cols = [
            'Subregions_Used', 'Num_Subregions', 'Optimal_Hyperparam',
            'Mean_Spearman', 'Var_Spearman', 'Mean_MAE', 'Var_MAE', 
            'Mean_RMSE', 'Var_RMSE', 'Mean_R2', 'Var_R2'
        ]
        results_df[csv_cols].to_excel(results_filename, index=False)
        
        print("\n🏆 Top 5 Subregion Combinations by Mean CV Spearman Correlation:")
        display_cols = ['Subregions_Used', 'Mean_Spearman', 'Mean_MAE', 'Mean_RMSE', 'Mean_R2']
        print(results_df[display_cols].head(5).to_string(index=False))
        
        best_run = results_df.iloc[0]
        best_combo_name = best_run['Subregions_Used']
        best_features = best_run['Features']

    # --- DEEP DIVE: FINALIZE BEST MODEL (Runs instantly if CV was skipped) ---
    print(f"\n--- Deep Dive: Finalizing Best Model ({best_combo_name}) ---")
    print("Retraining on 100% of available data to extract definitive feature importances...")
    
    cols_to_check_final = [target] + best_features
    if weight_target_col and weight_target_col in df.columns:
        cols_to_check_final.append(weight_target_col)
    if antibody_type_col and antibody_type_col in df.columns:
        cols_to_check_final.append(antibody_type_col)
        
    final_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols_to_check_final)
    
    # Extract the shape labels for the plot (if they exist) before they are left out of X
    type_labels = final_df[antibody_type_col].values if (antibody_type_col and antibody_type_col in final_df.columns) else None
    
    # Assemble X_final with the hidden weights column at the end!
    X_cols_final = list(best_features)
    if weight_target_col and weight_target_col in df.columns:
        X_cols_final.append(weight_target_col)
        
    X_final = final_df[X_cols_final]
    y_final = final_df[target]
    
    final_model = get_model(model_name, len(best_features), transform_type, weight_target_col)
    
    try:
        final_model.fit(X_final, y_final)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to finalize best model on full dataset: {e}")
        return
        
    best_estimator = final_model.best_estimator_ if hasattr(final_model, 'best_estimator_') else final_model
    base_pipeline = best_estimator.regressor_ if hasattr(best_estimator, 'regressor_') else best_estimator
    
    # Now that the final model is trained on 100% of the data, we can extract the true global optimal lambda for the plot!
    optimal_lambda_final = None
    if transform_type == 'weighted-yeo-johnson' and hasattr(best_estimator, 'lmbda_'):
        optimal_lambda_final = best_estimator.lmbda_
        print(f"\nCustom Weighted Yeo-Johnson lambda calculated for final model: {optimal_lambda_final:.3f}")
        
    # 🌟 UPDATED: Pass feature_tag instead of esm_tag
    plot_target_distribution(y_final, target, transform_type, output_dir, optimal_lambda=optimal_lambda_final, feature_tag=feature_tag)

    if hasattr(final_model, 'best_params_'):
        print(f"\nOptimal Hyperparameters Discovered: {final_model.best_params_}")
        
    # Safely extract feature names by making sure we don't accidentally grab the hidden weight column!
    if model_name == 'RandomForest':
        imp_vals = base_pipeline.feature_importances_
        feature_names = best_features
    elif model_name == 'PLSRegression':
        imp_vals = np.abs(base_pipeline.named_steps['pls'].coef_).flatten()
        feature_names = pd.Index(best_features)[base_pipeline.named_steps['vt'].get_support()]
    elif model_name == 'ElasticNet':
        imp_vals = np.abs(base_pipeline.named_steps['enet'].coef_).flatten()
        feature_names = pd.Index(best_features)[base_pipeline.named_steps['vt'].get_support()]
    elif model_name == 'SVR':
        if base_pipeline.named_steps['svr'].kernel == 'linear':
            imp_vals = np.abs(base_pipeline.named_steps['svr'].coef_).flatten()
        else:
            print("\n  -> Note: Feature importances are not mathematically defined for SVR with non-linear (RBF) kernels.")
            imp_vals = np.zeros(base_pipeline.named_steps['vt'].get_support().sum())
        feature_names = pd.Index(best_features)[base_pipeline.named_steps['vt'].get_support()]
    else:
        imp_vals = np.zeros(len(best_features))
        feature_names = best_features
        
    importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': imp_vals
    }).sort_values(by='Importance', ascending=False)
    
    def get_description(feat_name):
        if '_AAindex_' in feat_name:
            code = feat_name.split('_AAindex_')[-1]
            return aaindex_desc.get(code, "Unknown AAindex Property")
        elif '_AAC_' in feat_name:
            aa = feat_name.split('_AAC_')[-1]
            return f"Absolute Count of Amino Acid {aa} (AAC)"
        elif '_ESM_' in feat_name:
            parts = feat_name.split('_ESM_')[-1].split('_')
            dim = parts[-1]
            size_tag = "_".join(parts[:-1]) if len(parts) > 1 else "Unknown_Size"
            return f"ESM-2 Contextual Embedding (Size: {size_tag}, Dimension {dim})"
        elif feat_name.startswith('CQA_'):
            return "Global Target-Specific CQA Feature"
        # elif feat_name.endswith('_Length'): return "Sequence Length"
        return ""
        
    importances['Description'] = importances['Feature'].apply(get_description)
    importances = importances[['Feature', 'Description', 'Importance']]
    
    print("Most Critical Features extracted from the final model:")
    print(importances.head(20).to_string(index=False))
    
    model_filename = os.path.join(output_dir, f'best_{prefix}{model_name}_{target}{transform_suffix}{feat_suffix}_model.joblib')
    print(f"\nSaving the final model and feature list to '{model_filename}'...")
    joblib.dump({
        'model': final_model,
        'features': list(best_features),
        'target': target
    }, model_filename)
    print("Model successfully saved!")
    # --- 🌟 CALL SHAP ANALYSIS HERE (Safely after saving model) ---
    # We pass only the feature columns, keeping weighting targets out of the SHAP explainer
    # generate_shap_analysis(
    #     final_model=final_model,
    #     X=final_df[list(best_features)], 
    #     feature_names=best_features,
    #     model_name=model_name,
    #     target_col=target,
    #     output_dir=output_dir,
    #     prefix=prefix,
    #     transform_suffix=transform_suffix,
    #     feat_suffix=feat_suffix
    # )
    print("------------------------------------------------------------------\n")
    
    best_params_dict = final_model.best_params_ if hasattr(final_model, 'best_params_') else None
    # plot_best_model_diagnostics(
    #     X_final, y_final, best_combo_name, model_name, target, output_dir, 
    #     best_estimator, best_params=best_params_dict, 
    #     threshold=classification_threshold,
    #     transform_type=transform_type,
    #     feature_tag=feature_tag,
    #     type_labels=type_labels,
    #     type_col_name=antibody_type_col,
    #     prefix=prefix,  # 🌟 FIXED: Pass the prefix down to the plotting function
    #     top_quantile=top_quantile  # 
    # )
    plot_best_model_diagnostics2(
        X_final, y_final, best_combo_name, model_name, target, output_dir, 
        best_estimator, best_params=best_params_dict, 
        threshold=classification_threshold,
        transform_type=transform_type,
        feature_tag=feature_tag,
        type_labels=type_labels,
        type_col_name=antibody_type_col,
        prefix=prefix,  # 🌟 FIXED: Pass the prefix down to the plotting function
        top_quantile=top_quantile  # 
    )

def plot_target_boxplots(filepath, target_cols, output_filename='target_distributions_combined.png'):
    """
    Loads dataset, extracts specified target columns, and generates a single 
    combined plot with side-by-side boxplots, overlaid data points, and 
    aligned skewness statistics for direct comparison.
    """
    print(f"Loading data from '{filepath}'...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: Could not find '{filepath}'. Please check the path.")
        return

    # Strip whitespace from column names to ensure perfect matching
    df.columns = df.columns.str.strip()
    
    # Filter targets to only those that actually exist in the CSV
    valid_targets = [col for col in target_cols if col in df.columns]
    
    if not valid_targets:
        print("Error: None of the specified target columns were found in the dataset.")
        print(f"Available columns: {list(df.columns)}")
        return
        
    print(f"Found {len(valid_targets)} valid targets to plot: {valid_targets}")

    # Subset the dataframe to only our targets, then 'melt' it.
    # Melting converts it from wide format (many columns) to long format (Target, Value)
    df_filtered = df[valid_targets]
    df_melted = df_filtered.melt(var_name='Target', value_name='Value').dropna()

    # Dynamically calculate plot width based on the number of targets
    plot_width = max(10, 3.5 * len(valid_targets))
    plt.figure(figsize=(plot_width, 8))
    
    ax = plt.gca()

    # 1. Plot the Boxplot (Shared X-axis)
    sns.boxplot(
        x='Target', 
        y='Value',
        data=df_melted, 
        ax=ax, 
        color='lightgray', 
        width=0.5, 
        boxprops=dict(alpha=0.6, edgecolor='black'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        medianprops=dict(color='red', linewidth=2)
    )
    
    # 2. Overlay the Stripplot (Shows every single data point)
    sns.stripplot(
        x='Target', 
        y='Value',
        data=df_melted, 
        ax=ax, 
        color='royalblue', 
        size=5, 
        alpha=0.7, 
        jitter=0.2, # Spreads points horizontally so they don't overlap
        edgecolor='black',
        linewidth=0.5
    )

    # Find the maximum Y value across all data to position our text boxes cleanly above the plot
    global_max_y = df_melted['Value'].max()
    global_min_y = df_melted['Value'].min()
    y_range = global_max_y - global_min_y
    
    # Add 25% padding to the top of the Y-axis to make room for the stat boxes
    ax.set_ylim(global_min_y - (y_range * 0.05), global_max_y + (y_range * 0.25))
    text_y_position = global_max_y + (y_range * 0.05)

    # Loop through each category to calculate and place its specific stats
    for i, target in enumerate(valid_targets):
        data = df_filtered[target].dropna()
        
        n_samples = len(data)
        mean_val = data.mean()
        median_val = data.median()
        skewness = data.skew()
        std_dev = data.std()

        stats_text = (
            f"N = {n_samples}\n"
            f"Mean = {mean_val:.2f}\n"
            f"Median = {median_val:.2f}\n"
            f"Std = {std_dev:.2f}\n"
            f"Skew = {skewness:.2f}"
        )
        
        # Place text box aligned exactly with the x-tick for this target
        props = dict(boxstyle='round,pad=0.4', facecolor='#f8f9fa', alpha=0.9, edgecolor='gray')
        ax.text(
            i, text_y_position, stats_text, 
            ha='center', va='bottom',
            fontsize=11,
            bbox=props
        )

    ax.set_title('Target Distributions Comparison', fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel('Percentage (%)' if any('%' in t or 'Monomer' in t or 'HMW' in t for t in valid_targets) else 'Value', fontsize=14)
    ax.set_xlabel('', fontsize=14) # Hide the generic 'Target' label
    
    # Improve x-tick readability
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12)

    plt.tight_layout()
    
    plt.savefig(output_filename, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"\n✅ Success! Combined target distributions saved to '{output_filename}'.")

def main():
    filepath = 'data/tubespin.csv'
    # Define the exact column names you want to visualize
    # targets_to_visualize = [
    #     'ProA_Monomer_ActiPro',
    #     'ProA_Monomer_Excell',
    #     'ProA_HMW_ActiPro',
    #     'ProA_HMW_Excell',
    #     'ELISA_Polyreactivity_Excell'
    # ]
    # calculate_and_plot_variance(filepath, 'subregion_sequence_variance.png')
    # plot_target_boxplots(filepath, targets_to_visualize)

    # 🎯 Define targets and their MANUALLY set thresholds for the classification metrics
    targets_to_test = {
        'ProA_HMW_Excell': 10.0,  # <-- Set your custom HMW threshold here (e.g., 20.0%)
        # 'ProA_HMW_ActiPro': 10.0,  # <-- Set your custom HMW threshold here
        # 'ProA_Monomer_ActiPro': 85.0,
        # 'ProA_Monomer_Excell': 85.0
        # 'ELISA_Polyreactivity_Excell':10.0
    }
    
    # filepath = 'data/inhouse_supp_CD3+CD20only_UPDATED.csv'
    # targets_to_test = {
    #     'New_Purity%': 80.0,
    #     # 'HMW':10.0
    # }
    
    models_to_test = ['SVR']#, 'PLSRegression']#['SVR']#['PLSRegression'] 
    
    force_retrain_cv_grids = False
    
    # 🌟 NEW: Set your desired transformation here! 
    # Options: None, 'log1p', 'box-cox', 'yeo-johnson', or 'weighted-yeo-johnson'
    transform_strategy = None#'weighted-yeo-johnson'
    
    # 🌟 NEW: If using 'weighted-yeo-johnson', specify the custom weight column here!
    weighting_column = None#'HCCF Titer (mg/L)'

    # 🌟 NEW: Set the ESM-2 model you want to use here!
    # Options: 'facebook/esm2_t6_8M_UR50D' (Small), 'facebook/esm2_t12_35M_UR50D' (Medium), 'facebook/esm2_t33_650M_UR50D' (Big)
    esm_model_selection = "facebook/esm2_t6_8M_UR50D"
    
    # 🌟 NEW: Set an optional column name here to visualize different antibody types with distinct markers!
    antibody_format_column = 'Type' # e.g., 'Antibody_Type', 'Format', 'Scaffold_Type'
    
    # 🌟 NEW: Toggle to load pre-extracted external features (like Propermab) from a CSV
    use_external_features = False
    
    # 🌟 NEW: Toggle to load target-specific CLQ features from .npy files
    use_cqa_features = False

    # Set the quantile for the Enrichment and Hit Rate plots
    my_top_quantile = 0.2

    DROP_MONOMER_OUTLIER = False 


    # Determine the ESM tag based on the selection to appropriately label files
    if "8M" in esm_model_selection: esm_tag = "ESM_Small_8M"
    elif "35M" in esm_model_selection: esm_tag = "ESM_Medium_35M"
    elif "150M" in esm_model_selection: esm_tag = "ESM_Large_150M"
    elif "650M" in esm_model_selection: esm_tag = "ESM_Big_650M"
    elif "3B" in esm_model_selection: esm_tag = "ESM_Massive_3B"
    else: esm_tag = "ESM_Custom"
        
    try:
        df = load_and_clean_data(filepath, remove_outlier=DROP_MONOMER_OUTLIER)

        determine_preferred_media(df)
        plot_media_comparison(df)
        
        # Extract the raw filename to use as the cache folder
        dataset_name = os.path.splitext(os.path.basename(filepath))[0]
        # --- NEW: Dynamically tag the cache based on our pipeline settings ---
        outlier_tag = "OutliersRemoved" if DROP_MONOMER_OUTLIER else "AllSamples"
        df_features, seq_cols, generated_features, aaindex_desc = extract_sequence_features(
            df, 
            dataset_name=dataset_name, 
            esm_model_name=esm_model_selection,
            cache_tag=outlier_tag

        )
        
        # --- 🌟 NEW: Load and append External Features (e.g., Pre-computed Propermab) ---
        ext_feat_tag = ""
        if use_external_features:
            # Looks for a file named "tubespin_external_features.csv" inside the specific dataset's cache folder
            ext_csv_path = os.path.join("feature_cache", dataset_name, f"{dataset_name}_propermab.csv")            
            if os.path.exists(ext_csv_path):
                print(f"Loading external features from '{ext_csv_path}'...")
                ext_df = pd.read_csv(ext_csv_path)
                
                # Append the new columns (ignoring any that already exist to prevent duplicates)
                new_feats = [c for c in ext_df.columns if c not in df_features.columns]
                df_features = pd.concat([df_features.reset_index(drop=True), ext_df[new_feats].reset_index(drop=True)], axis=1)
                
                # Register the new features so the ML model uses them
                generated_features.extend(new_feats)
                ext_feat_tag = "-ExtPM"
                print(f"✅ Successfully appended {len(new_feats)} external features!\n")
            else:
                print(f"⚠️ Warning: External feature file '{ext_csv_path}' not found. Training without them.\n")

        # --- 🌟 NEW: Construct the Smart Tag ---
        # This string (e.g. "_Feats-AAC-AAidx-ESM-8M-ExtPM") will be stamped on all saved files!
        base_feature_tag = f"Feats-AAC-AAidx-{esm_tag}{ext_feat_tag}"
        
        if seq_cols:
            for target_column, manual_threshold in targets_to_test.items():
                # --- 🌟 NEW: Load Target-Specific CLQ Features (.npy) ---
                df_features_run = df_features.copy()
                generated_features_run = list(generated_features)
                run_feature_tag = base_feature_tag
                
                if use_cqa_features:
                    # Deduce media type from target string
                    media_type = "ActiPro" if "ActiPro" in target_column else "Excell"
                    cqa_npy_path = os.path.join("feature_cache", dataset_name, f"{dataset_name}_CQA_{media_type}.npy")
                    
                    if os.path.exists(cqa_npy_path):
                        print(f"Loading media-specific CQA features for {media_type} from '{cqa_npy_path}'...")
                        cqa_data = np.load(cqa_npy_path)
                        
                        # Handle both 1D and 2D arrays gracefully
                        if cqa_data.ndim == 1: cqa_data = cqa_data.reshape(-1, 1)
                            
                        # Generate dynamic column names
                        cqa_cols = [f"CQA_{media_type}_{i}" for i in range(cqa_data.shape[1])]
                        cqa_df = pd.DataFrame(cqa_data, columns=cqa_cols)
                        
                        if len(cqa_df) != len(df_features_run):
                            print(f"⚠️ Warning: CQA array has {len(cqa_df)} rows but dataset has {len(df_features_run)}. Check for mismatch!")
                            
                        # Append the features dynamically to this specific run
                        df_features_run = pd.concat([df_features_run.reset_index(drop=True), cqa_df.reset_index(drop=True)], axis=1)
                        generated_features_run.extend(cqa_cols)
                        run_feature_tag += f"-CQA-{media_type}"
                        print(f"✅ Successfully appended {len(cqa_cols)} CQA features for {target_column}!\n")
                    else:
                        print(f"⚠️ Warning: CQA feature file '{cqa_npy_path}' not found. Training without them.\n")

                for model_name in models_to_test:
                    # print(f"\n==================================================================")
                    # print(f"🚀 STARTING RUN: Target = {target_column} | Model = {model_name} | Cutoff = {manual_threshold}% | Transform = {transform_strategy}")
                    # print(f"==================================================================")
                    
                    # 1. Exhaustive Subregion Combinations
                    evaluate_subregion_combinations(
                        df_features_run, 
                        seq_cols, 
                        generated_features_run, 
                        aaindex_desc, 
                        target_col=target_column, 
                        model_name=model_name,
                        force_retrain=force_retrain_cv_grids,
                        classification_threshold=manual_threshold,
                        transform_type=transform_strategy,
                        weight_target_col=weighting_column,
                        feature_tag=run_feature_tag,
                        antibody_type_col=antibody_format_column,
                        test_mode="exhaustive"
                    )
                    
                    # # 2. 🌟 NEW: Global Sequence Context (VH, VL, Fv)
                    # print(f"\n==================================================================")
                    # print(f"🚀 STARTING RUN: GLOBAL SEQUENCES (VH, VL, Fv) | Target = {target_column}")
                    # print(f"==================================================================")
                    # evaluate_subregion_combinations(
                    #     df_features_run, 
                    #     seq_cols, 
                    #     generated_features_run, 
                    #     aaindex_desc, 
                    #     target_col=target_column, 
                    #     model_name=model_name,
                    #     force_retrain=force_retrain_cv_grids,
                    #     classification_threshold=manual_threshold,
                    #     transform_type=transform_strategy,
                    #     weight_target_col=weighting_column,
                    #     feature_tag=run_feature_tag,
                    #     antibody_type_col=antibody_format_column,
                    #     test_mode="fixed",
                    #     fixed_combos=[('Global_VH',), ('Global_VL',), ('Global_Fv',)],
                    #     prefix="global_",
                    #     top_quantile=my_top_quantile
                    # )
            
    except FileNotFoundError:
        print(f"Error: Could not find '{filepath}'.")

if __name__ == "__main__":
    main()
