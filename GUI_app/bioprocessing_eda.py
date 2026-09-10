import os
import glob
import csv
import warnings
import ast 
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import urllib.request
import joblib
import math
import shutil
import hashlib

from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_predict, LeaveOneOut, cross_validate, LearningCurveDisplay, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score
from matplotlib.colors import ListedColormap 
from scipy.stats import spearmanr
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import PowerTransformer
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from scipy.optimize import minimize_scalar
from scipy.stats import yeojohnson

# --- Optional Dependencies Check ---
try:
    from antiberty import AntiBERTyRunner
    ANTIBERTY_AVAILABLE = True
except ImportError:
    ANTIBERTY_AVAILABLE = False

try:
    import ablang2
    ABLANG2_AVAILABLE = True
except ImportError:
    ABLANG2_AVAILABLE = False

# --- Georgiev Amino Acid Principal Components ---
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

def completely_silence_warnings(*args, **kwargs):
    """Overrides the default warning handler to suppress console noise during bulk feature extraction."""
    pass
warnings.warn = completely_silence_warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings('ignore')

ESM_AVAILABLE = True
PROPERMAB_AVAILABLE = True
sns.set_theme(style="whitegrid")

def custom_spearman(y_true, y_pred):
    """
    Calculates the Spearman rank-order correlation coefficient safely.
    
    Args:
        y_true (array-like): Ground truth target values.
        y_pred (array-like): Estimated target values.
        
    Returns:
        float: Correlation coefficient (0 if NaN due to zero variance).
    """
    sc, _ = spearmanr(y_true, y_pred)
    return sc if not np.isnan(sc) else 0

def calculate_weighted_yeojohnson_lambda(y, weights):
    """
    Finds the optimal Yeo-Johnson transformation lambda, accounting for sample weights.
    
    Args:
        y (array-like): Target values to transform.
        weights (array-like): Sample weights for the optimization function.
        
    Returns:
        float: Optimized lambda value bounded between -3.0 and 3.0.
    """
    y_flat = np.asarray(y).flatten()
    w = np.asarray(weights).flatten()
    w = (w / np.sum(w)) * len(w)
    
    def weighted_nll(lmbda):
        y_trans = yeojohnson(y_flat, lmbda)
        w_mean = np.average(y_trans, weights=w)
        w_var = np.average((y_trans - w_mean)**2, weights=w)
        if w_var <= 0: return np.inf
        jacobian = np.sum(w * np.sign(y_flat) * np.log1p(np.abs(y_flat)))
        nll = (len(w) / 2.0) * np.log(w_var) - (lmbda - 1) * jacobian
        return nll
        
    res = minimize_scalar(weighted_nll, bounds=(-3.0, 3.0), method='bounded')
    return res.x

class CustomYeoJohnsonTransformer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer for applying a predefined Yeo-Johnson lambda."""
    def __init__(self, lmbda=1.0):
        self.lmbda = lmbda
        
    def fit(self, X, y=None): 
        return self
        
    def transform(self, X):
        X_flat = np.asarray(X).flatten()
        return yeojohnson(X_flat, self.lmbda).reshape(-1, 1)
        
    def inverse_transform(self, X_trans):
        X_trans_flat = np.asarray(X_trans).flatten()
        X_inv = np.zeros_like(X_trans_flat)
        pos, neg = X_trans_flat >= 0, ~(X_trans_flat >= 0)
        
        if abs(self.lmbda) < 1e-10: X_inv[pos] = np.exp(X_trans_flat[pos]) - 1
        else: X_inv[pos] = np.maximum(X_trans_flat[pos] * self.lmbda + 1, 0) ** (1 / self.lmbda) - 1
            
        if abs(self.lmbda - 2.0) < 1e-10: X_inv[neg] = 1 - np.exp(-X_trans_flat[neg])
        else: X_inv[neg] = 1 - np.maximum(1 - (2 - self.lmbda) * X_trans_flat[neg], 0) ** (1 / (2 - self.lmbda))
        return X_inv.reshape(-1, 1)

class SelfContainedTargetTransformRegressor(BaseEstimator, RegressorMixin):
    """
    Wrapper class that applies a target transformation (e.g., log1p, box-cox) strictly within 
    cross-validation folds to prevent data leakage, automatically handling the inverse transform on prediction.
    
    Args:
        regressor (estimator): Scikit-learn regression model.
        transform_type (str): Type of transformation ('log1p', 'box-cox', 'yeo-johnson').
        weight_col (str): Column name representing sample weights.
    """
    def __init__(self, regressor, transform_type=None, weight_col=None):
        self.regressor = regressor
        self.transform_type = transform_type
        self.weight_col = weight_col
        
    def fit(self, X, y):
        self.regressor_ = clone(self.regressor)
        
        # Separate sample weights from the feature matrix if provided
        if self.weight_col:
            if isinstance(X, pd.DataFrame):
                weights = X[self.weight_col].values
                X_clean = X.drop(columns=[self.weight_col])
            else:
                weights = X[:, -1]
                X_clean = X[:, :-1]
        else:
            X_clean = X.copy() if isinstance(X, pd.DataFrame) else np.copy(X)
            weights = None
            
        if self.transform_type == 'log1p': 
            y_trans = np.log1p(y)
        elif self.transform_type in ['box-cox', 'yeo-johnson']:
            self.pt_ = PowerTransformer(method=self.transform_type)
            y_trans = self.pt_.fit_transform(np.asarray(y).reshape(-1, 1)).flatten()
        elif self.transform_type == 'weighted-yeo-johnson':
            if weights is None: weights = np.ones_like(y)
            self.lmbda_ = calculate_weighted_yeojohnson_lambda(y, weights)
            self.pt_ = CustomYeoJohnsonTransformer(lmbda=self.lmbda_)
            y_trans = self.pt_.transform(np.asarray(y).reshape(-1, 1)).flatten()
        else:
            y_trans = y
            
        self.regressor_.fit(X_clean, y_trans)
        return self
        
    def predict(self, X):
        # Remove sample weights before predicting
        if self.weight_col:
            if isinstance(X, pd.DataFrame): X_clean = X.drop(columns=[self.weight_col])
            else: X_clean = X[:, :-1]
        else:
            X_clean = X
            
        y_pred_trans = self.regressor_.predict(X_clean)
        
        if self.transform_type == 'log1p': return np.expm1(y_pred_trans)
        elif self.transform_type in ['box-cox', 'yeo-johnson']:
            return self.pt_.inverse_transform(y_pred_trans.reshape(-1, 1)).flatten()
        elif self.transform_type == 'weighted-yeo-johnson':
            return self.pt_.inverse_transform(y_pred_trans.reshape(-1, 1)).flatten()
        else:
            return np.asarray(y_pred_trans).flatten()

def load_and_clean_data(filepath, remove_outlier=False):
    """
    Loads sequence data from a CSV, trims column names, and optionally removes known geometric outliers.
    
    Args:
        filepath (str): Path to the input dataset.
        remove_outlier (bool): If True, drops hardcoded outlier samples.
        
    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    
    if remove_outlier and 'Samples' in df.columns:
        initial_len = len(df)
        outliers_to_remove = ['H57L46', 'H57L38']
        df = df[~df['Samples'].str.strip().isin(outliers_to_remove)].reset_index(drop=True)
        rows_dropped = initial_len - len(df)
        if rows_dropped > 0: print(f"  -> Outliers removed. ({rows_dropped} rows dropped)")
    return df

def filter_redundant_sequences(df, seq_cols):
    """
    Scans sequence columns to identify and remove nested sub-sequences (e.g., preventing CDR1 
    from being treated as an independent region if it is already fully contained within another column).
    
    Args:
        df (pd.DataFrame): The sequence dataset.
        seq_cols (list): List of column names containing sequence strings.
        
    Returns:
        tuple: (Filtered list of sequence columns, List of identified parent columns).
    """
    parents = set()
    for col_a in seq_cols:
        for col_b in seq_cols:
            if col_a == col_b: continue
            val_a, val_b = df[col_a].dropna(), df[col_b].dropna()
            if val_a.empty or val_b.empty: continue
            str_a = ''.join(str(val_a.iloc[0]).split()).replace(',', '').upper()
            str_b = ''.join(str(val_b.iloc[0]).split()).replace(',', '').upper()
            if len(str_a) > 5 and str_a in str_b and len(str_a) < len(str_b):
                parents.add(col_b)
    return [c for c in seq_cols if c not in parents], list(parents)

def load_aaindex():
    """
    Downloads and parses the AAindex1 database, mapping amino acid properties to numerical values.
    
    Returns:
        tuple: (Dictionary mapping AAindex codes to 20-length float arrays, Dictionary of text descriptions).
    """
    filepath, aaindex_dict, aaindex_desc = 'aaindex1.txt', {}, {}
    if not os.path.exists(filepath):
        try: urllib.request.urlretrieve("https://www.genome.jp/ftp/db/community/aaindex/aaindex1", filepath)
        except Exception: return aaindex_dict, aaindex_desc
            
    try:
        with open(filepath, 'r') as f: lines = f.readlines()
        current_id, current_desc = None, None
        for i in range(len(lines)):
            if lines[i].startswith('H '): current_id = lines[i].split()[1]
            elif lines[i].startswith('D '): current_desc = lines[i][2:].strip()
            elif lines[i].startswith('I '):
                vals1 = [float(x) if x != 'NA' else np.nan for x in lines[i+1].strip().split()]
                vals2 = [float(x) if x != 'NA' else np.nan for x in lines[i+2].strip().split()]
                if len(vals1) == 10 and len(vals2) == 10:
                    aa_keys = list('ARNDCQEGHILKMFPSTWYV')
                    aaindex_dict[current_id] = dict(zip(aa_keys, vals1 + vals2))
                    aaindex_desc[current_id] = current_desc
    except Exception: pass
    return aaindex_dict, aaindex_desc

def compute_aac_features(sequences, prefix="", target_indices=None):
    """
    Calculates the Amino Acid Composition (counts of the 20 standard residues).
    
    Args:
        sequences (list): List of amino acid sequences.
        prefix (str): Prefix to append to the generated feature columns.
        target_indices (list): Optional 0-indexed positions to restrict extraction to.
        
    Returns:
        dict: Dictionary of generated AAC features.
    """
    standard_aas = list('ARNDCQEGHILKMFPSTWYV')
    aac_dict = {}
    for aa in standard_aas:
        counts = []
        for seq in sequences:
            if seq == 'NAN' or not seq:
                counts.append(0)
                continue
            
            # Extract sequence strictly at user-defined spatial indices
            if target_indices is not None:
                valid_aas = [seq[i] for i in target_indices if i < len(seq)]
                counts.append(valid_aas.count(aa))
            else:
                counts.append(seq.count(aa))
        aac_dict[f'{prefix}AAC_{aa}'] = np.array(counts)
    return aac_dict

def compute_aaindex_features(sequences, aaindex_db, prefix="", target_indices=None):
    """
    Maps physical properties from the AAindex database to sequences, averaging the values 
    either globally or across a targeted subset of residues.
    
    Args:
        sequences (list): List of amino acid sequences.
        aaindex_db (dict): Parsed AAindex dictionary.
        prefix (str): Prefix to append to the generated feature columns.
        target_indices (list): Optional 0-indexed positions to restrict extraction to.
        
    Returns:
        dict: Dictionary of generated AAindex features.
    """
    aaindex_dict = {}
    if aaindex_db:
        for code, prop_map in aaindex_db.items():
            vals_array = []
            for seq in sequences:
                if seq == 'NAN' or not seq: 
                    vals_array.append(0)
                    continue
                
                if target_indices is not None:
                    valid_aas = [seq[i] for i in target_indices if i < len(seq)]
                else:
                    valid_aas = list(seq)
                    
                vals = [prop_map.get(aa) for aa in valid_aas if prop_map.get(aa) is not None and not np.isnan(prop_map.get(aa))]
                vals_array.append(sum(vals)/len(vals) if vals else 0)
            aaindex_dict[f'{prefix}AAindex_{code}'] = np.array(vals_array)
    return aaindex_dict

def compute_georgiev_features(sequences, georgiev_dict, prefix="", target_indices=None):
    """
    Calculates the mean of the 19-dimensional Georgiev Principal Components for the input sequence.
    
    Args:
        sequences (list): List of amino acid sequences.
        georgiev_dict (dict): Dictionary of Georgiev PCA values.
        prefix (str): Prefix to append to the generated feature columns.
        target_indices (list): Optional 0-indexed positions to restrict extraction to.
        
    Returns:
        dict: Dictionary of generated Georgiev PC features.
    """
    geo_features = []
    for seq in sequences:
        if seq == 'NAN' or len(seq) == 0: 
            geo_features.append([0.0] * 19)
        else:
            if target_indices is not None:
                valid_aas = [seq[i] for i in target_indices if i < len(seq)]
            else:
                valid_aas = list(seq)
                
            seq_geo = [georgiev_dict[aa] for aa in valid_aas if aa in georgiev_dict]
            geo_features.append(np.mean(seq_geo, axis=0).tolist() if seq_geo else [0.0] * 19)
            
    geo_features = np.array(geo_features)
    return {f"{prefix}Georgiev_PC{i+1}": geo_features[:, i] for i in range(19)}

def compute_esm_embeddings(sequences, esm_model_name, prefix="", esm_tag="", target_indices=None):
    """
    Generates representations using Evolutionary Scale Modeling (ESM-2).
    Extracts the mean-pooled final hidden state, respecting targeted indices if supplied.
    
    Args:
        sequences (list): List of amino acid sequences.
        esm_model_name (str): HuggingFace model path for ESM.
        prefix (str): Prefix to append to the generated feature columns.
        esm_tag (str): Size descriptor tag (e.g., 'ESM_Big_650M').
        target_indices (list): Optional 0-indexed positions to restrict extraction to.
        
    Returns:
        tuple: (Dictionary of flattened embeddings, Raw 2D numpy matrix for SVD compression).
    """
    import torch
    from transformers import EsmModel, EsmTokenizer, logging
    logging.set_verbosity_error()
    
    tokenizer = EsmTokenizer.from_pretrained(esm_model_name)
    model = EsmModel.from_pretrained(esm_model_name)
    model.eval()
    
    embeddings = []
    for seq in sequences:
        if seq == 'NAN' or len(seq) < 2: 
            embeddings.append(np.zeros(model.config.hidden_size))
        else:
            inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
            with torch.no_grad():
                hidden = model(**inputs).last_hidden_state[0]
                seq_hidden = hidden[1:-1] if hidden.shape[0] > 2 else hidden
                
                # Targeted interface pooling
                if target_indices is not None:
                    valid_indices = [i for i in target_indices if i < seq_hidden.shape[0]]
                    if valid_indices:
                        pooled = seq_hidden[valid_indices].mean(dim=0).cpu().numpy()
                    else:
                        pooled = np.zeros(model.config.hidden_size)
                # Global pooling
                else:
                    pooled = seq_hidden.mean(dim=0).cpu().numpy()
                    
                embeddings.append(pooled)
                
    embeddings = np.array(embeddings)
    return {f'{prefix}{esm_tag}_{i}': embeddings[:, i] for i in range(embeddings.shape[1])}, embeddings

def compress_embeddings_svd(embeddings_matrix, prefix="", esm_tag="", svd_model=None):
    """
    Compresses high-dimensional neural embeddings down to 50 dimensions using Truncated SVD.
    
    Args:
        embeddings_matrix (np.ndarray): The raw 2D array of embeddings.
        prefix (str): Prefix to append to the generated feature columns.
        esm_tag (str): Size descriptor tag.
        svd_model (object): A pre-fitted TruncatedSVD model. If provided, strictly projects data 
                            (inference mode) to prevent data leakage. If None, fits a new model.
                            
    Returns:
        tuple: (Dictionary of SVD50 features, Fitted SVD model object).
    """
    from sklearn.decomposition import TruncatedSVD
    
    # Inference mode: Project into the existing mathematical space
    if svd_model is not None:
        embeddings_svd = svd_model.transform(embeddings_matrix)
        return {f'{prefix}{esm_tag}_SVD50_{i}': embeddings_svd[:, i] for i in range(embeddings_svd.shape[1])}, svd_model
        
    # Training mode: Fit a brand new SVD model
    n_comps = min(50, embeddings_matrix.shape[0] - 1, embeddings_matrix.shape[1] - 1)
    if n_comps > 0:
        svd = TruncatedSVD(n_components=n_comps, random_state=42)
        embeddings_svd = svd.fit_transform(embeddings_matrix)
        return {f'{prefix}{esm_tag}_SVD50_{i}': embeddings_svd[:, i] for i in range(embeddings_svd.shape[1])}, svd
        
    return None, None

def compute_antiberty_embeddings(sequences, prefix=""):
    """
    Generates 512-dimensional representations using the antibody-specific AntiBERTy language model.
    
    Args:
        sequences (list): List of amino acid sequences.
        prefix (str): Prefix to append to the generated feature columns.
        
    Returns:
        dict: Dictionary of flattened AntiBERTy features.
    """
    import torch
    from antiberty import AntiBERTyRunner
    antiberty = AntiBERTyRunner()
    antiberty.model.eval()
    
    antiberty_embeddings = []
    for seq in sequences:
        if seq == 'NAN' or len(seq) < 2:
            antiberty_embeddings.append(np.zeros(512))
        else:
            with torch.no_grad():
                emb = antiberty.embed([seq])[0]
                if emb.shape[0] > 2: emb_mean = emb[1:-1].mean(dim=0).cpu().numpy()
                else: emb_mean = emb.mean(dim=0).cpu().numpy()
                antiberty_embeddings.append(emb_mean)
                
    antiberty_embeddings = np.array(antiberty_embeddings)
    return {f'{prefix}AntiBERTy_{i}': antiberty_embeddings[:, i] for i in range(antiberty_embeddings.shape[1])}

def compute_ablang2_paired_embeddings(heavy_seqs, light_seqs, prefix="Paired_CD3_VH_VL_AbLang2_"):
    """
    Generates joint representations using AbLang2, explicitly capturing co-evolutionary dependencies
    between the provided Heavy and Light chains.
    
    Args:
        heavy_seqs (list): List of VH sequences.
        light_seqs (list): List of VL sequences.
        prefix (str): Prefix to append to the generated feature columns.
        
    Returns:
        dict: Dictionary of flattened AbLang2 features.
    """
    import ablang2
    ablang = ablang2.pretrained(model_to_use="ablang2-paired", random_init=False)
    
    ablang_embeddings = []
    emb_dim = None
    
    for h_seq, l_seq in zip(heavy_seqs, light_seqs):
        if h_seq in ['NAN', 'NONE', ''] or l_seq in ['NAN', 'NONE', ''] or len(h_seq) < 2 or len(l_seq) < 2:
            ablang_embeddings.append(None) 
        else:
            res = ablang([[h_seq, l_seq]], mode='seqcoding')
            emb = res[0]
            if hasattr(emb, 'cpu'): emb = emb.cpu()
            if hasattr(emb, 'numpy'): emb = emb.numpy()
            ablang_embeddings.append(emb)
            if emb_dim is None: emb_dim = emb.shape[0]
            
    if emb_dim is None: emb_dim = 480 
    
    final_embeddings = [emb if emb is not None else np.zeros(emb_dim) for emb in ablang_embeddings]
    final_embeddings = np.array(final_embeddings)
    
    return {f'{prefix}{i}': final_embeddings[:, i] for i in range(final_embeddings.shape[1])}

def compute_propermab_features(heavy_seqs, light_seqs, prefix="Propermab_CD3"):
    """
    Models the 3D Fv complex using ABodyBuilder2 and calculates biophysical properties 
    (e.g., surface charge, patch hydrophobicity) of the folded structure via Propermab.
    
    Args:
        heavy_seqs (list): List of VH sequences.
        light_seqs (list): List of VL sequences.
        prefix (str): Prefix to append to the generated feature columns.
        
    Returns:
        dict: Dictionary of calculated 3D structural physics features.
    """
    import subprocess
    import sys
    import json
    from ImmuneBuilder import ABodyBuilder2
    
    propermab_features_list = []
    safe_tmp_dir = os.path.join(os.getcwd(), "propermab_pipeline_tmp")
    os.makedirs(safe_tmp_dir, exist_ok=True)
    raw_pdb_dir = os.path.join(safe_tmp_dir, "raw_pdbs").replace('\\', '/')
    cleaned_pdb_dir = os.path.join(safe_tmp_dir, "cleaned_pdbs").replace('\\', '/')
    os.makedirs(raw_pdb_dir, exist_ok=True)
    os.makedirs(cleaned_pdb_dir, exist_ok=True)
    
    worker_code = f"""
import os
import sys
import json
import math
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.simplefilter('ignore', PDBConstructionWarning)
os.environ['OPENMM_DEFAULT_PLATFORM'] = 'CPU'
os.environ['OPENMM_CPU_THREADS'] = '1'
try:
    import propermab
    from propermab import defaults
    if os.path.exists('default_config.json'): defaults.system_config.update_from_json('default_config.json')
    try: from propermab.preprocess_structures import preprocess_directory
    except ImportError:
        try: from propermab.scripts.preprocess_structures import preprocess_directory
        except ImportError: from preprocess_structures import preprocess_directory
            
    preprocess_directory(input_dir='{raw_pdb_dir}', output_dir='{cleaned_pdb_dir}', heavy_chain_id="H", light_chain_id="L", ph=7.4, remove_water=True)
    
    pdb_name = sys.argv[1]
    cleaned_pdb_path = os.path.join('{cleaned_pdb_dir}', pdb_name)
    try: from propermab.features import feature_utils_mod as feature_utils
    except ImportError: from propermab.features import feature_utils
        
    mol_features = feature_utils.calculate_features_from_pdb(cleaned_pdb_path)
    clean_dict = {{}}
    for k, v in mol_features.items():
        if isinstance(v, float) and math.isnan(v): clean_dict[k] = None
        elif hasattr(v, 'item'): clean_dict[k] = v.item()
        else: clean_dict[k] = v
            
    print("---JSON_PAYLOAD_START---")
    print(json.dumps(clean_dict))
    print("---JSON_PAYLOAD_END---")
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
    worker_path = os.path.join(safe_tmp_dir, "pipeline_worker.py")
    with open(worker_path, "w") as f: f.write(worker_code)
        
    predictor = ABodyBuilder2()
    
    for i, (h_seq, l_seq) in enumerate(zip(heavy_seqs, light_seqs)):
        if h_seq in ['NAN', 'NONE', ''] or l_seq in ['NAN', 'NONE', ''] or len(h_seq) < 2 or len(l_seq) < 2:
            print(f"      [{i+1}/{len(heavy_seqs)}] Skipping invalid sequences...")
            propermab_features_list.append(None)
            continue
            
        print(f"      [{i+1}/{len(heavy_seqs)}] Predicting and featurizing...")
        pdb_name = f"seq_{i}.pdb"
        raw_pdb_path = os.path.join(raw_pdb_dir, pdb_name)
        
        try:
            struct = predictor.predict({'H': h_seq, 'L': l_seq})
            struct.save(raw_pdb_path)
            result = subprocess.run([sys.executable, worker_path, pdb_name], capture_output=True, text=True)
            
            if result.returncode == 0 and "---JSON_PAYLOAD_START---" in result.stdout:
                json_str = result.stdout.split("---JSON_PAYLOAD_START---")[1].split("---JSON_PAYLOAD_END---")[0].strip()
                propermab_features_list.append(json.loads(json_str))
            else:
                print(f"      -> ⚠️ Error extracting features: \n{result.stderr}")
                propermab_features_list.append(None)
                
            if os.path.exists(raw_pdb_path): os.remove(raw_pdb_path)
            cleaned_out = os.path.join(cleaned_pdb_dir, pdb_name)
            if os.path.exists(cleaned_out): os.remove(cleaned_out)
        except Exception as e:
            print(f"      -> ⚠️ Exception on sequence {i}: {e}")
            propermab_features_list.append(None)

    if os.path.exists(safe_tmp_dir): shutil.rmtree(safe_tmp_dir)
    
    all_keys = set()
    for feat_dict in propermab_features_list:
        if feat_dict: all_keys.update(feat_dict.keys())
    all_keys.discard('antibody_format') 
    
    propermab_dict = {}
    for k in all_keys:
        arr = []
        for feat_dict in propermab_features_list:
            if feat_dict and k in feat_dict and feat_dict[k] is not None: arr.append(feat_dict[k])
            else: arr.append(0.0) 
        propermab_dict[f"{prefix}{k}"] = np.array(arr)
        
    for trash_file in glob.glob("*_apbs.csv") + glob.glob("*_vertices.csv") + glob.glob("io.mc"):
        try: os.remove(trash_file)
        except OSError: pass

    return propermab_dict

def extract_sequence_features(df, is_inference=False, dataset_name="default_dataset", esm_model_names=["facebook/esm2_t6_8M_UR50D"], cache_tag="", extract_subregions=False, feature_types=None, svd_models_dict=None, targeted_pooling_dict=None):
    """
    Main orchestrator for sequence-level and structural feature generation.
    It builds functional domains (VH, VL, scFv), manages feature caches, and handles 
    the branching logic for Global vs Targeted feature extraction runs.
    
    Args:
        df (pd.DataFrame): Base dataset containing sequences.
        is_inference (bool): If True, bypasses cache writing and skips fitting new SVD models, strictly applying pre-fitted tools.
        dataset_name (str): Identifier used to route outputs to correct cache folders.
        esm_model_names (list): List of HF model strings for ESM processing.
        cache_tag (str): Appended to cache folders to separate outlier-dropped from raw dataset caches.
        extract_subregions (bool): If True, granularly breaks down sequences into framework/CDR subregions.
        feature_types (list): Overrides the default feature families if a smaller subset is desired.
        svd_models_dict (dict): Dictionary of pre-fitted SVD models (used exclusively when is_inference=True).
        targeted_pooling_dict (dict): Maps regions to semantic tags and targeted 0-indexed positions for extraction.
        
    Returns:
        tuple: (Augmented DataFrame, List of processed sequence columns, List of generated feature names, AAindex descriptors, SVD models dict)
    """
    if svd_models_dict is None: 
        svd_models_dict = {}

    ALL_FEATURE_FAMILIES = ['AAC', 'AAindex', 'Georgiev', 'ESM', 'AntiBERTy', 'AbLang2_Paired', 'Propermab']
    
    if feature_types is None:
        active_features = [f.lower() for f in ALL_FEATURE_FAMILIES]
    else:
        active_features = [str(f).strip().lower() for f in feature_types]
        
    print(f"\n🎯 Active Feature Pipelines: {[f for f in ALL_FEATURE_FAMILIES if f.lower() in active_features]}")

    if isinstance(esm_model_names, str):
        esm_model_names = [esm_model_names]
        
    df_feat = df.copy()
    standard_aas = list('ARNDCQEGHILKMFPSTWYV')
    
    initial_seq_cols = []
    for col in df_feat.columns:
        valid_data = df_feat[col].dropna()
        if not valid_data.empty:
            clean_sample = ''.join(str(valid_data.iloc[0]).split()).replace(',', '').upper()
            if len(clean_sample) > 3 and sum(c in standard_aas for c in clean_sample) / len(clean_sample) > 0.8:
                initial_seq_cols.append(col)

    seq_cols = initial_seq_cols

    if not extract_subregions:
        print("\n🛑 [extract_subregions=False]: Skipping subregion feature extraction.")
        seq_cols = [] 

    if extract_subregions and not is_inference:
        print(f"\n==================================================================")
        print(f"🧬 EVALUATING SUBREGION DIVERSITY & SUITABILITY")
        print(f"==================================================================")
        valid_seq_cols = []
        for col in seq_cols:
            valid_seqs = df_feat[col].dropna().astype(str).str.replace(r'\s+|,', '', regex=True).str.upper()
            valid_seqs = valid_seqs[~valid_seqs.isin(['NAN', 'NONE', ''])]
            
            if valid_seqs.empty: continue
            if len(valid_seqs.iloc[0]) <= 3: continue
                
            num_unique = valid_seqs.nunique()
            if num_unique <= 1: continue
            else: valid_seq_cols.append(col)
        
        seq_cols, redundant_parents = filter_redundant_sequences(df_feat, valid_seq_cols)
        
        # Explicitly remove known metadata columns that might be incorrectly parsed as sequences
        for override in ['G4S Linker1_HCK', 'Media_Type', 'HCH', 'Method', 'Manual_Split_Group']:
            if override in seq_cols: seq_cols.remove(override)
            
        print(f"\nProceeding with {len(seq_cols)} granular building blocks: {seq_cols}\n")
    
    # Stitch global chains from granular subregions
    cd3_vh_ordered_cols = ['seq_CD3_frh1', 'seq_CD3_cdrh1', 'seq_CD3_frh2', 'seq_CD3_cdrh2', 'seq_CD3_frh3', 'seq_CD3_cdrh3', 'seq_CD3_frh4'] 
    cd3_vl_ordered_cols = ['seq_CD3_frl1', 'seq_CD3_cdrl1', 'seq_CD3_frl2', 'seq_CD3_cdrl2', 'seq_CD3_frl3', 'seq_CD3_cdrl3', 'seq_CD3_frl4']

    def build_full_seq(row, cols):
        parts = []
        for c in cols:
            if c in df_feat.columns and pd.notna(row[c]):
                val = str(row[c]).strip().upper()
                if val not in ['NAN', 'NONE', '']:
                    parts.append(val)
        return "".join(parts) if parts else 'NAN'

    df_feat['CD3_VH'] = df_feat.apply(lambda r: build_full_seq(r, cd3_vh_ordered_cols), axis=1)
    df_feat['CD3_VL'] = df_feat.apply(lambda r: build_full_seq(r, cd3_vl_ordered_cols), axis=1)
    
    def build_fv_with_linker(r):
        if r['CD3_VH'] == 'NAN' or r['CD3_VL'] == 'NAN': return 'NAN'
        linker = ''
        if 'G4S Linker2_HCK' in df_feat.columns and pd.notna(r['G4S Linker2_HCK']):
            val = str(r['G4S Linker2_HCK']).strip().upper()
            if val not in ['NAN', 'NONE']: linker = val
        return r['CD3_VH'] + linker + r['CD3_VL']
        
    df_feat['scFv'] = df_feat.apply(build_fv_with_linker, axis=1)
    
    all_seq_cols_to_process = seq_cols + ['CD3_VH', 'CD3_VL', 'scFv'] if extract_subregions else ['CD3_VH', 'CD3_VL', 'scFv']

    aaindex_db, aaindex_desc = None, None
    if 'aaindex' in active_features:
        aaindex_db, aaindex_desc = load_aaindex()
    
    generated_features = []
    new_columns = {}
    expected_dataset_len = len(df_feat)
    
    cache_dir = os.path.join("feature_cache", f"{dataset_name}_{expected_dataset_len}samples" + (f"_{cache_tag}" if cache_tag else ""))
    os.makedirs(cache_dir, exist_ok=True)
    
    def _load_valid_cache(cache_path, expected_len):
        if os.path.exists(cache_path):
            try:
                with np.load(cache_path) as cached_data:
                    if len(cached_data.files) > 0 and len(cached_data[cached_data.files[0]]) == expected_len:
                        return {k: cached_data[k] for k in cached_data.files}
            except Exception: pass
        return None
        
    print("\n==================================================================")
    print("🧬 STARTING FEATURE EXTRACTION & LOADING")
    print("==================================================================")

    for col in all_seq_cols_to_process:
        print(f"\n⚙️  Processing region: {col}")
        seqs = df_feat[col].astype(str).str.replace(r'\s+|,', '', regex=True).str.upper()

        # Determine extraction passes: Pass 1 is Global, Pass 2 is Targeted (if defined).
        extraction_passes = [(False, "", None)]
        if targeted_pooling_dict and col in targeted_pooling_dict:
            # Unpack the tuple provided by the user
            semantic_tag, target_indices = targeted_pooling_dict[col]
            
            # Generate a 6-character MD5 hash of the index list to ensure cache integrity
            list_string = str(target_indices).encode('utf-8')
            short_hash = hashlib.md5(list_string).hexdigest()[:6]
            
            # Create the dynamic prefix 
            t_prefix = f"Targeted_{semantic_tag}_{short_hash}_"
            extraction_passes.append((True, t_prefix, target_indices))
        
        # 1. AAC
        if 'aac' in active_features:
            for is_targeted, t_prefix, t_indices in extraction_passes:
                aac_cache = os.path.join(cache_dir, f"{col}_{t_prefix}aac_features_N{expected_dataset_len}.npz")
                cached_dict = _load_valid_cache(aac_cache, expected_dataset_len) if not is_inference else None
                if cached_dict:
                    print(f"   -> [{t_prefix}AAC] Loaded from cache")
                    for k, v in cached_dict.items(): new_columns[k] = v; generated_features.append(k)
                else:
                    print(f"   -> [{t_prefix}AAC] Calculating new features...")
                    aac_dict = compute_aac_features(seqs, prefix=f"{col}_{t_prefix}", target_indices=t_indices)
                    if not is_inference: np.savez(aac_cache, **aac_dict)
                    for k, v in aac_dict.items(): new_columns[k] = v; generated_features.append(k)
                
        # 2. AAindex
        if 'aaindex' in active_features:
            for is_targeted, t_prefix, t_indices in extraction_passes:
                aaindex_cache = os.path.join(cache_dir, f"{col}_{t_prefix}aaindex_features_N{expected_dataset_len}.npz")
                cached_dict = _load_valid_cache(aaindex_cache, expected_dataset_len) if not is_inference else None
                if cached_dict:
                    print(f"   -> [{t_prefix}AAindex] Loaded from cache")
                    for k, v in cached_dict.items(): new_columns[k] = v; generated_features.append(k)
                else:
                    print(f"   -> [{t_prefix}AAindex] Calculating new features...")
                    aaindex_dict = compute_aaindex_features(seqs, aaindex_db, prefix=f"{col}_{t_prefix}", target_indices=t_indices)
                    if aaindex_dict:
                        if not is_inference: np.savez(aaindex_cache, **aaindex_dict)
                        for k, v in aaindex_dict.items(): new_columns[k] = v; generated_features.append(k)
                    
        # 3. Georgiev
        if 'georgiev' in active_features:
            for is_targeted, t_prefix, t_indices in extraction_passes:
                geo_cache = os.path.join(cache_dir, f"{col}_{t_prefix}georgiev_features_N{expected_dataset_len}.npz")
                cached_dict = _load_valid_cache(geo_cache, expected_dataset_len) if not is_inference else None
                if cached_dict:
                    print(f"   -> [{t_prefix}Georgiev] Loaded from cache")
                    for k, v in cached_dict.items(): new_columns[k] = v; generated_features.append(k)
                else:
                    print(f"   -> [{t_prefix}Georgiev] Calculating new features...")
                    geo_dict = compute_georgiev_features(seqs, GEORGIEV_DICT, prefix=f"{col}_{t_prefix}", target_indices=t_indices)
                    if not is_inference: np.savez(geo_cache, **geo_dict)
                    for k, v in geo_dict.items(): new_columns[k] = v; generated_features.append(k)

        # 4. ESM-2
        if 'esm' in active_features and ESM_AVAILABLE:
            for esm_model_name in esm_model_names:
                esm_tag = "ESM_Small_8M" if "8M" in esm_model_name else ("ESM_Medium_35M" if "35M" in esm_model_name else ("ESM_Big_650M" if "650M" in esm_model_name else "ESM_Custom"))
                
                for is_targeted, t_prefix, t_indices in extraction_passes:
                    esm_cache = os.path.join(cache_dir, f"{col}_{t_prefix}{esm_tag}_features_N{expected_dataset_len}.npz")
                    cached_dict = _load_valid_cache(esm_cache, expected_dataset_len) if not is_inference else None
                    
                    if cached_dict:
                        print(f"   -> [{t_prefix}{esm_tag}] Loaded from cache")
                        for k, v in cached_dict.items(): new_columns[k] = v; generated_features.append(k)
                    else:
                        print(f"   -> [{t_prefix}{esm_tag}] Generating neural embeddings...")
                        esm_dict, raw_embeddings_matrix = compute_esm_embeddings(seqs, esm_model_name, prefix=f"{col}_{t_prefix}", esm_tag=esm_tag, target_indices=t_indices)
                        if not is_inference: np.savez(esm_cache, **esm_dict)
                        for k, v in esm_dict.items(): new_columns[k] = v; generated_features.append(k)
                        
                    # SVD compression applies naturally to both Global and Targeted embedding spaces
                    if esm_tag == "ESM_Big_650M":
                        svd_cache = os.path.join(cache_dir, f"{col}_{t_prefix}{esm_tag}_SVD50_features_N{expected_dataset_len}.npz")
                        svd_model_path = os.path.join(cache_dir, f"{col}_{t_prefix}{esm_tag}_SVD50_model.joblib")
                        
                        cached_svd = _load_valid_cache(svd_cache, expected_dataset_len) if not is_inference else None
                        
                        # In inference, we MUST apply the saved SVD matrix to avoid test set data leakage
                        if is_inference:
                            svd_obj = svd_models_dict.get(f"{col}_{t_prefix}{esm_tag}")
                            if svd_obj:
                                print(f"   -> [{t_prefix}{esm_tag}_SVD50] Projecting using pre-fitted SVD...")
                                svd_dict, _ = compress_embeddings_svd(raw_embeddings_matrix, prefix=f"{col}_{t_prefix}", esm_tag=esm_tag, svd_model=svd_obj)
                                for k, v in svd_dict.items(): new_columns[k] = v; generated_features.append(k)
                            else:
                                print(f"   -> ⚠️ ERROR: No fitted SVD model for {col}_{t_prefix}{esm_tag}")
                                
                        elif cached_svd and os.path.exists(svd_model_path):
                            print(f"   -> [{t_prefix}{esm_tag}_SVD50] Loaded features and model from cache")
                            svd_models_dict[f"{col}_{t_prefix}{esm_tag}"] = joblib.load(svd_model_path)
                            for k, v in cached_svd.items(): new_columns[k] = v; generated_features.append(k)
                            
                        else:
                            print(f"   -> [{t_prefix}{esm_tag}_SVD50] Compressing 1280D to 50 dimensions using SVD...")
                            if cached_dict:
                                num_dims = len(cached_dict)
                                raw_embeddings_matrix = np.zeros((expected_dataset_len, num_dims))
                                for i in range(num_dims): raw_embeddings_matrix[:, i] = cached_dict[f'{col}_{t_prefix}{esm_tag}_{i}']
                                    
                            svd_dict, fitted_svd = compress_embeddings_svd(raw_embeddings_matrix, prefix=f"{col}_{t_prefix}", esm_tag=esm_tag)
                            if svd_dict and fitted_svd:
                                np.savez(svd_cache, **svd_dict)
                                joblib.dump(fitted_svd, svd_model_path)
                                svd_models_dict[f"{col}_{t_prefix}{esm_tag}"] = fitted_svd
                                for k, v in svd_dict.items(): new_columns[k] = v; generated_features.append(k)
                            else:
                                print(f"   -> ⚠️ Dataset too small for SVD compression. Skipping.")

        # 5. AntiBERTy
        if 'antiberty' in active_features and ANTIBERTY_AVAILABLE:
            antiberty_cache = os.path.join(cache_dir, f"{col}_AntiBERTy_features_N{expected_dataset_len}.npz")
            cached_dict = _load_valid_cache(antiberty_cache, expected_dataset_len) if not is_inference else None
            
            if cached_dict:
                print("   -> [AntiBERTy] Loaded from cache")
                for k, v in cached_dict.items(): new_columns[k] = v; generated_features.append(k)
            else:
                print("   -> [AntiBERTy] Generating antibody-specific embeddings...")
                antiberty_dict = compute_antiberty_embeddings(seqs, prefix=f"{col}_")
                if not is_inference: np.savez(antiberty_cache, **antiberty_dict)
                for k, v in antiberty_dict.items(): new_columns[k] = v; generated_features.append(k)

    # 6. AbLang2 PAIRED
    if any(k in active_features for k in ['ablang2', 'ablang2_paired']) and ABLANG2_AVAILABLE:
        print("\n⚙️  Processing region: Paired [CD3_VH + CD3_VL]")
        ablang2_cache = os.path.join(cache_dir, f"Paired_CD3_VH_VL_AbLang2_features_N{expected_dataset_len}.npz")
        cached_dict = _load_valid_cache(ablang2_cache, expected_dataset_len) if not is_inference else None
        
        if cached_dict:
            print("   -> [AbLang2 Paired] Loaded from cache")
            for k, v in cached_dict.items(): new_columns[k] = v; generated_features.append(k)
        else:
            print("   -> [AbLang2 Paired] Generating Heavy/Light joint embeddings...")
            heavy_seqs = df_feat['CD3_VH'].astype(str).str.replace(r'\s+|,', '', regex=True).str.upper().tolist()
            light_seqs = df_feat['CD3_VL'].astype(str).str.replace(r'\s+|,', '', regex=True).str.upper().tolist()
            ablang_dict = compute_ablang2_paired_embeddings(heavy_seqs, light_seqs)
            
            if not is_inference: np.savez(ablang2_cache, **ablang_dict)
            for k, v in ablang_dict.items(): new_columns[k] = v; generated_features.append(k)

    # 7. PROPERMAB PHYSICS
    if 'propermab' in active_features and PROPERMAB_AVAILABLE and 'CD3_VH' in df_feat.columns and 'CD3_VL' in df_feat.columns:
        print("\n⚙️  Processing region: Propermab 3D Physics [CD3_VH + CD3_VL]")
        propermab_cache = os.path.join(cache_dir, f"Propermab_CD3_features_N{expected_dataset_len}.npz")
        cached_dict = _load_valid_cache(propermab_cache, expected_dataset_len) if not is_inference else None
        
        if cached_dict:
            print("   -> [Propermab] Loaded from cache")
            for k, v in cached_dict.items(): new_columns[k] = v; generated_features.append(k)
        else:
            print("   -> [Propermab] Generating 3D Structural Features (This takes ~30s per sequence)...")
            heavy_seqs = df_feat['CD3_VH'].astype(str).str.replace(r'\s+|,', '', regex=True).str.upper().tolist()
            light_seqs = df_feat['CD3_VL'].astype(str).str.replace(r'\s+|,', '', regex=True).str.upper().tolist()
            
            propermab_dict = compute_propermab_features(heavy_seqs, light_seqs, prefix="Propermab_CD3_")
            
            if not is_inference: np.savez(propermab_cache, **propermab_dict)
            for k, v in propermab_dict.items(): new_columns[k] = v; generated_features.append(k)

    if new_columns:
        df_feat = pd.concat([df_feat, pd.DataFrame(new_columns)], axis=1)
        
    print("\n✅ FEATURE EXTRACTION COMPLETE!")
    return df_feat, seq_cols, generated_features, aaindex_desc, svd_models_dict

def get_model(model_name, n_features, transform_type=None, weight_col=None):
    """
    Constructs an un-fitted scikit-learn pipeline or GridSearchCV object.
    Automatically applies variance thresholds, scaling, and regularization constraints.
    
    Args:
        model_name (str): Identifier for the algorithm (e.g., 'XGBoost', 'SVR').
        n_features (int): Number of features (used to cap dimensions for PLS).
        transform_type (str): Optional target transformation string.
        weight_col (str): Column representing sample weights.
        
    Returns:
        estimator: GridSearchCV object or base Pipeline.
    """
    scoring = {'rmse': 'neg_root_mean_squared_error', 'mae': 'neg_mean_absolute_error', 'r2': 'r2', 'spearman': make_scorer(custom_spearman)}
    
    if model_name == 'RandomForest':
        base = RandomForestRegressor(n_estimators=100, random_state=42)
        grid = {}
    elif model_name == 'PLSRegression':
        base = Pipeline([('vt', VarianceThreshold()), ('scaler', StandardScaler()), ('pls', PLSRegression())])
        max_comp = min(10, n_features)
        grid = {'pls__n_components': range(2, max_comp + 1)} if max_comp >= 2 else {}
        if max_comp < 2: base.set_params(pls__n_components=1)
    elif model_name == 'ElasticNet':
        base = Pipeline([('vt', VarianceThreshold()), ('scaler', StandardScaler()), ('enet', ElasticNet(max_iter=10000, random_state=42))])
        grid = {'enet__alpha': [0.01, 0.1, 1.0, 10.0], 'enet__l1_ratio': [0.1, 0.5, 0.9]}
    elif model_name == 'SVR':
        base = Pipeline([('vt', VarianceThreshold()), ('scaler', StandardScaler()), ('svr', SVR())])
        grid = {'svr__kernel': ['linear'], 'svr__C': [0.001, 0.01, 0.1, 1.0], 'svr__epsilon':[0.01, 0.1, 0.5, 1.0]}
    elif model_name == 'XGBoost':
        base = Pipeline(steps=[
            ('vt', VarianceThreshold()),
            ('scaler', StandardScaler()),
            ('model', XGBRegressor(random_state=42, n_jobs=1, objective='reg:squarederror'))
        ])
        
        # Regularization grid designed to prevent scaffold memorization in shallow trees
        grid = {
            'model__n_estimators': [200],#[200],    
            'model__max_depth': [3],#[6],               # VERY shallow trees (prevents complex memorization rules)
            'model__learning_rate': [0.1],#[0.3],
            'model__subsample': [0.8], #[1],          # Forces the model to ignore 20-40% of the sequences per tree
            'model__colsample_bytree': [0.8], # [1]   # Forces the model to ignore 20-50% of the ESM features per tree
            'model__reg_alpha': [0.1, 1.0],#[0]           # L1 (Lasso) Penalty to crush useless features to 0
            'model__reg_lambda': [1.0, 5.0, 10.0] #[1.0]#    # L2 (Ridge) Penalty to keep feature weights small and stable
        }
        
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    model = SelfContainedTargetTransformRegressor(regressor=base, transform_type=transform_type, weight_col=weight_col)
    grid = {f'regressor__{k}': v for k, v in grid.items()}
    
    return GridSearchCV(model, grid, cv=3, scoring=scoring, refit='spearman', n_jobs=1) if grid else model

def plot_enrichment_comparison(y_true, y_pred, target_col, ax=None, top_quantile=0.2):
    """
    Plots an enrichment curve showing the model's ability to identify top-performing variants 
    compared to a random baseline selection strategy.
    
    Args:
        y_true (array-like): Ground truth values.
        y_pred (array-like): Predicted values.
        target_col (str): Name of the target variable for chart labeling.
        ax (matplotlib.axes.Axes): Optional axis to draw on.
        top_quantile (float): The fraction representing the "top tier" hits (default 0.2/20%).
    """
    if ax is None: fig, ax = plt.subplots(figsize=(8, 6))
    
    y_true_flat = np.asarray(y_true).flatten()
    y_pred_flat = np.asarray(y_pred).flatten()
    
    df = pd.DataFrame({'Actual': y_true_flat, 'Predicted': y_pred_flat})
    total_library = len(df)
    target_lower = target_col.lower()
    lower_is_better = any(t in target_lower for t in ['hmw', 'agg', 'viscosity', 'lmw', 'hcp', 'clearance'])
    
    df_sorted = df.sort_values(by='Predicted', ascending=lower_is_better).reset_index(drop=True)
    total_samples = len(df)
    
    true_top_threshold = df['Actual'].quantile(top_quantile) if lower_is_better else df['Actual'].quantile(1 - top_quantile)
    
    enrichment_counts = []
    x_abs_counts = list(range(1, total_samples + 1))
    
    for k in x_abs_counts:
        top_k = df_sorted.head(k)
        if lower_is_better: hits = (top_k['Actual'] <= true_top_threshold).sum()
        else: hits = (top_k['Actual'] >= true_top_threshold).sum()
        enrichment_counts.append(hits)
    total_top_performers = enrichment_counts[-1]    
    random_expected = [k * top_quantile for k in x_abs_counts]
    
    ax.plot(x_abs_counts, enrichment_counts, label=f'Model Selection Hits', color='blue', lw=2.5)
    ax.plot(x_abs_counts, random_expected, 'k--', label='Random Selection (Expected)', lw=2)
    ax.fill_between(x_abs_counts, enrichment_counts, random_expected, color='dodgerblue', alpha=0.15)

    ax.set_xlim(0, total_library+1)
    ax.set_ylim(0, total_top_performers+1)
    
    x_ticks = [int(t) for t in ax.get_xticks() if 0 <= t < total_library * 0.95]
    x_ticks.append(total_library)
    ax.set_xticks(x_ticks)
    
    y_ticks = [int(t) for t in ax.get_yticks() if 0 <= t < total_top_performers * 0.95]
    y_ticks.append(total_top_performers)
    ax.set_yticks(y_ticks)
    
    ax.set_title(f'Enrichment of Top {int(top_quantile*100)}% Hits\n({target_col})', fontsize=14)
    ax.set_xlabel('Number of Antibodies Synthesized / Tested', fontsize=12)
    ax.set_ylabel('Absolute Number of True Hits Discovered', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

def plot_best_model_diagnostics(X, y, subregions_name, features_name, model_name, target_col, output_dir, final_estimator, 
                                best_params=None, feature_tag="", prefix="", top_quantile=0.2, hue_data=None, hue_name=None):
    """
    Generates a 2x2 diagnostic grid (Scatter, Enrichment, PR-AUC, ROC-AUC) utilizing out-of-fold predictions.
    
    Args:
        X (pd.DataFrame): Final optimized feature matrix.
        y (pd.Series): Target array.
        subregions_name (str): The structural regions involved (for labeling).
        features_name (str): The feature combinations involved (for labeling).
        model_name (str): Model algorithm identifier.
        target_col (str): Name of the target variable.
        output_dir (str): Save directory.
        final_estimator (estimator): The locked/best model.
        best_params (dict): Optional model hyperparameters to display.
        feature_tag (str): Formatted ID for saving files safely.
        prefix (str): Prefix tag (e.g., 'global_').
        top_quantile (float): Hits threshold for enrichment curves.
        hue_data (array-like): Array used to colorize scatter points by class/dataset.
        hue_name (str): Label for the hue data.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score
    print(f"\nGenerating 2x2 diagnostic grid for {subregions_name} + {features_name}...")
    
    # 2x2 Grid
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    ax_scatter, ax_enrich = axes[0, 0], axes[0, 1]
    ax_pr, ax_roc = axes[1, 0], axes[1, 1]
    
    title_text = f"Diagnostic Analysis: {target_col}\nModel: {model_name}  |  Regions: [{subregions_name}]  | Features: [{features_name}]"
    fig.suptitle(title_text, fontsize=18, y=1.02, fontweight='bold')
    
    try:
        # Cross-validation for the plots
        loo = 5
        cv_preds = cross_val_predict(final_estimator, X, y, cv=loo, n_jobs=-1)
        
        y_flat = np.asarray(y).flatten()
        cv_preds_flat = np.asarray(cv_preds).flatten()
        
        # ==========================================
        # PANEL 1 (Top Left): PREDICTED VS ACTUAL
        # ==========================================
        c_r2 = r2_score(y_flat, cv_preds_flat)
        c_rmse = np.sqrt(mean_squared_error(y_flat, cv_preds_flat))
        c_mae = mean_absolute_error(y_flat, cv_preds_flat)
        c_spear = custom_spearman(y_flat, cv_preds_flat)
        
        metrics_text = f"Spearman: {c_spear:.3f}\nR² Score: {c_r2:.3f}\nRMSE: {c_rmse:.2f}\nMAE: {c_mae:.2f}"
        if best_params: metrics_text += "\n\nHyperparameters:\n" + "\n".join([f"{k.split('__')[-1]}: {v}" for k, v in best_params.items()])
            
        plot_df = pd.DataFrame({'Actual': y_flat, 'Predicted': cv_preds_flat})
            
        if hue_data is not None and hue_name is not None:
            plot_df[hue_name] = np.asarray(hue_data).flatten()
            sns.scatterplot(data=plot_df, x='Actual', y='Predicted', hue=hue_name, palette='tab10', ax=ax_scatter, alpha=0.8, edgecolor='k', s=80)
        else:
            sns.scatterplot(data=plot_df, x='Actual', y='Predicted', ax=ax_scatter, alpha=0.8, edgecolor='k', s=80, color='dodgerblue')
        
        min_val, max_val = min(y_flat.min(), cv_preds_flat.min()), max(y_flat.max(), cv_preds_flat.max())
        ax_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label="Perfect Prediction")
        
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
        ax_scatter.text(0.05, 0.95, metrics_text, transform=ax_scatter.transAxes, fontsize=11, verticalalignment='top', bbox=props)
        ax_scatter.set_title('Predicted vs Actual (5-Fold CV)')
        ax_scatter.legend(loc='lower right')

        # ==========================================
        # PANEL 2 (Top Right): ENRICHMENT PLOT
        # ==========================================
        plot_enrichment_comparison(y_flat, cv_preds_flat, target_col, ax=ax_enrich, top_quantile=top_quantile)
        
        # ==========================================
        # PANELS 3 & 4 (Bottom): PR-AUC & ROC-AUC SWEEPS
        # ==========================================
        target_lower = target_col.lower()
        lower_is_better = any(t in target_lower for t in ['hmw', 'agg', 'viscosity', 'lmw', 'hcp', 'clearance', 'poly'])
        
        thresholds = np.linspace(np.percentile(y_flat, 10), np.percentile(y_flat, 90), 40)
        pr_aucs, roc_aucs, baselines, valid_t = [], [], [], []
        
        for t in thresholds:
            if lower_is_better:
                y_bin = (y_flat <= t).astype(int)
                y_score = -cv_preds_flat 
            else:
                y_bin = (y_flat >= t).astype(int)
                y_score = cv_preds_flat
                
            if len(np.unique(y_bin)) == 2:
                prauc = average_precision_score(y_bin, y_score)
                rocauc = roc_auc_score(y_bin, y_score)
                baseline = y_bin.mean() 
                
                pr_aucs.append(prauc)
                roc_aucs.append(rocauc)
                baselines.append(baseline)
                valid_t.append(t)
        
        if valid_t:
            # PR-AUC Plot
            ax_pr.plot(valid_t, pr_aucs, label='Model PR-AUC', color='darkviolet', lw=2.5)
            ax_pr.plot(valid_t, baselines, 'k--', label='Random Baseline (Prevalence)', lw=2)
            ax_pr.fill_between(valid_t, pr_aucs, baselines, color='darkviolet', alpha=0.15)
            ax_pr.set_title(f'Robustness: PR-AUC vs Cutoff\n({target_col})', fontsize=14)
            ax_pr.set_xlabel(f'{target_col} Cutoff', fontsize=12)
            ax_pr.set_ylabel('Precision-Recall AUC', fontsize=12)
            ax_pr.legend(loc='best')
            ax_pr.grid(True, alpha=0.3)
            
            # ROC-AUC Plot
            ax_roc.plot(valid_t, roc_aucs, label='Model ROC-AUC', color='forestgreen', lw=2.5)
            ax_roc.plot(valid_t, [0.5]*len(valid_t), 'k--', label='Random Baseline (0.5)', lw=2)
            ax_roc.fill_between(valid_t, roc_aucs, [0.5]*len(valid_t), color='forestgreen', alpha=0.15)
            ax_roc.set_title(f'Robustness: ROC-AUC vs Cutoff\n({target_col})', fontsize=14)
            ax_roc.set_xlabel(f'{target_col} Cutoff', fontsize=12)
            ax_roc.set_ylabel('ROC AUC', fontsize=12)
            ax_roc.legend(loc='lower right')
            ax_roc.grid(True, alpha=0.3)
        else:
            for ax in [ax_pr, ax_roc]:
                ax.text(0.5, 0.5, "Insufficient variance\nfor Threshold Sweeps", ha='center', va='center')
                ax.set_title("AUC Sweeps")
            
    except Exception as e: print(f"Warning: Plot failed: {e}")
        
    plt.tight_layout()
    feat_suffix = f"_{feature_tag}" if feature_tag else ""
    plt.savefig(os.path.join(output_dir, f'best_{prefix}{model_name}_{target_col}{feat_suffix}_diagnostics.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_best_model_diagnostics_old(X, y, subregions_name, features_name, model_name, target_col, output_dir, final_estimator, best_params=None, threshold=None, transform_type=None, feature_tag="", hue_data=None, hue_name=None, prefix="", top_quantile=0.2):
    """
    Generates an extended 3x3 diagnostic dashboard, tracking residual distributions 
    and confusion matrices under Leave-One-Out Cross-Validation.
    """
    transform_suffix = f"_{transform_type}" if transform_type else ""
    feat_suffix = f"_{feature_tag}" if feature_tag else ""
    title_tag = f" ({transform_type.title()} Transformed)" if transform_type else ""
    
    combo_name = f"{subregions_name} | {features_name}"
    print(f"\nGenerating extended diagnostic plots for ({combo_name}) using {model_name}...")
    
    # 3x3 Grid (9 slots total)
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    fig.suptitle(f'Diagnostic Plots: Best Model ({combo_name} | {model_name} | {target_col}){title_tag}', fontsize=18, y=0.98)

    try:
        loo = LeaveOneOut()
        cv_preds = cross_val_predict(final_estimator, X, y, cv=loo, n_jobs=-1)

        y_flat = np.asarray(y).flatten()
        cv_preds_flat = np.asarray(cv_preds).flatten()
        residuals = y_flat - cv_preds_flat
        calc_r2 = r2_score(y_flat, cv_preds_flat)
        calc_rmse = np.sqrt(mean_squared_error(y_flat, cv_preds_flat))
        calc_mae = mean_absolute_error(y_flat, cv_preds_flat)
        calc_spearman = custom_spearman(y_flat, cv_preds_flat)
        
        if threshold is None:
            threshold = np.median(y_flat)
            
        y_binary = (y_flat >= threshold).astype(int)
        cv_preds_binary = (cv_preds_flat >= threshold).astype(int)
        acc = accuracy_score(y_binary, cv_preds_binary)
        prec = precision_score(y_binary, cv_preds_binary, zero_division=0)
        rec = recall_score(y_binary, cv_preds_binary, zero_division=0)
        f1 = f1_score(y_binary, cv_preds_binary, zero_division=0)
        cm = confusion_matrix(y_binary, cv_preds_binary)

        try:
            roc_auc = roc_auc_score(y_binary, cv_preds_flat)
            pr_auc = average_precision_score(y_binary, cv_preds_flat)
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
        for actual, pred in zip(y_flat, cv_preds_flat):
            if actual >= threshold and pred >= threshold:
                classification_labels.append('True Positive (TP)')
            elif actual < threshold and pred < threshold:
                classification_labels.append('True Negative (TN)')
            elif actual < threshold and pred >= threshold:
                classification_labels.append('False Positive (FP)')
            else:
                classification_labels.append('False Negative (FN)')
                
        plot_df = pd.DataFrame({'Actual': y_flat, 'Predicted': cv_preds_flat, 'Classification': classification_labels})
        plot_df_res = pd.DataFrame({'Predicted': cv_preds_flat, 'Residuals': residuals})

        style_col = None

        if hue_data is not None and hue_name is not None:
            style_col = hue_name
            plot_df[style_col] = np.asarray(hue_data).flatten()
            plot_df_res[style_col] = np.asarray(hue_data).flatten()

        custom_palette = {
            'True Negative (TN)': '#2ca02c', 'True Positive (TP)': '#d62728',
            'False Positive (FP)': '#ff7f0e', 'False Negative (FN)': '#1f77b4'
        }

        # --- Plot 1: Predicted vs Actual ---
        sns.scatterplot(data=plot_df, x='Actual', y='Predicted', hue='Classification',
                        style=style_col, palette=custom_palette, ax=axes[0, 0], alpha=0.8, edgecolor='k', s=60)
        axes[0, 0].axvline(threshold, color='gray', linestyle=':', linewidth=1.5, label=f'Threshold ({threshold:.1f})')
        axes[0, 0].axhline(threshold, color='gray', linestyle=':', linewidth=1.5)
        min_val = min(y_flat.min(), cv_preds_flat.min())
        max_val = max(y_flat.max(), cv_preds_flat.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        axes[0, 0].set_title(f'Predicted vs Actual {target_col} (LOO CV)')
        axes[0, 0].set_xlabel(f'Actual {target_col}')
        axes[0, 0].set_ylabel(f'Predicted {target_col} (Out-of-Fold)')
        axes[0, 0].legend(loc='lower right', fontsize=9)

        # --- Plot 2: Residuals vs Predicted ---
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
        # --- Plot 5: Enrichment Comparison ---
        plot_enrichment_comparison(y_flat, cv_preds_flat, target_col, ax=axes[1, 1], top_quantile=top_quantile)
        # ==========================================
        # ROW 3: SUMMARY & METRICS DASHBOARD
        # ==========================================
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
    plot_filename = os.path.join(output_dir, f'best_{prefix}{model_name}_{target_col}{transform_suffix}{feat_suffix}_EXTENDED_diagnostics.png')

    plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved Extended Plot '{plot_filename}'")

def plot_out_of_group_diagnostics(X, y, group_labels, subregions_name, features_name, model_name, target_col, output_dir, final_estimator, feature_tag="", prefix="", split_col_name=""):
    """
    Evaluates model generalizability by sequentially training on one custom subgroup (e.g., variant lineage)
    and testing exclusively on all unseen subgroups, generating scatter plots and a performance heatmap.
    
    Args:
        X (pd.DataFrame): Final optimized feature matrix.
        y (pd.Series): Target array.
        group_labels (array-like): Column designating which lineage/batch a variant belongs to.
        subregions_name (str): Label for structural domains.
        features_name (str): Label for feature combo.
        model_name (str): The algorithm in use.
        target_col (str): The target predicted.
        output_dir (str): File destination.
        final_estimator (estimator): Scikit-learn estimator instance to be cloned.
        feature_tag (str): Unique file naming tag.
        prefix (str): Preceding tag for saving files safely.
        split_col_name (str): Label representing the Out-Of-Group column.
    """
    print(f"\nGenerating Out-of-Group (OOG) diagnostic plots for {target_col}...")
    
    group_labels = np.asarray(group_labels).flatten()
    y_flat = np.asarray(y).flatten()
    
    ignore_values = ['IGNORE', 'EXCLUDE']
    
    valid_mask = []
    for val in group_labels:
        if pd.isna(val):
            valid_mask.append(False)
        elif str(val).strip().upper() in ignore_values:
            valid_mask.append(False)
        else:
            valid_mask.append(True)
            
    valid_mask = np.array(valid_mask)
    group_labels = group_labels[valid_mask]
    y_flat = y_flat[valid_mask]
    
    if hasattr(X, 'iloc'):
        X = X.iloc[valid_mask]
    else:
        X = X[valid_mask]
        
    unique_groups = np.unique(group_labels)
    num_groups = len(unique_groups)
    
    print(f"🔍 Detected {num_groups} valid unique groups in split column '{split_col_name}': {list(unique_groups)}")
    
    if num_groups < 2:
        print("  -> ⚠️ Not enough groups to perform out-of-group testing.")
        return
        
    if num_groups > 10:
        print(f"  -> ⚠️ Skipping out-of-group combinations: {num_groups} groups is too many (Limit is 10 to prevent plot explosion).")
        return

    # Determine Grid Size for Subplots (Max 3 columns wide)
    cols = min(3, num_groups)
    rows = math.ceil(num_groups / cols)
    
    fig_scatter, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
    # Ensure axes is always a flat array for easy iteration, even if it's 1x1 or 1xN
    axes = np.array(axes).flatten()
    
    title_text = f"Generalization: {target_col}\nModel: {model_name}  |  Regions: [{subregions_name}]"
    fig_scatter.suptitle(title_text, fontsize=18, y=1.02 + (0.02 if rows==1 else 0), fontweight='bold')
    
    results = []
    feat_suffix = f"_{feature_tag}" if feature_tag else ""
    split_name_tag = f"split_by_{split_col_name}_" if split_col_name else ""
    
    for idx, train_grp in enumerate(unique_groups):
        ax = axes[idx]
        mask_train = (group_labels == train_grp)
        
        # Safely index X
        if hasattr(X, 'iloc'):
            X_train = X.iloc[mask_train]
        else:
            X_train = X[mask_train]
            
        y_train = y_flat[mask_train]
        
        if len(y_train) < 5:
            print(f"     ⚠️ Skipping Train [{train_grp}] (Insufficient data: N={len(y_train)})")
            ax.set_visible(False)
            continue
            
        # Wrap the fitting step in a Try/Except block to catch Zero Variance errors!
        model = clone(final_estimator)
        try:
            model.fit(X_train, y_train)
            train_preds = model.predict(X_train).flatten()
        except ValueError as e:
            print(f"     ⚠️ Skipping Train [{train_grp}] (Zero variance in this specific split: {e})")
            ax.set_visible(False)
            continue
        except Exception as e:
            print(f"     ⚠️ Skipping Train [{train_grp}] (Model fit failed: {e})")
            ax.set_visible(False)
            continue
        
        # Prepare data for this specific subplot
        plot_df_list = []
        plot_df_list.append(pd.DataFrame({
            'Actual': y_train, 
            'Predicted': train_preds, 
            'Dataset': f'Train ({train_grp})'
        }))
        
        metrics_lines = ["Unseen Test Metrics:"]
        palette = {f'Train ({train_grp})': 'lightgray'}
        
        # Generate a distinct color palette for the test groups
        test_colors = sns.color_palette("husl", num_groups - 1)
        color_idx = 0
        
        # Test on every OTHER group
        for test_grp in unique_groups:
            if test_grp == train_grp:
                continue
                
            mask_test = (group_labels == test_grp)
            if hasattr(X, 'iloc'):
                X_test = X.iloc[mask_test]
            else:
                X_test = X[mask_test]
                
            y_test = y_flat[mask_test]
            
            if len(y_test) < 5:
                continue
                
            test_preds = model.predict(X_test).flatten()
            
            # Calculate strictly on unseen test data
            c_r2 = r2_score(y_test, test_preds)
            try:
                c_spear = custom_spearman(y_test, test_preds)
            except Exception:
                c_spear = 0.0
                
            results.append({
                'Train_Group': train_grp,
                'Test_Group': test_grp,
                'Spearman': c_spear,
                'R2': c_r2,
                'N_Train': len(y_train),
                'N_Test': len(y_test)
            })
            
            print(f"     ✅ Train [{str(train_grp):^10}] ➔ Test [{str(test_grp):^10}] | Spearman: {c_spear:^6.3f} | R²: {c_r2:^6.3f}")
            
            dataset_label = f'Test ({test_grp})'
            plot_df_list.append(pd.DataFrame({
                'Actual': y_test, 
                'Predicted': test_preds, 
                'Dataset': dataset_label
            }))
            
            palette[dataset_label] = test_colors[color_idx]
            color_idx += 1
            metrics_lines.append(f"[{test_grp}] Sp: {c_spear:.2f} | R²: {c_r2:.2f}")

        # Plot the combined scatter
        plot_df = pd.concat(plot_df_list, ignore_index=True)
        sns.scatterplot(data=plot_df, x='Actual', y='Predicted', hue='Dataset', 
                        palette=palette, ax=ax, alpha=0.8, edgecolor='k', s=70)
        
        # Perfect prediction diagonal line
        min_val = plot_df[['Actual', 'Predicted']].min().min()
        max_val = plot_df[['Actual', 'Predicted']].max().max()
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label="Perfect Line")
        
        # Add metrics text box
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
        ax.text(0.05, 0.95, "\n".join(metrics_lines), transform=ax.transAxes, 
                fontsize=9.5, verticalalignment='top', bbox=props)
        
        ax.set_title(f'Trained strictly on Group: {train_grp}', fontsize=14, pad=10)
        ax.set_xlabel(f'Actual {target_col}')
        ax.set_ylabel(f'Predicted {target_col}')
        ax.legend(loc='lower right', fontsize=9)
        
    # Hide any unused subplots (if num_groups doesn't perfectly fill the grid)
    for idx in range(num_groups, len(axes)):
        axes[idx].set_visible(False)
        
    fig_scatter.tight_layout()
    scatter_file = os.path.join(output_dir, f"oog_{split_name_tag}{prefix}{model_name}_{target_col}{feat_suffix}_Combined_Scatter.png")
    fig_scatter.savefig(scatter_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_scatter)

    if len(results) > 0:
        res_df = pd.DataFrame(results)
        pivot_spearman = res_df.pivot(index='Train_Group', columns='Test_Group', values='Spearman')
        
        # Dynamic figure sizing for heatmap
        fig_width = max(8, num_groups * 1.5)
        fig_height = max(6, num_groups * 1.2)
        
        plt.figure(figsize=(fig_width, fig_height))
        
        sns.heatmap(pivot_spearman, annot=True, cmap='coolwarm', center=0, fmt=".2f", 
                    linewidths=1, linecolor='white',
                    cbar_kws={'label': 'Spearman Correlation Score'})
                    
        plt.title(f"OOG Generalization Heatmap (Spearman Rank)\nTarget: {target_col} | Model: {model_name}", fontsize=14, pad=20)
        plt.ylabel(f"Model Trained On ({split_col_name})", fontsize=12, fontweight='bold')
        plt.xlabel(f"Model Tested On ({split_col_name})", fontsize=12, fontweight='bold')
        
        heatmap_file = os.path.join(output_dir, f"oog_{split_name_tag}{prefix}{model_name}_{target_col}{feat_suffix}_Summary_Heatmap.png")
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"\n  -> 📊 Saved OOG Scatter Plot and Summary Heatmap to '{output_dir}/'")

def generate_shap_analysis(model, X, y, output_dir, feature_names, model_name, target_col, prefix="", feature_tag="", aaindex_desc=None):
    """
    Computes SHAP (SHapley Additive exPlanations) values to interpret feature importance,
    generating a summary plot augmented with a biological feature dictionary.
    
    Args:
        model (estimator): Pre-fitted scikit-learn model.
        X (pd.DataFrame): Training feature matrix.
        y (pd.Series): Target array.
        output_dir (str): Location to save the image.
        feature_names (list): List of ordered feature strings.
        model_name (str): Identifier (e.g., 'XGBoost').
        target_col (str): Current target being predicted.
        prefix (str): Pipeline branch identifier.
        feature_tag (str): Global identifier for file saving.
        aaindex_desc (dict): Dictionary mapping index properties for the side-box legend.
    """
    print(f"\nExtracting SHAP feature importances for {model_name}...")
    try:
        import shap
        import re
        
        # Sanitize feature names to comply with XGBoost requirements
        clean_features = []
        for f in feature_names:
            f_clean = str(f).encode('ascii', 'ignore').decode('ascii')
            f_clean = re.sub(r'[\[\]<>]', '_', f_clean)
            clean_features.append(f_clean)
        feature_names = clean_features
        
        if isinstance(X, pd.DataFrame):
            X.columns = feature_names
        
        # Use base estimator directly if a cross-validation pipeline was provided
        if hasattr(model, 'best_estimator_'):
            model = model.best_estimator_
            
        # Fit the single winning pipeline on 100% of data 
        model.fit(X, y)
        
        # Safely extract the base pipeline
        base_pipe = model.regressor_ if hasattr(model, 'regressor_') else model
        
        if hasattr(base_pipe, 'named_steps') and 'vt' in base_pipe.named_steps:
            X_trans = base_pipe.named_steps['vt'].transform(X)
            if 'scaler' in base_pipe.named_steps: X_trans = base_pipe.named_steps['scaler'].transform(X_trans)
            active_feats = np.array(feature_names)[base_pipe.named_steps['vt'].get_support()]
            predictor = base_pipe.named_steps[list(base_pipe.named_steps.keys())[-1]]
        else:
            X_trans, active_feats, predictor = X.values if isinstance(X, pd.DataFrame) else X, feature_names, base_pipe

        # --- Robust Tree Model Check ---
        is_tree = model_name in ['RandomForest', 'XGBoost'] or type(predictor).__name__ in ['RandomForestRegressor', 'XGBRegressor']

        if is_tree:
            print("  -> Using TreeExplainer...")
            explainer = shap.TreeExplainer(predictor)
            shap_values = explainer.shap_values(X_trans)[1] if isinstance(explainer.shap_values(X_trans), list) else explainer.shap_values(X_trans)
        elif model_name in ['PLSRegression', 'ElasticNet'] or (model_name == 'SVR' and getattr(predictor, 'kernel', '') == 'linear'):
            print("  -> Using linear explainer...")
            coef = predictor.coef_.flatten()
            bg_mean = X_trans.mean(axis=0)
            shap_values = (X_trans - bg_mean) * coef
        else:
            print("  -> Compressing background to prevent memory allocation failure...")
            background = shap.kmeans(X_trans, 5)
            explainer = shap.KernelExplainer(predictor.predict, background)
            shap_values = explainer.shap_values(X_trans, nsamples=100, silent=True)

        aac_desc = {
            'A': 'Alanine', 'R': 'Arginine', 'N': 'Asparagine', 'D': 'Aspartic Acid',
            'C': 'Cysteine', 'Q': 'Glutamine', 'E': 'Glutamic Acid', 'G': 'Glycine',
            'H': 'Histidine', 'I': 'Isoleucine', 'L': 'Leucine', 'K': 'Lysine',
            'M': 'Methionine', 'F': 'Phenylalanine', 'P': 'Proline', 'S': 'Serine',
            'T': 'Threonine', 'W': 'Tryptophan', 'Y': 'Tyrosine', 'V': 'Valine'
        }
        
        display_features = []
        legend_dict = {} 
        for f in active_feats:
            if '_AAindex_' in f and aaindex_desc:
                try:
                    base_col, code = f.split('_AAindex_')
                    clean_name = f"{code}"
                    display_features.append(clean_name)
                    raw_desc = aaindex_desc.get(code, "Unknown property").split('(')[0].strip()
                    legend_dict[code] = raw_desc.encode('ascii', 'ignore').decode('ascii')
                except ValueError:
                    display_features.append(f)
            elif '_AAC_' in f:
                try:
                    base_col, aa = f.split('_AAC_')
                    clean_name = f"{base_col} | AAC_{aa}"
                    display_features.append(clean_name)
                    legend_dict[f"AAC_{aa}"] = f"{aac_desc.get(aa, 'Unknown')} Frequency"
                except ValueError:
                    display_features.append(f)
            else:
                display_features.append(f)

        # Calculate mean absolute SHAP values to find top 20 features
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-20:]
        top_features = [display_features[i] for i in top_indices]
        
        legend_lines = ["Feature Dictionary:"]
        import textwrap
        for feat in reversed(top_features): # Top-to-bottom plot order
            # 1. Exact match (for AAindex codes like 'KRIW790101')
            if feat in legend_dict and feat not in str(legend_lines):
                wrapped_desc = textwrap.fill(legend_dict[feat], width=65)
                legend_lines.append(f"• {feat}: {wrapped_desc}")
            # 2. Substring match (for AAC like 'seq_cdrh3 | AAC_W')
            else:
                for key in legend_dict:
                    if key in feat and key not in str(legend_lines):
                        wrapped_desc = textwrap.fill(legend_dict[key], width=65)
                        legend_lines.append(f"• {key}: {wrapped_desc}")
                        break
                        
        legend_text = "\n".join(legend_lines)
        fig = plt.figure(figsize=(22, 8))
        gs = fig.add_gridspec(1, 2, width_ratios=[2, 1.2])
        ax_shap = fig.add_subplot(gs[0])
        ax_text = fig.add_subplot(gs[1])
        # Draw the SHAP plot on the left axis
        plt.sca(ax_shap) 
        shap.summary_plot(shap_values, X_trans, feature_names=display_features, show=False)
        ax_shap.set_title(f"SHAP Value Impact: {model_name} on {target_col}", fontsize=14)
        # Draw the Dictionary Text Box on the right axis
        ax_text.axis('off')
        if len(legend_lines) > 1:
            props = dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', alpha=0.9, edgecolor='gray')
            ax_text.text(0.0, 0.95, legend_text, transform=ax_text.transAxes, fontsize=10,
                         verticalalignment='top', bbox=props, family='monospace') 
        feat_suffix = f"_{feature_tag}" if feature_tag else ""
        plt.savefig(os.path.join(output_dir, f'shap_{prefix}{model_name}_{target_col}{feat_suffix}.png'), dpi=300, facecolor='white', bbox_inches='tight')
        plt.close()
        print(f"✅ SHAP Summary Plot successfully saved!")
    except MemoryError: print("⚠️ SHAP Memory Error: The KernelExplainer ran out of RAM. Skipping plot.")
    except Exception as e: print(f"⚠️ SHAP failed: {e}")

def filter_active_features(all_features, sub_combo, feat_combo, global_features, custom_feature_groups=None):
    """
    Subsets the massive generated feature pool into a lean matrix specific to the current grid search permutation.
    Enforces strict mathematical isolation rules to prevent global and targeted features from overlapping.
    
    Args:
        all_features (list): Every single column name available in the dataframe.
        sub_combo (tuple): The structural regions currently being evaluated (e.g., ['CD3_VH']).
        feat_combo (tuple): The feature categories currently being evaluated (e.g., ['Targeted_AAC']).
        global_features (list): Base columns that should remain unaffected by subregion logic.
        custom_feature_groups (list): Additional CSV columns explicitly requested by the user.
        
    Returns:
        list: Alphabetically sorted list of exactly which feature columns should be sent to the model.
    """
    if custom_feature_groups is None: custom_feature_groups = []
    active = list(global_features)
    for f in all_features:
        if f in active: continue

        # Preserve explicit tabular CSV columns in the feature space
        if f in custom_feature_groups:
            if f in feat_combo: active.append(f)
            continue 

        # Bypass subregion checks for whole-molecule assays and metrics
        if f.startswith('CQA_'):
            if 'CQA' in feat_combo: active.append(f)
            continue

        # Structural models require paired VH/VL chains or an scFv
        if f.startswith('Propermab_'):
            if 'Propermab' in feat_combo and (('CD3_VH' in sub_combo and 'CD3_VL' in sub_combo) or 'scFv' in sub_combo):
                active.append(f)
            continue

        if f.startswith('Paired_CD3_VH_VL_AbLang2_'):
            if 'AbLang2_Paired' in feat_combo and (('CD3_VH' in sub_combo and 'CD3_VL' in sub_combo) or 'scFv' in sub_combo):
                active.append(f)
            continue
        
        has_sub = any(f.startswith(sub + '_') for sub in sub_combo)
        if not has_sub: continue
            
        # Filter sequence features based on targeted vs. global status to prevent leakage
        has_feat = False
        is_targeted_feat = '_Targeted_' in f
        
        for ft in feat_combo:
            if ft == 'AAC':
                if '_AAC_' in f and not is_targeted_feat: active.append(f); break
            elif ft == 'Targeted_AAC':
                if '_AAC_' in f and is_targeted_feat: active.append(f); break
                
            elif ft == 'AAindex':
                if '_AAindex_' in f and not is_targeted_feat: active.append(f); break
            elif ft == 'Targeted_AAindex':
                if '_AAindex_' in f and is_targeted_feat: active.append(f); break
                
            elif ft == 'Georgiev':
                if '_Georgiev_' in f and not is_targeted_feat: active.append(f); break
            elif ft == 'Targeted_Georgiev':
                if '_Georgiev_' in f and is_targeted_feat: active.append(f); break
                
            elif ft == 'ESM_Small_8M':
                if '_ESM_Small_8M_' in f and not is_targeted_feat: active.append(f); break
            elif ft == 'Targeted_ESM_Small_8M':
                if '_ESM_Small_8M_' in f and is_targeted_feat: active.append(f); break
                
            elif ft == 'ESM_Medium_35M':
                if '_ESM_Medium_35M_' in f and not is_targeted_feat: active.append(f); break
            elif ft == 'Targeted_ESM_Medium_35M':
                if '_ESM_Medium_35M_' in f and is_targeted_feat: active.append(f); break
                
            elif ft == 'ESM_Big_650M':
                if '_ESM_Big_650M_' in f and '_SVD50_' not in f and not is_targeted_feat: active.append(f); break
            elif ft == 'Targeted_ESM_Big_650M':
                if '_ESM_Big_650M_' in f and '_SVD50_' not in f and is_targeted_feat: active.append(f); break
                
            elif ft == 'ESM_Big_650M_SVD50':
                if '_ESM_Big_650M_SVD50_' in f and not is_targeted_feat: active.append(f); break
            elif ft == 'Targeted_ESM_Big_650M_SVD50':
                if '_ESM_Big_650M_SVD50_' in f and is_targeted_feat: active.append(f); break
                
            elif ft == 'AntiBERTy':
                if '_AntiBERTy_' in f and not is_targeted_feat: active.append(f); break
                
        is_untyped = not any(m in f for m in ['_AAC_', '_AAindex_', '_ESM_', '_Georgiev_', 'Propermab_', 'CQA_', '_AntiBERTy_', '_AbLang2_'])
        if has_feat or is_untyped: active.append(f)
            
    # Alphabetically sort the features to ensure stable output across permutations
    return sorted(list(set(active)))

def evaluate_exhaustive_combinations(df, seq_cols, generated_features, target_col, model_name, output_dir, 
                                     transform_type=None, weight_col=None, prefix="", hue_col=None, rank_to_plot=0,
                                     oog_col=None, aaindex_desc=None, extended_plots=False, manual_threshold=None,
                                     custom_feature_groups=None, exclude_groups=None):
    """
    Executes a fully automated combinatorial search across all designated subregions and feature groups.
    Enforces rules against multicollinearity and automatically produces a final ranked leaderboard.
    
    Args:
        df (pd.DataFrame): Dataset including targets and extracted features.
        seq_cols (list): Structural sequences available to combine (e.g., VH, VL).
        generated_features (list): Master list of all feature names present in df.
        target_col (str): Dependent variable to predict.
        model_name (str): Scikit-learn estimator ID (e.g., 'SVR').
        output_dir (str): Destination for all generated plots and CSVs.
        rank_to_plot (int): Leaderboard position to extract, fit, and plot at the end (0 = best model).
        oog_col (str): Column name mapping variants to out-of-group splits for validation.
        exclude_groups (list): Specific feature sets to omit from combinatorial logic.
    """
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_csv = os.path.join(output_dir, f"{prefix}exhaustive_search_checkpoint_{model_name}_{target_col}.csv")
    final_excel = os.path.join(output_dir, f"{prefix}exhaustive_search_results_{model_name}_{target_col}.xlsx")
    
    global_features = [
        f for f in generated_features 
        if not f.startswith('seq_CD3_')
        and not f.startswith('CD3_')
        and not f.startswith('scFv_')
        and not f.startswith('CQA_')
        and not f.startswith('Propermab_')
        and not f.startswith('Paired_') 
        and (custom_feature_groups is None or f not in custom_feature_groups)
    ]

    available_groups = []
    
    # Register available global sequence features
    if any('_AAC_' in f and '_Targeted_' not in f for f in generated_features): available_groups.append('AAC')
    if any('_AAindex_' in f and '_Targeted_' not in f for f in generated_features): available_groups.append('AAindex')
    if any('_Georgiev_' in f and '_Targeted_' not in f for f in generated_features): available_groups.append('Georgiev')
    
    if any('_ESM_Small_8M_' in f and '_Targeted_' not in f for f in generated_features): available_groups.append('ESM_Small_8M')
    if any('_ESM_Medium_35M_' in f and '_Targeted_' not in f for f in generated_features): available_groups.append('ESM_Medium_35M')
    if any('_ESM_Big_650M_' in f and '_SVD50_' not in f and '_Targeted_' not in f for f in generated_features): available_groups.append('ESM_Big_650M')
    if any('_ESM_Big_650M_SVD50_' in f and '_Targeted_' not in f for f in generated_features): available_groups.append('ESM_Big_650M_SVD50')
    
    # Register available targeted sequence features
    if any('_Targeted_' in f and '_AAC_' in f for f in generated_features): available_groups.append('Targeted_AAC')
    if any('_Targeted_' in f and '_AAindex_' in f for f in generated_features): available_groups.append('Targeted_AAindex')
    if any('_Targeted_' in f and '_Georgiev_' in f for f in generated_features): available_groups.append('Targeted_Georgiev')
    
    if any('_Targeted_' in f and '_ESM_Small_8M_' in f for f in generated_features): available_groups.append('Targeted_ESM_Small_8M')
    if any('_Targeted_' in f and '_ESM_Medium_35M_' in f for f in generated_features): available_groups.append('Targeted_ESM_Medium_35M')
    if any('_Targeted_' in f and '_ESM_Big_650M_' in f and '_SVD50_' not in f for f in generated_features): available_groups.append('Targeted_ESM_Big_650M')
    if any('_Targeted_' in f and '_ESM_Big_650M_SVD50_' in f for f in generated_features): available_groups.append('Targeted_ESM_Big_650M_SVD50')
    
    # Register advanced language models and 3D structural features
    if any('_AntiBERTy_' in f and '_Targeted_' not in f for f in generated_features): available_groups.append('AntiBERTy')
    if any('Paired_CD3_VH_VL_AbLang2_' in f for f in generated_features): available_groups.append('AbLang2_Paired')
    if any(f.startswith('Propermab_') for f in generated_features): available_groups.append('Propermab')   
    
    # Register custom tabular columns
    if custom_feature_groups:
        for grp in custom_feature_groups:
            if grp in generated_features:
                available_groups.append(grp)
                
    # Prune explicitly blacklisted feature groups
    if exclude_groups:
        available_groups = [g for g in available_groups if g not in exclude_groups]
        print(f"🚫 Excluded requested feature groups: {exclude_groups}")
    
    completed_combos = set()
    file_exists = os.path.isfile(checkpoint_csv)
    if file_exists:
        try:
            df_check = pd.read_csv(checkpoint_csv)
            for _, row in df_check.iterrows(): completed_combos.add(f"{row['Subregions']}|{row['Features']}")
            print(f"✅ Found Checkpoint! Resuming search. {len(completed_combos)} combinations already completed.")
        except Exception: file_exists = False

    with open(checkpoint_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists: writer.writerow(['Subregions', 'Features', 'Num_Features', 'Spearman', 'Spearman_Std', 'RMSE', 'RMSE_Std', 'MAE', 'MAE_Std', 'R2', 'R2_Std'])
            
        scorer = {'spearman': make_scorer(custom_spearman), 'r2': 'r2', 'rmse': 'neg_root_mean_squared_error', 'mae': 'neg_mean_absolute_error'}
        cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
        
        all_cols = df.columns.tolist()
        
        valid_feat_combos = []
        for FL in range(1, len(available_groups) + 1):
            for feat_combo in itertools.combinations(available_groups, FL):
                
                # Constraint 1: Limit to one neural network model per permutation
                lm_count = sum(1 for g in feat_combo if 'ESM_' in g or 'AntiBERTy' in g or 'AbLang2_Paired' in g)
                if lm_count > 1:
                    continue
                    
                # Constraint 2: Prevent combining Global and Targeted features of the same base type
                conflict = False
                for base_feat in ['AAC', 'AAindex', 'Georgiev', 'ESM_Small_8M', 'ESM_Big_650M', 'ESM_Big_650M_SVD50']:
                    if base_feat in feat_combo and f"Targeted_{base_feat}" in feat_combo:
                        conflict = True
                        break
                if conflict:
                    continue

                # Constraint 3: Georgiev and AAindex are mutually exclusive
                has_aaindex = any('AAindex' in g for g in feat_combo)
                has_georgiev = any('Georgiev' in g for g in feat_combo)
                if has_aaindex and has_georgiev:
                    continue

                valid_feat_combos.append(feat_combo)
                    
        valid_sub_combos = []
        for L in range(1, len(seq_cols) + 1):
            for sub_combo in itertools.combinations(seq_cols, L):
                if 'scFv' in sub_combo and len(sub_combo) > 1:
                    continue
                valid_sub_combos.append(sub_combo)

        valid_experiments = []
        for sub_combo in valid_sub_combos:
            for feat_combo in valid_feat_combos:
                # Force AbLang2 and Propermab on Paired VH/VL OR scFv
                if 'AbLang2_Paired' in feat_combo or 'Propermab' in feat_combo:
                    has_paired = ('CD3_VH' in sub_combo and 'CD3_VL' in sub_combo)
                    has_scfv = ('scFv' in sub_combo)
                    if not (has_paired or has_scfv): 
                        continue
                
                # Constraint 4: Prevent evaluating targeted features on regions that lack them in the dataset
                has_targeted_feat = any('Targeted_' in g for g in feat_combo)
                if has_targeted_feat:
                    has_valid_targeted_region = False
                    for sub in sub_combo:
                        if any(f.startswith(f"{sub}_Targeted_") for f in generated_features):
                            has_valid_targeted_region = True
                            break
                            
                    if not has_valid_targeted_region:
                        continue 
                        
                valid_experiments.append((sub_combo, feat_combo))
            
        total_runs = len(valid_experiments)
        current_run = 0

        print(f"\n🚀 Starting Exhaustive Search: {total_runs} Total Valid Combinations...")
        for sub_combo, feat_combo in valid_experiments:
            sub_name = " + ".join(sub_combo)
            feat_name = " + ".join(feat_combo)
            combo_id = f"{sub_name}|{feat_name}"
            current_run += 1

            if combo_id in completed_combos: continue

            # Passed custom_feature_groups here
            selected_cols = filter_active_features(all_cols, sub_combo, feat_combo, global_features, custom_feature_groups)
            
            if 'Media_Encoded' in all_cols and 'Media_Encoded' not in selected_cols:
                selected_cols.append('Media_Encoded')
                if 'Media' not in feat_combo:
                    feat_combo = list(feat_combo) + ['Media']

            cols_to_check = [target_col] + selected_cols
            if weight_col and weight_col in df.columns: cols_to_check.append(weight_col)
            model_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols_to_check).reset_index(drop=True)
            
            if len(model_df) < 15 or len(selected_cols) == len(global_features):
                writer.writerow([sub_name, feat_name, len(selected_cols), 0, 0, 0, 0, 0, 0, 0, 0])
                f.flush()
                continue

            X = model_df[selected_cols + ([weight_col] if weight_col else [])]
            y = model_df[target_col]
            
            model = get_model(model_name, len(selected_cols), transform_type, weight_col)
            print(f"[{current_run}/{total_runs}] Testing: [{sub_name}] with [{feat_name}] ({len(selected_cols)} feats)...")
            
            scores = cross_validate(model, X, y, cv=cv, scoring=scorer, n_jobs=-1, error_score=np.nan)
            valid = ~np.isnan(scores['test_r2'])
            
            if valid.sum() == 0: 
                writer.writerow([sub_name, feat_name, len(selected_cols), 0, 0, 0, 0, 0, 0, 0, 0])
            else:
                writer.writerow([
                    sub_name, feat_name, len(selected_cols), 
                    np.mean(scores['test_spearman'][valid]), np.std(scores['test_spearman'][valid]),
                    -np.mean(scores['test_rmse'][valid]), np.std(scores['test_rmse'][valid]), 
                    -np.mean(scores['test_mae'][valid]), np.std(scores['test_mae'][valid]), 
                    np.mean(scores['test_r2'][valid]), np.std(scores['test_r2'][valid])
                ])
            f.flush()
                        
    df_results_raw = pd.read_csv(checkpoint_csv)
    valid_mask = df_results_raw['Features'].apply(lambda x: all(feat in available_groups for feat in str(x).split(" + ")))
    df_results = df_results_raw[valid_mask].sort_values(by="Spearman", ascending=False)
    if df_results.empty:
        print(f"⚠️ No valid completed combinations found for current settings. Skipping plotting.")
        return 
    df_results.to_excel(final_excel, index=False)
    
    # --- Final Model Production ---
    print(f"\n 🌟 Extracting Model Rank #{rank_to_plot + 1} from leaderboard...")
    best_row = df_results.iloc[rank_to_plot]
    best_subs, best_feats = best_row['Subregions'].split(" + "), best_row['Features'].split(" + ")
    
    final_sub_tag = best_row['Subregions'].replace(' + ', '-')
    final_feat_tag = best_row['Features'].replace(' + ', '-')
    global_tag = f"{final_sub_tag}_{final_feat_tag}"
    
    # Passed custom_feature_groups here
    final_cols = filter_active_features(all_cols, best_subs, best_feats, global_features, custom_feature_groups)
    
    cols_to_check = [target_col] + final_cols
    if weight_col and weight_col in df.columns: cols_to_check.append(weight_col)
    
    df_subset_cols = cols_to_check + ([hue_col] if hue_col and hue_col in df.columns else []) + ([oog_col] if oog_col and oog_col in df.columns else [])
    final_df = df[df_subset_cols].replace([np.inf, -np.inf], np.nan).dropna(subset=cols_to_check).reset_index(drop=True)
    
    X_final = final_df[final_cols + ([weight_col] if weight_col else [])]
    y_final = final_df[target_col]
    hue_data_final = final_df[hue_col] if hue_col and hue_col in final_df.columns else None
    oog_data_final = final_df[oog_col] if oog_col and oog_col in final_df.columns else None
    
    model_filename = os.path.join(output_dir, f"Production_{prefix}{model_name}_{target_col}_{global_tag}.joblib")
    
    if os.path.exists(model_filename):
        print(f"\n⚡ Found existing Production Model! Loading '{model_filename}' (Skipping GridSearch)...")
        loaded_package = joblib.load(model_filename)
        if isinstance(loaded_package, dict):
            locked_best_estimator = loaded_package['model']
            trained_features = loaded_package.get('features', final_cols)
            expected_cols = trained_features + ([weight_col] if weight_col else [])
            X_final = X_final[expected_cols]
            final_cols = trained_features
            best_params = loaded_package.get('best_params', None)
        else:
            locked_best_estimator = loaded_package
            best_params = None

    else:
        print(f"\n⚙️ Training Final Production Model...")
        final_model = get_model(model_name, len(final_cols), transform_type, weight_col)
        final_model.fit(X_final, y_final)
        best_params = final_model.best_params_ if hasattr(final_model, 'best_params_') else None
        locked_best_estimator = final_model.best_estimator_ if hasattr(final_model, 'best_estimator_') else final_model    

    plot_best_model_diagnostics(
        X=X_final, 
        y=y_final, 
        subregions_name=best_row['Subregions'],
        features_name=best_row['Features'],
        model_name=model_name, 
        target_col=target_col, 
        output_dir=output_dir, 
        final_estimator=locked_best_estimator,
        best_params=best_params,
        feature_tag=global_tag,
        prefix=prefix,
        hue_data=hue_data_final,
        hue_name=hue_col
    )

    if extended_plots:
        plot_best_model_diagnostics_old(
            X=X_final, y=y_final,
            subregions_name=best_row['Subregions'], features_name=best_row['Features'],
            model_name=model_name, target_col=target_col, output_dir=output_dir,
            final_estimator=locked_best_estimator, best_params=best_params, feature_tag=global_tag,
            prefix=prefix, hue_data=hue_data_final, hue_name=hue_col, threshold=manual_threshold
        ) 
    
    if oog_data_final is not None:
        plot_out_of_group_diagnostics(
            X=X_final,
            y=y_final,
            group_labels=oog_data_final,
            subregions_name=best_row['Subregions'],
            features_name=best_row['Features'],
            model_name=model_name, 
            target_col=target_col, 
            output_dir=output_dir, 
            final_estimator=locked_best_estimator,
            feature_tag=global_tag,
            prefix=prefix,
            split_col_name=oog_col
        )

    generate_shap_analysis(
        model=locked_best_estimator, X=X_final, y=y_final, output_dir=output_dir, feature_names=final_cols, 
        model_name=model_name, target_col=target_col, prefix=prefix, feature_tag=global_tag, aaindex_desc=aaindex_desc
    )

    if not os.path.exists(model_filename):
        print(f"Saving Final Production Model and Feature Metadata...")
        joblib.dump({
            'model': locked_best_estimator,
            'features': final_cols,
            'target': target_col,
            'best_params': best_params 
        }, model_filename)
        print(f"✅ Production package successfully saved to: {model_filename}")
    else:
        print(f"\n✅ Production model already exists on disk. Skipping save.")

def evaluate_single_combination(df, target_col, model_name, output_dir, sub_combo, feat_combo, generated_features, 
                                transform_type=None, weight_col=None, prefix="targeted_", hue_col=None, oog_col=None, aaindex_desc=None,
                                custom_feature_groups=None): 
    """
    Evaluates a specific region and feature combination directly, bypassing the grid search logic entirely.
    
    Args:
        df (pd.DataFrame): Dataset including targets and extracted features.
        target_col (str): Dependent variable to predict.
        model_name (str): Scikit-learn estimator ID (e.g., 'SVR').
        output_dir (str): File destination for results.
        sub_combo (list): Explicit regions to test (e.g., ['CD3_VH']).
        feat_combo (list): Explicit feature groups to test (e.g., ['Targeted_ESM_Big_650M_SVD50']).
        generated_features (list): Master list of all feature names present in df.
    """
    print(f"\n==================================================================")
    print(f"🎯 TARGETED EVALUATION: {model_name} on {target_col}")
    print(f"   Regions: {sub_combo}")
    print(f"   Features: {feat_combo}")
    print(f"==================================================================")
    
    # Validate structural requirements for paired-chain models to prevent silent failures
    if 'AbLang2_Paired' in feat_combo or 'Propermab' in feat_combo:
        has_paired = ('CD3_VH' in sub_combo and 'CD3_VL' in sub_combo)
        has_scfv = ('scFv' in sub_combo)
        if not (has_paired or has_scfv):
            print("\n⚠️ WARNING: Propermab & AbLang2 mathematically require either Paired 'CD3_VH'+'CD3_VL' OR 'scFv'.")
            print("   Your current regions will cause these features to be skipped (0 features loaded).\n")

    # Warn against mixing global and targeted versions of the same space
    for base_feat in ['AAC', 'AAindex', 'Georgiev', 'ESM_Small_8M', 'ESM_Big_650M', 'ESM_Big_650M_SVD50']:
        if base_feat in feat_combo and f"Targeted_{base_feat}" in feat_combo:
            print(f"\n⚠️ WARNING: You are manually mixing Global '{base_feat}' and 'Targeted_{base_feat}'.")
            print("   This causes massive multicollinearity and will likely degrade your model!\n")

    os.makedirs(output_dir, exist_ok=True)
    results_excel = os.path.join(output_dir, f"{prefix}single_eval_results_{model_name}_{target_col}.xlsx")
    
    all_cols = df.columns.tolist()
    global_features = [
        f for f in generated_features 
        if not f.startswith('seq_CD3_')
        and not f.startswith('CD3_')
        and not f.startswith('scFv_')
        and not f.startswith('CQA_')
        and not f.startswith('Propermab_')
        and not f.startswith('Paired_')
        and (custom_feature_groups is None or f not in custom_feature_groups)
    ]
    
    selected_cols = filter_active_features(all_cols, sub_combo, feat_combo, global_features, custom_feature_groups)
    
    if 'Media_Encoded' in all_cols and 'Media_Encoded' not in selected_cols:
        selected_cols.append('Media_Encoded')
        if 'Media' not in feat_combo:
            feat_combo = list(feat_combo) + ['Media']
            print(f"   -> 🧪 Auto-injected Media_Encoded feature into combination.")

    cols_to_check = [target_col] + selected_cols
    if weight_col and weight_col in df.columns: cols_to_check.append(weight_col)
    
    df_subset_cols = cols_to_check + ([hue_col] if hue_col and hue_col in df.columns else []) + ([oog_col] if oog_col and oog_col in df.columns else [])
    model_df = df[df_subset_cols].replace([np.inf, -np.inf], np.nan).dropna(subset=cols_to_check).reset_index(drop=True)
    
    if len(model_df) < 15:
        print("⚠️ Not enough data points to evaluate this combination!")
        return
        
    X = model_df[selected_cols + ([weight_col] if weight_col else [])]
    y = model_df[target_col]
    hue_data = model_df[hue_col] if hue_col and hue_col in model_df.columns else None
    oog_data = model_df[oog_col] if oog_col and oog_col in model_df.columns else None
    
    model = get_model(model_name, len(selected_cols), transform_type, weight_col)
    scorer = {'spearman': make_scorer(custom_spearman), 'r2': 'r2', 'rmse': 'neg_root_mean_squared_error', 'mae': 'neg_mean_absolute_error'}
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    
    scores = cross_validate(model, X, y, cv=cv, scoring=scorer, n_jobs=-1, error_score=np.nan)
    valid = ~np.isnan(scores['test_r2'])
    
    sub_name = " + ".join(sub_combo)
    feat_name = " + ".join(feat_combo)
    
    global_tag = f"{sub_name.replace(' + ', '-')}_{feat_name.replace(' + ', '-')}"
    model_filename = os.path.join(output_dir, f"Production_{prefix}{model_name}_{target_col}_{global_tag}.joblib")
    
    new_result = pd.DataFrame([{
        'Subregions': sub_name, 'Features': feat_name, 'Num_Features': len(selected_cols),
        'Spearman': np.mean(scores['test_spearman'][valid]), 'Spearman_Std': np.std(scores['test_spearman'][valid]),
        'RMSE': -np.mean(scores['test_rmse'][valid]), 'RMSE_Std': np.std(scores['test_rmse'][valid]),
        'MAE': -np.mean(scores['test_mae'][valid]), 'MAE_Std': np.std(scores['test_mae'][valid]),
        'R2': np.mean(scores['test_r2'][valid]), 'R2_Std': np.std(scores['test_r2'][valid])
    }])
    
    if os.path.exists(results_excel):
        existing_df = pd.read_excel(results_excel)
        final_df = pd.concat([existing_df, new_result], ignore_index=True)
    else:
        final_df = new_result
    final_df.to_excel(results_excel, index=False)
    
    if os.path.exists(model_filename):
        print(f"\n⚡ Found existing Production Model! Loading '{model_filename}' (Skipping GridSearch)...")
        loaded_package = joblib.load(model_filename)
        if isinstance(loaded_package, dict):
            locked_best_estimator = loaded_package['model']
            trained_features = loaded_package.get('features', selected_cols)
            expected_cols = trained_features + ([weight_col] if weight_col else [])
            X = X[expected_cols]
            selected_cols = trained_features
            best_params = loaded_package.get('best_params', None)
        else:
            locked_best_estimator = loaded_package
            best_params = None
    else:
        print(f"\n⚙️ Training Final Production Model...")
        model.fit(X, y)
        best_params = model.best_params_ if hasattr(model, 'best_params_') else None
        locked_best_estimator = model.best_estimator_ if hasattr(model, 'best_estimator_') else model
    
    plot_best_model_diagnostics(
        X=X, y=y, subregions_name=sub_name, features_name=feat_name, model_name=model_name, 
        target_col=target_col, output_dir=output_dir, final_estimator=locked_best_estimator,
        best_params=best_params, feature_tag=global_tag, prefix=prefix, hue_data=hue_data, hue_name=hue_col
    )

    if oog_data is not None:
        plot_out_of_group_diagnostics(
            X=X, y=y, group_labels=oog_data, subregions_name=sub_name, features_name=feat_name,
            model_name=model_name, target_col=target_col, output_dir=output_dir, final_estimator=locked_best_estimator,
            feature_tag=global_tag, prefix=prefix, split_col_name=oog_col
        )

    generate_shap_analysis(
        model=locked_best_estimator, X=X, y=y, output_dir=output_dir, feature_names=selected_cols, 
        model_name=model_name, target_col=target_col, prefix=prefix, feature_tag=global_tag, aaindex_desc=aaindex_desc
    )

    if not os.path.exists(model_filename):
        print(f"\n💾 Saving Final Production Model and Feature Metadata...")
        joblib.dump({
            'model': locked_best_estimator,
            'features': selected_cols,
            'target': target_col,
            'best_params': best_params 
        }, model_filename)
        print(f"✅ Production package successfully saved to: {model_filename}")
    else:
        print(f"\n✅ Production model already exists on disk. Skipping save.")
    
    print(f"✅ Targeted Evaluation Complete! Results saved to {results_excel}")

def main():
    """
    Main execution block. Configures the dataset path, target variables, structural indices, 
    and drives the combinatorial search logic.
    """

    # filepath = 'data/tubespin.csv'
    # targets_to_test = {'ELISA_Polyreactivity_Excell': 12.0}#, 'ProA_HMW_ActiPro':20, 'ProA_HMW_Excell': 20.0}

    # filepath = 'data/inhouse_supp_CD3+CD20only_UPDATED.csv'
    # targets_to_test = {
    #     'Purity%': 80.0,
    #     # 'HMW':10.0
    # }

    # filepath = 'data/2+1_Humanized_VH5-VL_anti-CD3_variant_sequece_GA.csv'
    # targets_to_test = {
    #     'Monomer': 80.0,
    #     'HMW%':10.0
    # }

    # filepath = 'data/tubespin_extended.csv'

    # targets_to_test = {
    #     # 'Monomer_combined':80.0,
    #     'HMW_combined':10.0
    # }

    # filepath = 'data/50-50_sequences.csv'
    # targets_to_test = {'50-50_HMW%':10.0,
    #                   # '50-50_HCCF_Titer':750.0,
    #                   #   'Normalized_50-50_HCCF_Titer':0.5
    #                     }

    # filepath = 'data/50-50_sequences_subset.csv'
    # targets_to_test = {'SUBSET_50-50_HMW%':20.0}
    filepath = 'data/tubespin_subset.csv'
    targets_to_test = {'SUBSET_ELISA_Polyreactivity_Excell': 12.0}#, 'ProA_HMW_ActiPro':20, 'ProA_HMW_Excell': 20.0}

    models_to_test = ['SVR', 'PLSRegression']#, 'XGBoost']#['XGBoost']#['ElasticNet', 'SVR','PLSRegression']#, 'SVR'] 
    
    transform_strategy = None
    
    esm_model_selections = ["facebook/esm2_t6_8M_UR50D", "facebook/esm2_t33_650M_UR50D"]
    antibody_format_column = 'Type'
    out_of_group_split_column = 'Manual_Split_Group'
    weighting_column = None
    
    DROP_MONOMER_OUTLIER = False 
    GENERATE_EXTENDED_PLOTS = False
    
    my_target_indices = {
        'CD3_VH': ('Interface_looseness', [34, 36, 38, 42, 43, 44, 45, 46, 49, 60, 61, 62, 63, 96, 101, 102, 103, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117]),
        'CD3_VL': ('Interface_looseness', [30, 33, 34, 35, 36, 37, 39, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 88, 90, 92, 94, 95, 96, 97, 98, 99, 100, 101])
    }
    
    FEATURES_TO_EXCLUDE = ['ESM_Big_650M_SVD50', 'Targeted_ESM_Big_650M_SVD50']
    
    # Simply list the exact column names from your CSV you want to use as combinational features!
    csv_feature_columns = [] 
    USE_MEDIA_FEATURE = False
    media_column = 'Media_Type'
    
    RUN_SINGLE_EVAL = False
    single_eval_regions = ['Global_VL'] 
    single_eval_features = ['ESM_Big_650M_SVD50']

    try:
        df = load_and_clean_data(filepath, remove_outlier=DROP_MONOMER_OUTLIER)
        dataset_name = os.path.splitext(os.path.basename(filepath))[0]
        outlier_tag = "OutliersRemoved" if DROP_MONOMER_OUTLIER else ""

        if USE_MEDIA_FEATURE and media_column in df.columns:
            print(f"\nIntegrating '{media_column}' as a single universal integer feature...")
            media_mapping = dict(enumerate(df[media_column].astype('category').cat.categories))
            df[media_column] = df[media_column].astype('category').cat.codes
            print(f"   -> Added feature: '{media_column}' to ALL models.")
            print(f"   -> Media Dictionary: {media_mapping}")

        # Execute extraction logic. Setting is_inference=False enables cache writing and SVD fitting.
        df_features, seq_cols, generated_features, aaindex_desc,_ = extract_sequence_features(
            df, dataset_name=dataset_name, esm_model_names=esm_model_selections,
            cache_tag=outlier_tag, targeted_pooling_dict=my_target_indices
        )

        for target_column, manual_threshold in targets_to_test.items():
            df_features_run = df_features.copy()
            generated_features_run = list(generated_features)
            
            # Dynamically push tabular CSV columns into the combinatorial feature space
            if csv_feature_columns:
                for col in csv_feature_columns:
                    if col in df_features_run.columns:
                        if col not in generated_features_run:
                            generated_features_run.append(col)
                        print(f"✅ SUCCESS: Added CSV column '{col}' to the combinatorial feature set.")
                    else:
                        print(f"⚠️ WARNING: Requested feature column '{col}' not found in dataset. Skipping!")
                    
            for model_name in models_to_test:
                if USE_MEDIA_FEATURE and media_column in df.columns:
                    if media_column not in generated_features_run:
                        generated_features_run.append(media_column)

                if RUN_SINGLE_EVAL:
                    evaluate_single_combination(
                        df=df_features_run, target_col=target_column, model_name=model_name, output_dir=model_name,
                        sub_combo=single_eval_regions, feat_combo=single_eval_features,
                        generated_features=generated_features_run, transform_type=transform_strategy,
                        weight_col=weighting_column, prefix="targeted_", hue_col=antibody_format_column,
                        oog_col=out_of_group_split_column, aaindex_desc=aaindex_desc,
                        custom_feature_groups=csv_feature_columns 
                    )
                else:
                    print(f"\n==================================================================")
                    print(f"🚀 EXHAUSTIVE SEARCH: GLOBAL SEQUENCES (VH, VL, Fv)")
                    print(f"==================================================================")
                    
                    evaluate_exhaustive_combinations(
                        df_features_run, ['CD3_VH', 'CD3_VL', 'scFv'], generated_features_run, 
                        target_col=target_column, model_name=model_name, output_dir=model_name, 
                        transform_type=transform_strategy, weight_col=weighting_column, prefix="global_",
                        hue_col=antibody_format_column, rank_to_plot=0,# oog_col=out_of_group_split_column,
                        aaindex_desc=aaindex_desc, extended_plots=GENERATE_EXTENDED_PLOTS, manual_threshold=manual_threshold,
                        custom_feature_groups=csv_feature_columns,
                        exclude_groups=FEATURES_TO_EXCLUDE
                    )
            
    except FileNotFoundError:
        print(f"Error: Could not find '{filepath}'.")

if __name__ == "__main__":
    main()
