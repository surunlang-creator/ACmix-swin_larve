"""
ACmix-Swin Phenotype Classifier with WGCNA Integration
======================================================
版本: v2.0 - 支持任意数量分类

改进点:
1. 动态颜色方案生成
2. 完全基于sample_file识别分类
3. 所有可视化函数支持N分类
4. 移除硬编码的类别限制
#如果想实现两个都是qkv输入则：
#1422 out_att = self.norm_attn(self.swin_attn(q + k + v))
#1424 conv_input = (q + k + v).transpose(1, 2)
Date: 2025
Author: Enhanced Version
"""

import os
import math
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings('ignore')

# 尝试导入openpyxl用于写xlsx
try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False
    print("Warning: openpyxl not installed. Will use CSV format instead of XLSX.")

# Set random seed
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


# ============================================================================
# Dynamic Color Schemes (新增)
# ============================================================================

def generate_color_palette(n_colors, palette_name='auto'):
    """
    动态生成n个不同的颜色
    
    Parameters:
    -----------
    n_colors : int
        需要的颜色数量
    palette_name : str
        颜色方案名称 ('auto', 'tab10', 'tab20', 'Set1', 'Set2', 'Set3', 'Paired')
    
    Returns:
    --------
    list of hex colors
    """
    if palette_name == 'auto':
        if n_colors <= 10:
            palette_name = 'tab10'
        elif n_colors <= 20:
            palette_name = 'tab20'
        else:
            palette_name = 'hsv'
    
    if palette_name == 'hsv' or n_colors > 20:
        # 对于更多颜色，使用hsv色轮
        colors = []
        for i in range(n_colors):
            hue = i / n_colors
            rgb = mcolors.hsv_to_rgb([hue, 0.7, 0.9])
            colors.append(mcolors.rgb2hex(rgb))
    else:
        cmap = plt.cm.get_cmap(palette_name)
        if palette_name in ['Set1', 'Set2', 'Set3', 'Paired']:
            n_max = cmap.N
            colors = [mcolors.rgb2hex(cmap(i % n_max)) for i in range(n_colors)]
        else:
            colors = [mcolors.rgb2hex(cmap(i)) for i in range(n_colors)]
    
    return colors


def create_class_color_mapping(class_names):
    """
    为每个类别创建颜色映射
    
    Parameters:
    -----------
    class_names : list
        类别名称列表
    
    Returns:
    --------
    dict : {class_name: hex_color}
    """
    n_classes = len(class_names)
    colors = generate_color_palette(n_classes)
    
    color_mapping = {}
    for i, class_name in enumerate(class_names):
        color_mapping[class_name] = colors[i]
    
    return color_mapping


# 基础配色（用于非类别相关的图表）
SCI_COLORS = {
    'primary': '#E64B35',
    'secondary': '#4DBBD5',
    'tertiary': '#00A087',
    'quaternary': '#3C5488',
    'fifth': '#F39B7F',
    'sixth': '#8491B4',
    'seventh': '#91D1C2',
    'eighth': '#DC0000',
}

# WGCNA-style module colors
WGCNA_COLORS = [
    'turquoise', 'blue', 'brown', 'yellow', 'green', 'red', 'black',
    'pink', 'magenta', 'purple', 'greenyellow', 'tan', 'salmon',
    'cyan', 'midnightblue', 'lightcyan', 'grey60', 'lightgreen',
    'lightyellow', 'royalblue', 'darkred', 'darkgreen', 'darkturquoise',
    'darkgrey', 'orange', 'darkorange', 'white', 'skyblue', 'saddlebrown'
]

def set_sci_style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 12,
        'axes.linewidth': 1.5,
        'axes.edgecolor': 'black',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'legend.frameon': False,
        'legend.fontsize': 10,
    })


# ============================================================================
# WGCNA-Style Analysis Module (Python Implementation)
# ============================================================================

class PythonWGCNA:
    """
    Python implementation of WGCNA-style weighted gene co-expression network analysis.
    Designed to produce outputs compatible with R visualization scripts.
    """
    
    def __init__(self, min_module_size=10, merge_cut_height=0.25,
                 network_type='signed', deep_split=2):
        self.min_module_size = min_module_size
        self.merge_cut_height = merge_cut_height
        self.network_type = network_type
        self.deep_split = deep_split
        
        # Results storage
        self.soft_power = None
        self.adjacency = None
        self.TOM = None
        self.module_colors = None
        self.module_eigengenes = None
        self.gene_module_membership = None
        self.module_trait_cor = None
        self.module_trait_pvalue = None
        self.module_cor_matrix = None
        self.module_p_matrix = None
        
    def pick_soft_threshold(self, expr_data, powers=None, r_sq_cutoff=0.85):
        """
        Pick soft-thresholding power for network construction.
        expr_data: samples x genes matrix
        """
        if powers is None:
            powers = list(range(1, 11)) + list(range(12, 21, 2))
        
        n_samples, n_genes = expr_data.shape
        
        # Subsample genes if too many (for speed)
        if n_genes > 2000:
            gene_vars = np.var(expr_data, axis=0)
            top_idx = np.argsort(gene_vars)[-2000:]
            expr_subset = expr_data[:, top_idx]
        else:
            expr_subset = expr_data
        
        # Calculate correlation matrix
        cor_matrix = np.corrcoef(expr_subset.T)
        
        # For signed network: (1 + cor) / 2
        if self.network_type == 'signed':
            cor_matrix = (1 + cor_matrix) / 2
        else:
            cor_matrix = np.abs(cor_matrix)
        
        # Test different powers
        scale_free_fits = []
        mean_connectivity = []
        
        for power in powers:
            adj = np.power(cor_matrix, power)
            np.fill_diagonal(adj, 0)
            
            # Connectivity
            k = np.sum(adj, axis=1)
            mean_k = np.mean(k)
            mean_connectivity.append(mean_k)
            
            # Scale-free topology fit (R²)
            # log(p(k)) vs log(k)
            k_positive = k[k > 0]
            if len(k_positive) > 10:
                log_k = np.log10(k_positive + 1)
                hist, bin_edges = np.histogram(log_k, bins=10)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                # Remove zeros
                mask = hist > 0
                if np.sum(mask) > 2:
                    x = bin_centers[mask]
                    y = np.log10(hist[mask] + 1)
                    
                    # Linear regression
                    slope, intercept = np.polyfit(x, y, 1)
                    y_pred = slope * x + intercept
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r_sq = 1 - ss_res / (ss_tot + 1e-10)
                    
                    # Penalize positive slope (should be negative for scale-free)
                    if slope > 0:
                        r_sq = r_sq * 0.5
                    
                    scale_free_fits.append(r_sq)
                else:
                    scale_free_fits.append(0)
            else:
                scale_free_fits.append(0)
        
        # Select power
        scale_free_fits = np.array(scale_free_fits)
        
        # Find first power with R² > cutoff
        above_cutoff = np.where(scale_free_fits > r_sq_cutoff)[0]
        if len(above_cutoff) > 0:
            self.soft_power = powers[above_cutoff[0]]
        else:
            # Use power with highest R²
            best_idx = np.argmax(scale_free_fits)
            self.soft_power = powers[best_idx]
        
        print(f"  Selected soft-thresholding power: {self.soft_power}")
        print(f"  Scale-free R²: {scale_free_fits[powers.index(self.soft_power)]:.3f}")
        
        return self.soft_power, powers, scale_free_fits, mean_connectivity
    
    def calculate_adjacency(self, expr_data):
        """Calculate adjacency matrix."""
        print("  Calculating adjacency matrix...")
        
        # Correlation matrix
        cor_matrix = np.corrcoef(expr_data.T)
        
        # Handle NaN
        cor_matrix = np.nan_to_num(cor_matrix, nan=0)
        
        # Transform for signed network
        if self.network_type == 'signed':
            cor_matrix = (1 + cor_matrix) / 2
        else:
            cor_matrix = np.abs(cor_matrix)
        
        # Apply soft threshold
        self.adjacency = np.power(cor_matrix, self.soft_power)
        np.fill_diagonal(self.adjacency, 0)
        
        return self.adjacency
    
    def calculate_TOM(self):
        """Calculate Topological Overlap Matrix."""
        print("  Calculating TOM (Topological Overlap Matrix)...")
        
        adj = self.adjacency
        n = adj.shape[0]
        
        # Connectivity
        k = np.sum(adj, axis=1)
        
        # TOM calculation
        # TOM[i,j] = (sum_u(a_iu * a_uj) + a_ij) / (min(k_i, k_j) + 1 - a_ij)
        
        # Numerator: adjacency product + adjacency
        num = np.dot(adj, adj) + adj
        
        # Denominator: min(k_i, k_j) + 1 - a_ij
        k_matrix = np.minimum.outer(k, k)
        denom = k_matrix + 1 - adj
        
        # TOM
        self.TOM = num / (denom + 1e-10)
        np.fill_diagonal(self.TOM, 1)
        
        # Ensure valid range
        self.TOM = np.clip(self.TOM, 0, 1)
        
        return self.TOM
    
    def detect_modules(self, gene_names=None):
        """Detect modules using hierarchical clustering on TOM dissimilarity."""
        print("  Detecting modules...")
        
        # TOM dissimilarity
        dissTOM = 1 - self.TOM
        
        # Hierarchical clustering
        # Convert to condensed distance matrix
        dissTOM_condensed = squareform(dissTOM, checks=False)
        
        # Average linkage clustering
        linkage_matrix = linkage(dissTOM_condensed, method='average')
        
        # Dynamic tree cut (simplified version)
        # Use different heights to find optimal cut
        n_genes = dissTOM.shape[0]
        
        best_n_modules = 0
        best_labels = None
        best_height = 0
        
        for height in np.linspace(0.1, 0.9, 20):
            labels = fcluster(linkage_matrix, height, criterion='distance')
            
            # Count modules with minimum size
            unique, counts = np.unique(labels, return_counts=True)
            valid_modules = np.sum(counts >= self.min_module_size)
            
            if valid_modules > best_n_modules and valid_modules <= 20:
                best_n_modules = valid_modules
                best_labels = labels
                best_height = height
        
        if best_labels is None:
            # Fallback: use fixed number of clusters
            labels = fcluster(linkage_matrix, t=5, criterion='maxclust')
            best_labels = labels
        
        # Assign colors to modules
        unique_modules = np.unique(best_labels)
        module_to_color = {}
        color_idx = 0
        
        for mod in unique_modules:
            count = np.sum(best_labels == mod)
            if count >= self.min_module_size:
                module_to_color[mod] = WGCNA_COLORS[color_idx % len(WGCNA_COLORS)]
                color_idx += 1
            else:
                module_to_color[mod] = 'grey'
        
        self.module_colors = np.array([module_to_color[m] for m in best_labels])
        self.module_labels = best_labels
        
        # Summary
        unique_colors = [c for c in np.unique(self.module_colors) if c != 'grey']
        print(f"  Detected {len(unique_colors)} modules (excluding grey)")
        for color in unique_colors:
            count = np.sum(self.module_colors == color)
            print(f"    {color}: {count} genes")
        
        grey_count = np.sum(self.module_colors == 'grey')
        if grey_count > 0:
            print(f"    grey (unassigned): {grey_count} genes")
        
        return self.module_colors
    
    def calculate_module_eigengenes(self, expr_data):
        """Calculate module eigengenes (first PC of each module)."""
        print("  Calculating module eigengenes...")
        
        unique_colors = [c for c in np.unique(self.module_colors) if c != 'grey']
        
        eigengenes = {}
        
        for color in unique_colors:
            mask = self.module_colors == color
            module_expr = expr_data[:, mask]
            
            if module_expr.shape[1] > 1:
                # PCA
                pca = PCA(n_components=1)
                me = pca.fit_transform(module_expr).flatten()
                
                # Standardize
                me = (me - np.mean(me)) / (np.std(me) + 1e-10)
                
                # Check sign (should correlate positively with average expression)
                avg_expr = np.mean(module_expr, axis=1)
                if np.corrcoef(me, avg_expr)[0, 1] < 0:
                    me = -me
                
                eigengenes[f'ME{color}'] = me
            else:
                eigengenes[f'ME{color}'] = module_expr.flatten()
        
        self.module_eigengenes = pd.DataFrame(eigengenes)
        
        return self.module_eigengenes
    
    def calculate_module_trait_correlation(self, trait_data):
        """
        Calculate correlation between module eigengenes and traits.
        trait_data: DataFrame with samples as rows and traits as columns
        """
        print("  Calculating module-trait correlations...")
        
        n_modules = len(self.module_eigengenes.columns)
        n_traits = trait_data.shape[1]
        
        cor_matrix = np.zeros((n_modules, n_traits))
        pvalue_matrix = np.zeros((n_modules, n_traits))
        
        for i, me_col in enumerate(self.module_eigengenes.columns):
            me = self.module_eigengenes[me_col].values
            
            for j, trait_col in enumerate(trait_data.columns):
                trait = trait_data[trait_col].values
                
                # Pearson correlation
                cor, pval = pearsonr(me, trait)
                cor_matrix[i, j] = cor
                pvalue_matrix[i, j] = pval
        
        self.module_trait_cor = pd.DataFrame(
            cor_matrix,
            index=self.module_eigengenes.columns,
            columns=trait_data.columns
        )
        
        self.module_trait_pvalue = pd.DataFrame(
            pvalue_matrix,
            index=self.module_eigengenes.columns,
            columns=trait_data.columns
        )
        
        return self.module_trait_cor, self.module_trait_pvalue
    
    def calculate_module_correlation(self):
        """Calculate correlation between module eigengenes."""
        print("  Calculating module-module correlations...")
        
        me_values = self.module_eigengenes.values
        n_modules = me_values.shape[1]
        module_names = list(self.module_eigengenes.columns)
        
        # Remove 'ME' prefix for compatibility with R script
        module_colors = [name.replace('ME', '') for name in module_names]
        
        cor_matrix = np.zeros((n_modules, n_modules))
        pvalue_matrix = np.zeros((n_modules, n_modules))
        
        for i in range(n_modules):
            for j in range(n_modules):
                if i == j:
                    cor_matrix[i, j] = 1.0
                    pvalue_matrix[i, j] = 0.0
                else:
                    cor, pval = pearsonr(me_values[:, i], me_values[:, j])
                    cor_matrix[i, j] = cor
                    pvalue_matrix[i, j] = pval
        
        self.module_cor_matrix = pd.DataFrame(
            cor_matrix,
            index=module_colors,
            columns=module_colors
        )
        
        self.module_p_matrix = pd.DataFrame(
            pvalue_matrix,
            index=module_colors,
            columns=module_colors
        )
        
        return self.module_cor_matrix, self.module_p_matrix
    
    def calculate_gene_significance(self, expr_data, trait_data):
        """Calculate gene significance (correlation with traits)."""
        print("  Calculating gene significance...")
        
        n_genes = expr_data.shape[1]
        n_traits = trait_data.shape[1]
        
        gs_matrix = np.zeros((n_genes, n_traits))
        
        for i in range(n_genes):
            gene_expr = expr_data[:, i]
            for j, trait_col in enumerate(trait_data.columns):
                trait = trait_data[trait_col].values
                cor, _ = pearsonr(gene_expr, trait)
                gs_matrix[i, j] = abs(cor)
        
        self.gene_significance = pd.DataFrame(
            gs_matrix,
            columns=[f'GS_{col}' for col in trait_data.columns]
        )
        
        return self.gene_significance
    
    def calculate_module_membership(self, expr_data):
        """Calculate module membership (correlation with module eigengenes)."""
        print("  Calculating module membership...")
        
        n_genes = expr_data.shape[1]
        n_modules = len(self.module_eigengenes.columns)
        
        mm_matrix = np.zeros((n_genes, n_modules))
        
        for i in range(n_genes):
            gene_expr = expr_data[:, i]
            for j, me_col in enumerate(self.module_eigengenes.columns):
                me = self.module_eigengenes[me_col].values
                cor, _ = pearsonr(gene_expr, me)
                mm_matrix[i, j] = abs(cor)
        
        self.gene_module_membership = pd.DataFrame(
            mm_matrix,
            columns=[f'MM_{col.replace("ME", "")}' for col in self.module_eigengenes.columns]
        )
        
        return self.gene_module_membership
    
    def fit(self, expr_data, trait_data, gene_names=None):
        """
        Run full WGCNA analysis.
        
        Parameters:
        -----------
        expr_data : np.ndarray
            Expression matrix (samples x genes)
        trait_data : pd.DataFrame
            Trait matrix (samples x traits), typically one-hot encoded phenotypes
        gene_names : list
            Gene names
        
        Returns:
        --------
        dict with all results
        """
        print("\n" + "=" * 60)
        print("WGCNA Analysis (Python Implementation)")
        print("=" * 60)
        print(f"  Input: {expr_data.shape[0]} samples x {expr_data.shape[1]} genes")
        print(f"  Traits: {list(trait_data.columns)}")
        
        # Step 1: Pick soft threshold
        self.pick_soft_threshold(expr_data)
        
        # Step 2: Calculate adjacency
        self.calculate_adjacency(expr_data)
        
        # Step 3: Calculate TOM
        self.calculate_TOM()
        
        # Step 4: Detect modules
        self.detect_modules(gene_names)
        
        # Step 5: Calculate module eigengenes
        self.calculate_module_eigengenes(expr_data)
        
        # Step 6: Calculate module-trait correlation
        self.calculate_module_trait_correlation(trait_data)
        
        # Step 7: Calculate module-module correlation
        self.calculate_module_correlation()
        
        # Step 8: Calculate gene significance
        self.calculate_gene_significance(expr_data, trait_data)
        
        # Step 9: Calculate module membership
        self.calculate_module_membership(expr_data)
        
        print("\n  WGCNA analysis complete!")
        
        return {
            'soft_power': self.soft_power,
            'module_colors': self.module_colors,
            'module_eigengenes': self.module_eigengenes,
            'module_trait_cor': self.module_trait_cor,
            'module_trait_pvalue': self.module_trait_pvalue,
            'module_cor_matrix': self.module_cor_matrix,
            'module_p_matrix': self.module_p_matrix,
            'gene_significance': self.gene_significance,
            'gene_module_membership': self.gene_module_membership
        }


# ============================================================================
# Network File Generator (R-Compatible)
# ============================================================================

class NetworkFileGenerator:
    """
    Generate network files compatible with R visualization script (V9.2).
    """
    
    def __init__(self, output_dir='./Input'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def save_dataframe(self, df, filename, index=True):
        """Save DataFrame as xlsx or csv."""
        filepath = os.path.join(self.output_dir, filename)
        
        if HAS_OPENPYXL and filename.endswith('.xlsx'):
            df.to_excel(filepath, index=index)
        else:
            # Fallback to CSV
            csv_path = filepath.replace('.xlsx', '.csv')
            df.to_csv(csv_path, index=index)
            filepath = csv_path
        
        return filepath
    
    def generate_hub_genes(self, gene_importance_df, wgcna_results,
                           gene_names, class_names,
                           n_overall_hub=20, n_phenotype_hub=10):
        """
        Select hub genes based on deep learning importance and WGCNA results.
        """
        print("\n[Hub Gene Selection]")
        
        # Get WGCNA scores
        module_colors = wgcna_results['module_colors']
        gene_significance = wgcna_results['gene_significance']
        module_membership = wgcna_results['gene_module_membership']
        
        # Calculate combined scores
        scores_df = pd.DataFrame({'gene': gene_names})
        
        # Deep learning importance (normalized)
        if 'importance_grad_input' in gene_importance_df.columns:
            dl_importance = gene_importance_df.set_index('gene')['importance_grad_input']
            dl_importance = (dl_importance - dl_importance.min()) / (dl_importance.max() - dl_importance.min() + 1e-8)
            scores_df['dl_score'] = scores_df['gene'].map(dl_importance).fillna(0)
        else:
            scores_df['dl_score'] = 0
        
        # WGCNA scores: max GS * max MM
        max_gs = gene_significance.max(axis=1).values
        max_mm = module_membership.max(axis=1).values
        wgcna_score = max_gs * max_mm
        wgcna_score = (wgcna_score - wgcna_score.min()) / (wgcna_score.max() - wgcna_score.min() + 1e-8)
        scores_df['wgcna_score'] = wgcna_score
        
        # Combined score
        scores_df['combined_score'] = 0.5 * scores_df['dl_score'] + 0.5 * scores_df['wgcna_score']
        
        # Module assignment
        scores_df['module'] = module_colors
        
        # Overall hub genes (top combined score)
        overall_hub = scores_df.nlargest(n_overall_hub, 'combined_score')['gene'].tolist()
        print(f"  Overall hub genes: {len(overall_hub)}")
        
        # Phenotype-specific hub genes
        phenotype_hub = {}
        for i, class_name in enumerate(class_names):
            gs_col = f'GS_{class_name}'
            if gs_col in gene_significance.columns:
                # Score = GS for this class * MM * DL importance
                pheno_score = gene_significance[gs_col].values * max_mm * scores_df['dl_score'].values
            else:
                pheno_score = scores_df['combined_score'].values
            
            scores_df[f'score_{class_name}'] = pheno_score
            top_genes = scores_df.nlargest(n_phenotype_hub, f'score_{class_name}')['gene'].tolist()
            phenotype_hub[class_name] = top_genes
            print(f"  {class_name} hub genes: {len(top_genes)}")
        
        return overall_hub, phenotype_hub, scores_df
    
    def generate_node_edge_layout(self, overall_hub, phenotype_hub,
                                   gene_scores_df, class_names,
                                   wgcna_results, expr_data, sample_groups):
        """
        Generate node.xlsx, edge.xlsx, layout.xlsx for R visualization.
        """
        print("\n[Generating Network Files]")
        
        # Collect all hub genes
        all_hub_genes = set(overall_hub)
        for genes in phenotype_hub.values():
            all_hub_genes.update(genes)
        all_hub_genes = list(all_hub_genes)
        
        print(f"  Total unique hub genes: {len(all_hub_genes)}")
        
        # ========== NODE TABLE ==========
        nodes = []
        node_id = 1
        
        # Gene nodes
        gene_node_ids = {}
        for gene in all_hub_genes:
            # Determine type
            if gene in overall_hub:
                is_overall = True
            else:
                is_overall = False
            
            # Find which phenotype it belongs to
            pheno_type = None
            for class_name, genes in phenotype_hub.items():
                if gene in genes:
                    pheno_type = class_name
                    break
            
            if pheno_type:
                node_type = f'Hub_Group_{pheno_type}'
                shape = 'diamond'
            elif is_overall:
                node_type = 'Hub_Overall'
                shape = 'circle'
            else:
                node_type = 'Hub_Overall'
                shape = 'circle'
            
            nodes.append({
                'name': gene,
                'annotation': node_type,
                'type': node_type,
                'node_id': node_id,
                'shape': shape
            })
            gene_node_ids[gene] = node_id
            node_id += 1
        
        # Phenotype nodes
        group_node_ids = {}
        for class_name in class_names:
            group_name = f'Group_{class_name}'
            nodes.append({
                'name': group_name,
                'annotation': 'Phenotype',
                'type': 'Phenotype',
                'node_id': node_id,
                'shape': 'diamond'
            })
            group_node_ids[class_name] = node_id
            node_id += 1
        
        node_df = pd.DataFrame(nodes)
        
        # ========== EDGE TABLE ==========
        edges = []
        
        # Get gene indices for hub genes
        gene_to_idx = {gene: i for i, gene in enumerate(gene_scores_df['gene'])}
        
        for gene in all_hub_genes:
            if gene not in gene_to_idx:
                continue
            
            gene_idx = gene_to_idx[gene]
            gene_expr = expr_data[:, gene_idx]
            
            # Calculate mean expression per group
            group_means = {}
            for class_name in class_names:
                mask = np.array([sample_groups[i] == class_name for i in range(len(sample_groups))])
                if np.sum(mask) > 0:
                    group_means[class_name] = np.mean(gene_expr[mask])
            
            overall_mean = np.mean(list(group_means.values()))
            overall_std = np.std(list(group_means.values()))
            if overall_std == 0:
                overall_std = 1
            
            # Create edges to each group
            for class_name in class_names:
                if class_name in group_means:
                    weight = (group_means[class_name] - overall_mean) / overall_std
                    edges.append({
                        'from': gene_node_ids[gene],
                        'to': group_node_ids[class_name],
                        'weight': weight,
                        'type': 'Gene-Group'
                    })
        
        edge_df = pd.DataFrame(edges)
        
        # ========== LAYOUT TABLE ==========
        layouts = []
        
        # Layout parameters
        hub_types = node_df[node_df['type'] != 'Phenotype']['type'].unique()
        n_hub_types = len(hub_types)
        y_spacing = 25
        circle_radius = 8
        
        total_height = (n_hub_types - 1) * y_spacing
        y_positions = np.linspace(total_height / 2, -total_height / 2, n_hub_types)
        
        for idx, hub_type in enumerate(hub_types):
            genes_this_type = node_df[node_df['type'] == hub_type]['name'].tolist()
            n_genes = len(genes_this_type)
            
            if n_genes == 0:
                continue
            
            angles = np.linspace(0, 2 * np.pi, n_genes + 1)[:-1]
            center_y = y_positions[idx]
            
            for i, gene in enumerate(genes_this_type):
                layouts.append({
                    'node': gene,
                    'x': -20 + np.cos(angles[i]) * circle_radius,
                    'y': center_y + np.sin(angles[i]) * circle_radius,
                    'annotation': hub_type
                })
        
        # Phenotype nodes layout
        n_groups = len(class_names)
        if n_groups > 0:
            group_y = np.linspace(y_positions[0] - 5, y_positions[-1] + 5, n_groups)
            for i, class_name in enumerate(class_names):
                layouts.append({
                    'node': f'Group_{class_name}',
                    'x': 6,
                    'y': group_y[i],
                    'annotation': 'Phenotype'
                })
        
        layout_df = pd.DataFrame(layouts)
        
        # ========== SAVE FILES ==========
        self.save_dataframe(node_df, 'node.xlsx', index=False)
        self.save_dataframe(edge_df, 'edge.xlsx', index=False)
        self.save_dataframe(layout_df, 'layout.xlsx', index=False)
        
        print(f"  Saved: node.xlsx ({len(node_df)} nodes)")
        print(f"  Saved: edge.xlsx ({len(edge_df)} edges)")
        print(f"  Saved: layout.xlsx ({len(layout_df)} layouts)")
        
        return node_df, edge_df, layout_df
    
    def generate_wgcna_files(self, wgcna_results, class_names):
        """
        Generate WGCNA-related files for R visualization.
        """
        print("\n[Generating WGCNA Files]")
        
        # Module correlation matrix
        module_cor = wgcna_results['module_cor_matrix']
        module_cor.to_csv(os.path.join(self.output_dir, 'module_correlation_matrix.csv'))
        print(f"  Saved: module_correlation_matrix.csv")
        
        # Module P-value matrix
        module_p = wgcna_results['module_p_matrix']
        module_p.to_csv(os.path.join(self.output_dir, 'module_pvalue_matrix.csv'))
        print(f"  Saved: module_pvalue_matrix.csv")
        
        # Module-trait correlation
        trait_cor = wgcna_results['module_trait_cor']
        trait_cor.to_csv(os.path.join(self.output_dir, 'module_trait_correlation.csv'))
        print(f"  Saved: module_trait_correlation.csv")
        
        # Module-trait P-value
        trait_p = wgcna_results['module_trait_pvalue']
        trait_p.to_csv(os.path.join(self.output_dir, 'module_trait_pvalue.csv'))
        print(f"  Saved: module_trait_pvalue.csv")
        
        # Groups info
        groups_df = pd.DataFrame({'groups': class_names})
        groups_df.to_csv(os.path.join(self.output_dir, 'groups.csv'), index=False)
        print(f"  Saved: groups.csv")
        
        # Save full results as pickle
        with open(os.path.join(self.output_dir, 'WGCNA_results.pkl'), 'wb') as f:
            pickle.dump(wgcna_results, f)
        print(f"  Saved: WGCNA_results.pkl")
    
    def generate_metabolite_types(self, overall_hub, phenotype_hub):
        """Generate metabolite_types.xlsx (gene types for visualization)."""
        
        all_genes = set(overall_hub)
        for genes in phenotype_hub.values():
            all_genes.update(genes)
        
        types_data = []
        for gene in all_genes:
            # Check phenotype-specific first
            gene_type = None
            for class_name, genes in phenotype_hub.items():
                if gene in genes:
                    gene_type = f'Hub_Group_{class_name}'
                    break
            
            if gene_type is None and gene in overall_hub:
                gene_type = 'Hub_Overall'
            
            if gene_type is None:
                gene_type = 'Hub_Overall'
            
            types_data.append({
                'metabolite': gene,
                'type': gene_type
            })
        
        types_df = pd.DataFrame(types_data)
        self.save_dataframe(types_df, 'metabolite_types.xlsx', index=False)
        print(f"  Saved: metabolite_types.xlsx")
        
        return types_df


# ============================================================================
# Feature Selection
# ============================================================================

class FeatureSelector:
    def __init__(self, n_features=100, method='knn_combined'):
        self.n_features = n_features
        self.method = method
        self.selected_indices = None
        self.selected_names = None
        self.scaler = StandardScaler()
        self.selection_scores = None
        
    def _knn_importance(self, X, y, k=5):
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import cross_val_score
        
        n_features = X.shape[1]
        importance_scores = np.zeros(n_features)
        
        for i in range(n_features):
            X_single = X[:, i:i+1]
            knn = KNeighborsClassifier(n_neighbors=min(k, len(y)-1))
            try:
                scores = cross_val_score(knn, X_single, y, cv=min(5, len(y)), scoring='accuracy')
                importance_scores[i] = scores.mean()
            except:
                importance_scores[i] = 0
        
        return importance_scores
        
    def fit(self, X, y, feature_names=None):
        print(f"\nFeature Selection (Original: {X.shape[1]} -> Target: {self.n_features})")
        
        X_scaled = self.scaler.fit_transform(X)
        
        if self.method == 'knn_combined':
            print("  Using KNN + Mutual Information + F-test combination...")
            
            knn_scores = self._knn_importance(X_scaled, y, k=3)
            knn_scores = (knn_scores - knn_scores.min()) / (knn_scores.max() - knn_scores.min() + 1e-8)
            
            mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
            mi_scores = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min() + 1e-8)
            
            f_scores, _ = f_classif(X_scaled, y)
            f_scores = np.nan_to_num(f_scores, nan=0)
            f_scores = (f_scores - f_scores.min()) / (f_scores.max() - f_scores.min() + 1e-8)
            
            combined_scores = 0.4 * knn_scores + 0.35 * mi_scores + 0.25 * f_scores
            top_indices = np.argsort(combined_scores)[-self.n_features:]
            
            self.selection_scores = {
                'knn': knn_scores,
                'mutual_info': mi_scores,
                'f_test': f_scores,
                'combined': combined_scores
            }
        else:
            mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
            top_indices = np.argsort(mi_scores)[-self.n_features:]
        
        self.selected_indices = np.sort(top_indices)
        if feature_names is not None:
            self.selected_names = [feature_names[i] for i in self.selected_indices]
        
        print(f"  Final selection: {len(self.selected_indices)} features")
        return self
    
    def transform(self, X):
        return X[:, self.selected_indices]
    
    def fit_transform(self, X, y, feature_names=None):
        self.fit(X, y, feature_names)
        return self.transform(X)


# ============================================================================
# WGAN-GP
# ============================================================================

class Generator(nn.Module):
    def __init__(self, noise_dim, label_dim, hidden_dim, output_dim):
        super().__init__()
        self.label_emb = nn.Embedding(label_dim, label_dim)
        self.model = nn.Sequential(
            nn.Linear(noise_dim + label_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        label_emb = self.label_emb(labels)
        x = torch.cat([noise, label_emb], dim=1)
        return self.model(x)


class Critic(nn.Module):
    def __init__(self, input_dim, label_dim, hidden_dim):
        super().__init__()
        self.label_emb = nn.Embedding(label_dim, label_dim)
        self.model = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(input_dim + label_dim, hidden_dim)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x, labels):
        label_emb = self.label_emb(labels)
        x = torch.cat([x, label_emb], dim=1)
        return self.model(x)


class WGAN_GP:
    def __init__(self, input_dim, num_classes, noise_dim=64, hidden_dim=128,
                 lambda_gp=10, n_critic=5, device='cpu'):
        self.device = device
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.lambda_gp = lambda_gp
        self.n_critic = n_critic
        
        self.generator = Generator(noise_dim, num_classes, hidden_dim, input_dim).to(device)
        self.critic = Critic(input_dim, num_classes, hidden_dim).to(device)
        
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-4, betas=(0.0, 0.9))
        self.c_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4, betas=(0.0, 0.9))
        
        self.history = {'g_loss': [], 'c_loss': [], 'w_distance': [], 'gp': [],
                       'real_score': [], 'fake_score': []}
        
    def _gradient_penalty(self, real_samples, fake_samples, labels):
        batch_size = real_samples.size(0)
        alpha = torch.rand(batch_size, 1).to(self.device)
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        d_interpolates = self.critic(interpolates, labels)
        gradients = torch.autograd.grad(
            outputs=d_interpolates, inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True, retain_graph=True
        )[0]
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    def train(self, real_data, labels, epochs=800, batch_size=8,
              early_stop_patience=150, min_epochs=200):
        dataset = TensorDataset(torch.FloatTensor(real_data), torch.LongTensor(labels))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        
        print(f"\nTraining WGAN-GP (max_epochs={epochs})")
        
        for epoch in range(epochs):
            epoch_g_loss = 0
            epoch_c_loss = 0
            epoch_w_dist = 0
            epoch_gp = 0
            epoch_real_score = 0
            epoch_fake_score = 0
            n_batches = 0
            
            for real_samples, real_labels in dataloader:
                batch_size_actual = real_samples.size(0)
                real_samples = real_samples.to(self.device)
                real_labels = real_labels.to(self.device)
                
                # Train Critic
                for _ in range(self.n_critic):
                    self.c_optimizer.zero_grad()
                    
                    # Real samples
                    real_score = self.critic(real_samples, real_labels)
                    
                    # Fake samples
                    noise = torch.randn(batch_size_actual, self.noise_dim).to(self.device)
                    fake_samples = self.generator(noise, real_labels)
                    fake_score = self.critic(fake_samples.detach(), real_labels)
                    
                    # Gradient penalty
                    gp = self._gradient_penalty(real_samples, fake_samples.detach(), real_labels)
                    
                    # Critic loss
                    c_loss = -real_score.mean() + fake_score.mean() + self.lambda_gp * gp
                    c_loss.backward()
                    self.c_optimizer.step()
                    
                    # Record scores (only from last critic update)
                    epoch_real_score += real_score.mean().item()
                    epoch_fake_score += fake_score.mean().item()
                    epoch_gp += gp.item()
                
                # Train Generator
                self.g_optimizer.zero_grad()
                noise = torch.randn(batch_size_actual, self.noise_dim).to(self.device)
                fake_samples = self.generator(noise, real_labels)
                fake_score_g = self.critic(fake_samples, real_labels)
                g_loss = -fake_score_g.mean()
                g_loss.backward()
                self.g_optimizer.step()
                
                # Record losses
                epoch_g_loss += g_loss.item()
                epoch_c_loss += c_loss.item()
                
                # Wasserstein distance approximation
                with torch.no_grad():
                    real_score_wd = self.critic(real_samples, real_labels).mean().item()
                    fake_score_wd = self.critic(fake_samples.detach(), real_labels).mean().item()
                    w_dist = real_score_wd - fake_score_wd
                    epoch_w_dist += w_dist
                
                n_batches += 1
            
            # Average over batches
            avg_g_loss = epoch_g_loss / n_batches
            avg_c_loss = epoch_c_loss / n_batches
            avg_w_dist = epoch_w_dist / n_batches
            avg_gp = epoch_gp / (n_batches * self.n_critic)
            avg_real_score = epoch_real_score / (n_batches * self.n_critic)
            avg_fake_score = epoch_fake_score / (n_batches * self.n_critic)
            
            # Store history
            self.history['g_loss'].append(avg_g_loss)
            self.history['c_loss'].append(avg_c_loss)
            self.history['w_distance'].append(avg_w_dist)
            self.history['gp'].append(avg_gp)
            self.history['real_score'].append(avg_real_score)
            self.history['fake_score'].append(avg_fake_score)
            
            if (epoch + 1) % 200 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: G_Loss={avg_g_loss:.4f}, C_Loss={avg_c_loss:.4f}, "
                      f"W_Dist={avg_w_dist:.4f}, GP={avg_gp:.4f}")
        
        print(f"\nWGAN-GP Training Complete!")
        print(f"  Final Wasserstein Distance: {self.history['w_distance'][-1]:.4f}")
        print(f"  Final Gradient Penalty: {self.history['gp'][-1]:.4f}")
        
        return self.history
    
    def generate(self, num_samples_per_class):
        self.generator.eval()
        generated_data, generated_labels = [], []
        with torch.no_grad():
            for class_idx in range(self.num_classes):
                noise = torch.randn(num_samples_per_class, self.noise_dim).to(self.device)
                labels = torch.full((num_samples_per_class,), class_idx, dtype=torch.long).to(self.device)
                fake_samples = self.generator(noise, labels)
                generated_data.append(fake_samples.cpu().numpy())
                generated_labels.extend([class_idx] * num_samples_per_class)
        return np.vstack(generated_data), np.array(generated_labels)


# ============================================================================
# Data Augmenter
# ============================================================================

class ImprovedAugmenter:
    def __init__(self, num_classes, device='cpu'):
        self.num_classes = num_classes
        self.device = device
        self.scaler = StandardScaler()
        self.gan = None
        self.gan_history = None
        
    def fit(self, X, y, gan_epochs=800, hidden_dim=128):
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = np.clip(X_scaled, -3, 3) / 3
        
        self.gan = WGAN_GP(
            input_dim=X.shape[1], num_classes=self.num_classes,
            noise_dim=64, hidden_dim=hidden_dim, device=self.device
        )
        self.gan_history = self.gan.train(X_scaled, y, epochs=gan_epochs, batch_size=min(8, len(y)))
        return self.gan_history
    
    def _filter_quality(self, generated, original, threshold=2.0):
        """Filter generated samples by distance to original data."""
        distances = cdist(generated, original).min(axis=1)
        median_dist = np.median(distances)
        mask = distances < threshold * median_dist
        return generated[mask], mask
    
    def augment(self, X, y, samples_per_class=50,
                gan_ratio=0.30, noise_ratio=0.35, mixup_ratio=0.20, smote_ratio=0.15):
        """Augment data with multiple methods and return statistics."""
        X_scaled = self.scaler.transform(X)
        X_scaled = np.clip(X_scaled, -3, 3) / 3
        
        augmented_X = []
        augmented_y = []
        aug_stats = {
            'original': len(y),
            'gan': 0,
            'gan_filtered': 0,
            'noise': 0,
            'mixup': 0,
            'smote': 0
        }
        
        for class_idx in range(self.num_classes):
            class_mask = y == class_idx
            class_samples = X_scaled[class_mask]
            n_original = len(class_samples)
            n_needed = samples_per_class - n_original
            
            if n_needed <= 0:
                augmented_X.append(class_samples)
                augmented_y.extend([class_idx] * n_original)
                continue
            
            # Add original samples
            augmented_X.append(class_samples)
            augmented_y.extend([class_idx] * n_original)
            
            feature_std = np.std(class_samples, axis=0) + 1e-8
            
            # 1. GAN samples
            n_gan = int(n_needed * gan_ratio)
            if n_gan > 0 and self.gan is not None:
                gan_samples_raw, _ = self.gan.generate(int(n_gan * 1.5))
                start_idx = class_idx * int(n_gan * 1.5)
                end_idx = start_idx + int(n_gan * 1.5)
                if end_idx <= len(gan_samples_raw):
                    gan_class_samples = gan_samples_raw[start_idx:end_idx]
                else:
                    gan_class_samples = gan_samples_raw[:int(n_gan * 1.5)]
                
                # Filter low quality samples
                gan_filtered, mask = self._filter_quality(gan_class_samples, class_samples, threshold=2.5)
                aug_stats['gan_filtered'] += int(n_gan * 1.5) - len(gan_filtered)
                
                gan_samples = gan_filtered[:n_gan]
                if len(gan_samples) > 0:
                    augmented_X.append(gan_samples)
                    augmented_y.extend([class_idx] * len(gan_samples))
                    aug_stats['gan'] += len(gan_samples)
                    n_needed -= len(gan_samples)
            
            # Recalculate remaining needed
            n_remaining = samples_per_class - len([yy for yy in augmented_y if yy == class_idx])
            
            # 2. SMOTE samples
            n_smote = int(n_remaining * smote_ratio / (1 - gan_ratio)) if n_remaining > 0 else 0
            if n_smote > 0 and n_original > 1:
                smote_samples = []
                for _ in range(n_smote):
                    idx = np.random.randint(0, n_original)
                    other_idx = np.random.randint(0, n_original)
                    while other_idx == idx and n_original > 1:
                        other_idx = np.random.randint(0, n_original)
                    alpha = np.random.uniform(0.3, 0.7)
                    smote_samples.append(alpha * class_samples[idx] + (1-alpha) * class_samples[other_idx])
                augmented_X.append(np.array(smote_samples))
                augmented_y.extend([class_idx] * n_smote)
                aug_stats['smote'] += n_smote
            
            # 3. Gaussian noise samples
            n_noise = int(n_remaining * noise_ratio / (1 - gan_ratio)) if n_remaining > 0 else 0
            if n_noise > 0:
                noise_samples = []
                for _ in range(n_noise):
                    idx = np.random.randint(0, n_original)
                    noise_strength = np.random.uniform(0.05, 0.15)
                    noise = np.random.normal(0, noise_strength * feature_std)
                    noise_samples.append(np.clip(class_samples[idx] + noise, -1, 1))
                augmented_X.append(np.array(noise_samples))
                augmented_y.extend([class_idx] * n_noise)
                aug_stats['noise'] += n_noise
            
            # 4. Mixup samples
            n_mixup = n_remaining - n_smote - n_noise
            if n_mixup > 0:
                mixup_samples = []
                for _ in range(n_mixup):
                    idx1, idx2 = np.random.choice(n_original, 2, replace=True)
                    lam = np.random.beta(0.4, 0.4)
                    mixup_samples.append(lam * class_samples[idx1] + (1 - lam) * class_samples[idx2])
                augmented_X.append(np.array(mixup_samples))
                augmented_y.extend([class_idx] * n_mixup)
                aug_stats['mixup'] += n_mixup
        
        X_final = np.vstack(augmented_X)
        y_final = np.array(augmented_y)
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(y_final))
        X_final = X_final[shuffle_idx]
        y_final = y_final[shuffle_idx]
        
        # Rescale
        X_final = X_final * 3
        X_final = self.scaler.inverse_transform(X_final)
        X_final = self.scaler.fit_transform(X_final)
        
        print(f"\nAugmentation Statistics:")
        print(f"  Original samples: {aug_stats['original']}")
        print(f"  GAN generated: {aug_stats['gan']} (filtered: {aug_stats['gan_filtered']})")
        print(f"  SMOTE: {aug_stats['smote']}")
        print(f"  Gaussian noise: {aug_stats['noise']}")
        print(f"  Mixup: {aug_stats['mixup']}")
        print(f"  Total: {len(y_final)}")
        
        return X_final, y_final, aug_stats


# ============================================================================
# ACmix-Swin Classifier
# ============================================================================

class SwinWindowAttention1D(nn.Module):
    def __init__(self, dim, window_size, num_heads, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.relative_position_bias_table = nn.Parameter(torch.zeros(2 * window_size - 1, num_heads))
        coords = torch.arange(window_size)
        relative_coords = coords[:, None] - coords[None, :] + window_size - 1
        self.register_buffer("relative_position_index", relative_coords)
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        
    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(self.window_size, self.window_size, -1).permute(2, 0, 1)
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class ACmixSwin1D(nn.Module):
    def __init__(self, in_planes, out_planes, num_heads=4, window_size=7, dropout=0.1):
        super().__init__()
        self.rate1 = nn.Parameter(torch.tensor(0.5))
        self.rate2 = nn.Parameter(torch.tensor(0.5))
        
        self.conv1 = nn.Linear(in_planes, out_planes)
        self.conv2 = nn.Linear(in_planes, out_planes)
        self.conv3 = nn.Linear(in_planes, out_planes)
        
        self.swin_attn = SwinWindowAttention1D(out_planes, window_size, num_heads, dropout, dropout)
        self.norm_attn = nn.LayerNorm(out_planes)
        
        self.depth_conv = nn.Conv1d(out_planes, out_planes, 3, padding=1, groups=out_planes)
        self.point_conv = nn.Conv1d(out_planes, out_planes, 1)
        self.conv_bn = nn.BatchNorm1d(out_planes)
        self.norm_conv = nn.LayerNorm(out_planes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)
        #out_att = self.norm_attn(self.swin_attn(q + k))
        out_att = self.norm_attn(self.swin_attn(q + k + v))
        #conv_input = v.transpose(1, 2)
        conv_input = (q + k + v).transpose(1, 2)
        out_conv = self.conv_bn(self.point_conv(self.depth_conv(conv_input)))
        out_conv = self.norm_conv(F.gelu(out_conv).transpose(1, 2))
        
        return self.dropout(self.rate1 * out_att + self.rate2 * out_conv).squeeze(1)
    
    def get_fusion_weights(self):
        return {'attention': self.rate1.item(), 'convolution': self.rate2.item()}


class ACmixSwinClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=3, embed_dim=64, num_heads=4,
                 window_size=7, dropout=0.3):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = window_size
        
        self.patch_embed = nn.Sequential(
            nn.Linear(input_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(embed_dim * 2, embed_dim * window_size),
        )
        
        self.acmix = ACmixSwin1D(embed_dim, embed_dim, num_heads, window_size, dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, num_classes))
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).view(B, self.seq_len, self.embed_dim)
        x = x + self.acmix(x).view(B, self.seq_len, self.embed_dim)
        x = self.norm(x).transpose(1, 2)
        x = self.avgpool(x).squeeze(-1)
        return self.classifier(x)
    
    def get_acmix_weights(self):
        return {'ACmix_Layer_1': self.acmix.get_fusion_weights()}


# ============================================================================
# Trainer
# ============================================================================

class Trainer:
    def __init__(self, model, device, learning_rate=1e-4, weight_decay=1e-3,
                 warmup_epochs=10, label_smoothing=0.05, mixup_alpha=0.0):
        self.model = model.to(device)
        self.device = device
        self.warmup_epochs = warmup_epochs
        self.mixup_alpha = mixup_alpha
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.base_lr = learning_rate
        
    def _get_lr(self, epoch, total_epochs):
        if epoch < self.warmup_epochs:
            return self.base_lr * (epoch + 1) / self.warmup_epochs
        progress = (epoch - self.warmup_epochs) / (total_epochs - self.warmup_epochs)
        return self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    
    def _mixup_data(self, x, y):
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def _mixup_criterion(self, pred, y_a, y_b, lam):
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)
    
    def train_epoch(self, train_loader, epoch, total_epochs, use_mixup=False):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        
        lr = self._get_lr(epoch, total_epochs)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            self.optimizer.zero_grad()
            
            if use_mixup and self.mixup_alpha > 0:
                X_mixed, y_a, y_b, lam = self._mixup_data(X_batch, y_batch)
                outputs = self.model(X_mixed)
                loss = self._mixup_criterion(outputs, y_a, y_b, lam)
                
                # 使用原始数据计算准确率
                with torch.no_grad():
                    outputs_clean = self.model(X_batch)
                    _, predicted = outputs_clean.max(1)
                    correct += predicted.eq(y_batch).sum().item()
            else:
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                _, predicted = outputs.max(1)
                correct += predicted.eq(y_batch).sum().item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total += y_batch.size(0)
        
        return total_loss / len(train_loader), 100. * correct / total, lr
    
    def evaluate(self, val_loader):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        all_preds, all_labels, all_probs = [], [], []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                
                total_loss += loss.item()
                probs = F.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                correct += predicted.eq(y_batch).sum().item()
                total += y_batch.size(0)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return (total_loss / len(val_loader), 100. * correct / total,
                np.array(all_preds), np.array(all_labels), np.array(all_probs))


def train_classifier(X, y, num_classes, device,
                     embed_dim=64, num_heads=4, window_size=7, dropout=0.3,
                     epochs=300, batch_size=16, learning_rate=1e-4,
                     weight_decay=1e-3, patience=50,
                     use_mixup=False, mixup_alpha=0.2, label_smoothing=0.05):
    """Train ACmix-Swin classifier with full parameters."""
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    print(f"\nData Split:")
    print(f"  Training set: {len(y_train)} samples")
    print(f"  Test set: {len(y_test)} samples")
    
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
                              batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
                             batch_size=batch_size)
    
    model = ACmixSwinClassifier(
        input_dim=X.shape[1],
        num_classes=num_classes,
        embed_dim=embed_dim,
        num_heads=num_heads,
        window_size=window_size,
        dropout=dropout
    ).to(device)
    
    print(f"\nModel Configuration:")
    print(f"  embed_dim={embed_dim}, num_heads={num_heads}, window_size={window_size}")
    print(f"  dropout={dropout}, lr={learning_rate}, weight_decay={weight_decay}")
    print(f"  use_mixup={use_mixup}, mixup_alpha={mixup_alpha}, label_smoothing={label_smoothing}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = Trainer(
        model, device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_epochs=10,
        label_smoothing=label_smoothing,
        mixup_alpha=mixup_alpha if use_mixup else 0.0
    )
    
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'lr': []}
    best_loss, patience_counter, best_state = float('inf'), 0, None
    
    print(f"\nTraining (epochs={epochs}, patience={patience}, mixup={use_mixup})...")
    
    for epoch in range(epochs):
        train_loss, train_acc, lr = trainer.train_epoch(train_loader, epoch, epochs, use_mixup=use_mixup)
        test_loss, test_acc, _, _, _ = trainer.evaluate(test_loader)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['lr'].append(lr)
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs} (lr={lr:.2e})")
            print(f"    Train: Loss={train_loss:.4f}, Acc={train_acc:.1f}%")
            print(f"    Test:  Loss={test_loss:.4f}, Acc={test_acc:.1f}%")
            
            gap = train_acc - test_acc
            if gap > 15:
                print(f"    Warning: Overfitting - gap = {gap:.1f}%")
        
        if test_loss < best_loss - 0.001:
            best_loss = test_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stop @ epoch {epoch+1}")
                break
    
    if best_state:
        model.load_state_dict(best_state)
    
    _, test_acc, all_preds, all_labels, all_probs = trainer.evaluate(test_loader)
    
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    
    # Print ACmix weights
    acmix_weights = model.get_acmix_weights()
    print("\nACmix Fusion Weights:")
    for layer_name, weights in acmix_weights.items():
        print(f"  {layer_name}: Attention={weights['attention']:.3f}, Convolution={weights['convolution']:.3f}")
    
    return model, history, all_preds, all_labels, all_probs, X_test, y_test


# ============================================================================
# Visualization Functions (支持任意分类数)
# ============================================================================

def plot_training_curves(history, save_path):
    set_sci_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], color=SCI_COLORS['primary'],
             linewidth=2.5, label='Train Loss')
    ax1.plot(epochs, history['test_loss'], color=SCI_COLORS['secondary'],
             linewidth=2.5, label='Test Loss')
    
    best_epoch = np.argmin(history['test_loss']) + 1
    best_loss = min(history['test_loss'])
    ax1.axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.5)
    ax1.scatter([best_epoch], [best_loss], color=SCI_COLORS['eighth'], s=100, zorder=5,
               label=f'Best: {best_loss:.4f} (epoch {best_epoch})')
    
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Loss', fontweight='bold')
    ax1.set_title('Training and Test Loss', fontweight='bold', pad=15)
    ax1.legend(frameon=False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['train_acc'], color=SCI_COLORS['primary'],
             linewidth=2.5, label='Train Accuracy')
    ax2.plot(epochs, history['test_acc'], color=SCI_COLORS['secondary'],
             linewidth=2.5, label='Test Accuracy')
    
    best_acc_epoch = np.argmax(history['test_acc']) + 1
    best_acc = max(history['test_acc'])
    ax2.axvline(x=best_acc_epoch, color='gray', linestyle='--', alpha=0.5)
    ax2.scatter([best_acc_epoch], [best_acc], color=SCI_COLORS['eighth'], s=100, zorder=5,
               label=f'Best: {best_acc:.1f}% (epoch {best_acc_epoch})')
    
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontweight='bold')
    ax2.set_title('Training and Test Accuracy', fontweight='bold', pad=15)
    ax2.legend(frameon=False, loc='lower right')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])
    
    ax3 = axes[1, 0]
    train_test_gap = np.array(history['train_acc']) - np.array(history['test_acc'])
    ax3.fill_between(epochs, 0, train_test_gap, alpha=0.3, color=SCI_COLORS['fifth'])
    ax3.plot(epochs, train_test_gap, color=SCI_COLORS['fifth'], linewidth=2.5)
    ax3.axhline(y=10, color=SCI_COLORS['primary'], linestyle='--', linewidth=1.5,
                label='Warning (10%)')
    ax3.axhline(y=20, color=SCI_COLORS['eighth'], linestyle='--', linewidth=1.5,
                label='Critical (20%)')
    ax3.set_xlabel('Epoch', fontweight='bold')
    ax3.set_ylabel('Train - Test Accuracy (%)', fontweight='bold')
    ax3.set_title('Overfitting Indicator', fontweight='bold', pad=15)
    ax3.legend(frameon=False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    ax4.plot(epochs, history['lr'], color=SCI_COLORS['sixth'], linewidth=2.5)
    ax4.set_xlabel('Epoch', fontweight='bold')
    ax4.set_ylabel('Learning Rate', fontweight='bold')
    ax4.set_title('Learning Rate Schedule (Warmup + Cosine)', fontweight='bold', pad=15)
    ax4.set_yscale('log')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_wgan_training(gan_history, save_path):
    set_sci_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, len(gan_history['g_loss']) + 1)
    
    ax1 = axes[0, 0]
    ax1.plot(epochs, gan_history['g_loss'], color=SCI_COLORS['primary'],
             linewidth=2.5, label='Generator Loss')
    ax1.plot(epochs, gan_history['c_loss'], color=SCI_COLORS['secondary'],
             linewidth=2.5, label='Critic Loss')
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Loss', fontweight='bold')
    ax1.set_title('WGAN-GP Losses', fontweight='bold', pad=15)
    ax1.legend(frameon=False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    if 'w_distance' in gan_history and len(gan_history['w_distance']) > 0:
        ax2.plot(epochs, gan_history['w_distance'], color=SCI_COLORS['tertiary'], linewidth=2.5)
        if len(epochs) > 20:
            w_smooth = pd.Series(gan_history['w_distance']).rolling(20).mean()
            ax2.plot(epochs, w_smooth, color=SCI_COLORS['quaternary'], linewidth=2,
                    linestyle='--', label='Moving Avg (20)')
            ax2.legend(frameon=False)
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Wasserstein Distance', fontweight='bold')
    ax2.set_title('Wasserstein Distance', fontweight='bold', pad=15)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    if 'real_score' in gan_history and len(gan_history['real_score']) > 0:
        ax3.plot(epochs, gan_history['real_score'], color=SCI_COLORS['tertiary'],
                 linewidth=2.5, label='Real Score')
        ax3.plot(epochs, gan_history['fake_score'], color=SCI_COLORS['fifth'],
                 linewidth=2.5, label='Fake Score')
        ax3.legend(frameon=False)
    ax3.set_xlabel('Epoch', fontweight='bold')
    ax3.set_ylabel('Critic Score', fontweight='bold')
    ax3.set_title('Critic Scores', fontweight='bold', pad=15)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    if 'gp' in gan_history and len(gan_history['gp']) > 0:
        ax4.plot(epochs, gan_history['gp'], color=SCI_COLORS['sixth'], linewidth=2.5)
    ax4.set_xlabel('Epoch', fontweight='bold')
    ax4.set_ylabel('Gradient Penalty', fontweight='bold')
    ax4.set_title('Gradient Penalty', fontweight='bold', pad=15)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """支持任意分类数的混淆矩阵"""
    set_sci_style()
    
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # 动态调整图片大小
    n_classes = len(class_names)
    figsize = (max(8, n_classes * 1.2), max(7, n_classes * 1.0))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#FFFFFF', '#DEEBF7', '#9ECAE1', '#4292C6', '#08519C']
    cmap = LinearSegmentedColormap.from_list('blues', colors)
    
    im = ax.imshow(cm_normalized, cmap=cmap, aspect='auto', vmin=0, vmax=100)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Percentage (%)', fontweight='bold')
    
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, fontweight='bold', rotation=45, ha='right')
    ax.set_yticklabels(class_names, fontweight='bold')
    
    # 动态调整字体大小
    fontsize = max(8, min(12, 200 // n_classes))
    
    for i in range(n_classes):
        for j in range(n_classes):
            color = 'white' if cm_normalized[i, j] > 50 else 'black'
            ax.text(j, i, f'{cm_normalized[i,j]:.1f}%\n(n={cm[i,j]})',
                   ha='center', va='center', color=color, fontsize=fontsize, fontweight='bold')
    
    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_title('Confusion Matrix', fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_acmix_weights(model, save_path):
    set_sci_style()
    
    weights = model.get_acmix_weights()
    if not weights:
        print("  Warning: No ACmix weights to plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    layers = list(weights.keys())
    attn_weights = [weights[l]['attention'] for l in layers]
    conv_weights = [weights[l]['convolution'] for l in layers]
    
    x = np.arange(len(layers))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, attn_weights, width, label='Swin Attention',
                   color=SCI_COLORS['primary'], edgecolor='black', linewidth=2)
    bars2 = ax.bar(x + width/2, conv_weights, width, label='Convolution',
                   color=SCI_COLORS['secondary'], edgecolor='black', linewidth=2)
    
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Layer', fontweight='bold')
    ax.set_ylabel('Weight Value', fontweight='bold')
    ax.set_title('ACmix Fusion Weights', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=15)
    ax.legend(frameon=False, loc='upper right')
    ax.set_ylim([0, max(max(attn_weights), max(conv_weights)) * 1.2])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_data_distribution(X, y, class_names, title, save_path, class_colors=None):
    """支持任意分类数的PCA可视化"""
    set_sci_style()
    
    # 如果没有提供颜色，自动生成
    if class_colors is None:
        class_colors = create_class_color_mapping(class_names)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for i, name in enumerate(class_names):
        mask = y == i
        color = class_colors[name]
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=name,
                  s=80, alpha=0.7, edgecolor='black', linewidth=1)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=15)
    ax.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_data_comparison(X_original, y_original, X_augmented, y_augmented,
                         class_names, save_path, class_colors=None):
    """支持任意分类数的数据对比"""
    set_sci_style()
    
    if class_colors is None:
        class_colors = create_class_color_mapping(class_names)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    pca = PCA(n_components=2)
    X_orig_pca = pca.fit_transform(X_original)
    X_aug_pca = pca.transform(X_augmented)
    
    # 原始数据
    ax1 = axes[0]
    for i, name in enumerate(class_names):
        mask = y_original == i
        color = class_colors[name]
        ax1.scatter(X_orig_pca[mask, 0], X_orig_pca[mask, 1], c=color,
                   label=name, s=150, alpha=0.9, edgecolor='black', linewidth=2)
    ax1.set_xlabel(f'PC1', fontweight='bold')
    ax1.set_ylabel(f'PC2', fontweight='bold')
    ax1.set_title(f'Original Data (n={len(y_original)})', fontweight='bold', pad=15)
    ax1.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 增强数据
    ax2 = axes[1]
    for i, name in enumerate(class_names):
        mask = y_augmented == i
        color = class_colors[name]
        ax2.scatter(X_aug_pca[mask, 0], X_aug_pca[mask, 1], c=color,
                   label=name, s=30, alpha=0.5, edgecolor='white', linewidth=0.5)
    ax2.set_xlabel(f'PC1', fontweight='bold')
    ax2.set_ylabel(f'PC2', fontweight='bold')
    ax2.set_title(f'Augmented Data (n={len(y_augmented)})', fontweight='bold', pad=15)
    ax2.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # 叠加图
    ax3 = axes[2]
    for i, name in enumerate(class_names):
        mask = y_augmented == i
        color = class_colors[name]
        ax3.scatter(X_aug_pca[mask, 0], X_aug_pca[mask, 1], c=color,
                   s=20, alpha=0.3, edgecolor='none')
    for i, name in enumerate(class_names):
        mask = y_original == i
        color = class_colors[name]
        ax3.scatter(X_orig_pca[mask, 0], X_orig_pca[mask, 1], c=color,
                   label=f'{name} (orig)', s=200, alpha=1.0, edgecolor='black',
                   linewidth=2, marker='*')
    ax3.set_xlabel(f'PC1', fontweight='bold')
    ax3.set_ylabel(f'PC2', fontweight='bold')
    ax3.set_title('Overlay (star=Original)', fontweight='bold', pad=15)
    ax3.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_augmentation_summary(aug_stats, save_path):
    """Plot augmentation method distribution."""
    set_sci_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    label_map = {
        'original': 'Original', 'gan': 'WGAN-GP', 'gan_filtered': 'Filtered',
        'noise': 'Gaussian Noise', 'mixup': 'Mixup', 'smote': 'SMOTE'
    }
    
    color_map = {
        'original': SCI_COLORS['quaternary'], 'gan': SCI_COLORS['primary'],
        'noise': SCI_COLORS['tertiary'], 'mixup': SCI_COLORS['secondary'],
        'smote': SCI_COLORS['fifth']
    }
    
    display_items = ['original', 'gan', 'noise', 'mixup', 'smote']
    labels = [label_map[m] for m in display_items if aug_stats.get(m, 0) > 0]
    counts = [aug_stats[m] for m in display_items if aug_stats.get(m, 0) > 0]
    colors = [color_map[m] for m in display_items if aug_stats.get(m, 0) > 0]
    
    ax1 = axes[0]
    if counts:
        wedges, texts, autotexts = ax1.pie(counts, labels=labels, colors=colors,
                                           autopct='%1.1f%%', startangle=90,
                                           explode=[0.03]*len(counts))
        for autotext in autotexts:
            autotext.set_fontweight('bold')
    ax1.set_title('Data Augmentation Methods', fontweight='bold', pad=15)
    
    ax2 = axes[1]
    if counts:
        x_pos = np.arange(len(counts))
        bars = ax2.bar(x_pos, counts, color=colors, edgecolor='black', linewidth=1.5)
        for bar, count in zip(bars, counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    str(count), ha='center', fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(labels, rotation=15)
    ax2.set_ylabel('Number of Samples', fontweight='bold')
    ax2.set_title('Samples per Method', fontweight='bold', pad=15)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_class_distance_analysis(X_original, y_original, X_augmented, y_augmented,
                                  class_names, save_path):
    set_sci_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    def compute_distances(X, y, num_classes):
        intra_dists, inter_dists = [], []
        
        for i in range(num_classes):
            mask_i = y == i
            X_i = X[mask_i]
            
            if len(X_i) > 1:
                intra_dists.append(np.mean(pdist(X_i)))
            
            for j in range(i+1, num_classes):
                mask_j = y == j
                X_j = X[mask_j]
                if len(X_i) > 0 and len(X_j) > 0:
                    inter_dists.append(np.mean(cdist(X_i, X_j)))
        
        return np.mean(intra_dists) if intra_dists else 0, np.mean(inter_dists) if inter_dists else 0
    
    orig_intra, orig_inter = compute_distances(X_original, y_original, len(class_names))
    aug_intra, aug_inter = compute_distances(X_augmented, y_augmented, len(class_names))
    
    ax1 = axes[0]
    x = np.arange(2)
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, [orig_intra, orig_inter], width,
                    label='Original', color=SCI_COLORS['primary'], edgecolor='black')
    bars2 = ax1.bar(x + width/2, [aug_intra, aug_inter], width,
                    label='Augmented', color=SCI_COLORS['secondary'], edgecolor='black')
    
    ax1.set_ylabel('Average Distance', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Intra-class\n(smaller=better)', 'Inter-class\n(larger=better)'])
    ax1.set_title('Distance Analysis', fontweight='bold', pad=15)
    ax1.legend(frameon=False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2 = axes[1]
    orig_ratio = orig_inter / (orig_intra + 1e-8)
    aug_ratio = aug_inter / (aug_intra + 1e-8)
    
    bars = ax2.bar(['Original', 'Augmented'], [orig_ratio, aug_ratio],
                   color=[SCI_COLORS['primary'], SCI_COLORS['secondary']],
                   edgecolor='black', linewidth=2)
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax2.set_ylabel('Inter/Intra Distance Ratio', fontweight='bold')
    ax2.set_title('Separability Index\n(higher=better)', fontweight='bold', pad=15)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_gene_importance(importance_df, top_n=30, save_path='gene_importance.pdf'):
    set_sci_style()
    
    actual_n = min(top_n, len(importance_df))
    top_genes = importance_df.head(actual_n)
    
    fig, ax = plt.subplots(figsize=(10, max(6, actual_n * 0.25)))
    
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, actual_n))
    
    y_pos = np.arange(actual_n)
    ax.barh(y_pos, top_genes['importance_grad_input'].values[::-1],
            color=colors[::-1], edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_genes['gene'].values[::-1], fontsize=9)
    ax.set_xlabel('Importance Score (Gradient x Input)', fontweight='bold')
    ax.set_title(f'Top {actual_n} Important Genes', fontweight='bold', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_gene_importance_by_class(class_importance_df, class_names, top_n=20,
                                   save_path='gene_importance_by_class.pdf', class_colors=None):
    """支持任意分类数的基因重要性"""
    set_sci_style()
    
    if class_colors is None:
        class_colors = create_class_color_mapping(class_names)
    
    n_classes = len(class_names)
    
    # 动态调整布局
    if n_classes <= 3:
        ncols = n_classes
        nrows = 1
    elif n_classes <= 6:
        ncols = 3
        nrows = 2
    else:
        ncols = 4
        nrows = (n_classes + 3) // 4
    
    figsize = (5*ncols, 8*nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    # 确保axes是数组
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    
    axes_flat = axes.flatten()
    
    for idx, class_name in enumerate(class_names):
        ax = axes_flat[idx]
        col_name = f'{class_name}_importance'
        
        if col_name not in class_importance_df.columns:
            ax.axis('off')
            continue
        
        sorted_df = class_importance_df.sort_values(col_name, ascending=False)
        actual_n = min(top_n, len(sorted_df))
        sorted_df = sorted_df.head(actual_n)
        
        y_pos = np.arange(actual_n)
        color = class_colors[class_name]
        ax.barh(y_pos, sorted_df[col_name].values[::-1],
               color=color, edgecolor='black', linewidth=0.5, alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_df['gene'].values[::-1], fontsize=8)
        ax.set_xlabel('Importance', fontweight='bold')
        ax.set_title(f'{class_name}', fontweight='bold', color=color)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # 隐藏多余的子图
    for idx in range(n_classes, len(axes_flat)):
        axes_flat[idx].axis('off')
    
    plt.suptitle('Top Genes by Phenotype', fontweight='bold', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_gene_heatmap(sample_gene_contrib, y, gene_names, class_names, top_n=30,
                      save_path='gene_heatmap.pdf', class_colors=None):
    """支持任意分类数的基因热图"""
    set_sci_style()
    
    if class_colors is None:
        class_colors = create_class_color_mapping(class_names)
    
    mean_contrib = sample_gene_contrib.mean(axis=0)
    actual_n = min(top_n, len(mean_contrib))
    top_genes = mean_contrib.nlargest(actual_n).index.tolist()
    
    heatmap_data = sample_gene_contrib[top_genes].values
    
    # 按类别排序
    sort_idx = np.argsort(y)
    heatmap_data = heatmap_data[sort_idx]
    y_sorted = y[sort_idx]
    
    fig, ax = plt.subplots(figsize=(14, max(6, actual_n * 0.25)))
    
    heatmap_norm = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min() + 1e-8)
    
    im = ax.imshow(heatmap_norm.T, aspect='auto', cmap='RdYlBu_r')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Normalized Contribution', fontweight='bold')
    
    ax.set_yticks(range(actual_n))
    ax.set_yticklabels(top_genes, fontsize=8)
    ax.set_xlabel('Sample', fontweight='bold')
    ax.set_ylabel('Gene', fontweight='bold')
    ax.set_title(f'Gene Contribution Heatmap (Top {actual_n} Genes)', fontweight='bold', pad=15)
    
    # 标记类别边界
    class_boundaries = []
    for i in range(len(class_names)):
        count = np.sum(y_sorted == i)
        if class_boundaries:
            class_boundaries.append(class_boundaries[-1] + count)
        else:
            class_boundaries.append(count)
    
    # 绘制分界线
    for boundary in class_boundaries[:-1]:
        ax.axvline(x=boundary - 0.5, color='white', linewidth=2)
    
    # 添加类别标签
    prev = 0
    for i, (boundary, name) in enumerate(zip(class_boundaries, class_names)):
        mid = (prev + boundary) / 2
        ax.text(mid, -1.5, name, ha='center', va='top', fontweight='bold',
               fontsize=10, color=class_colors[name])
        prev = boundary
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_wgcna_modules(wgcna_results, save_path):
    """Plot WGCNA module-trait heatmap."""
    set_sci_style()
    
    module_trait_cor = wgcna_results['module_trait_cor']
    module_trait_p = wgcna_results['module_trait_pvalue']
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(module_trait_cor) * 0.5)))
    
    # Heatmap
    im = ax.imshow(module_trait_cor.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation', fontweight='bold')
    
    # Labels
    ax.set_xticks(range(len(module_trait_cor.columns)))
    ax.set_yticks(range(len(module_trait_cor.index)))
    ax.set_xticklabels(module_trait_cor.columns, rotation=45, ha='right', fontweight='bold')
    ax.set_yticklabels(module_trait_cor.index, fontweight='bold')
    
    # Add correlation values and significance
    for i in range(len(module_trait_cor.index)):
        for j in range(len(module_trait_cor.columns)):
            cor = module_trait_cor.iloc[i, j]
            p = module_trait_p.iloc[i, j]
            
            sig = ''
            if p < 0.001:
                sig = '***'
            elif p < 0.01:
                sig = '**'
            elif p < 0.05:
                sig = '*'
            
            color = 'white' if abs(cor) > 0.5 else 'black'
            ax.text(j, i, f'{cor:.2f}\n{sig}', ha='center', va='center',
                   color=color, fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Phenotype', fontweight='bold')
    ax.set_ylabel('Module', fontweight='bold')
    ax.set_title('Module-Trait Correlation Heatmap', fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_module_correlation(wgcna_results, save_path):
    """Plot module-module correlation heatmap."""
    set_sci_style()
    
    module_cor = wgcna_results['module_cor_matrix']
    module_p = wgcna_results['module_p_matrix']
    
    n = len(module_cor)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.8), max(6, n * 0.6)))
    
    # Heatmap
    im = ax.imshow(module_cor.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation', fontweight='bold')
    
    # Labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(module_cor.columns, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(module_cor.index, fontsize=10)
    
    # Add significance
    for i in range(n):
        for j in range(n):
            if i != j:
                p = module_p.iloc[i, j]
                if p < 0.05:
                    sig = '*' if p >= 0.01 else ('**' if p >= 0.001 else '***')
                    color = 'white' if abs(module_cor.iloc[i, j]) > 0.5 else 'black'
                    ax.text(j, i, sig, ha='center', va='center', color=color, fontsize=10)
    
    ax.set_title('Module-Module Correlation', fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def compute_gene_importance_by_class(model, X, y, gene_names, class_names, device='cpu'):
    """Compute gene importance for each class."""
    model.eval()
    model.to(device)
    
    class_importance = {name: [] for name in class_names}
    
    for class_idx, class_name in enumerate(class_names):
        mask = y == class_idx
        X_class = X[mask]
        
        if len(X_class) == 0:
            continue
        
        X_tensor = torch.FloatTensor(X_class).to(device)
        X_tensor.requires_grad = True
        
        for i in range(len(X_class)):
            model.zero_grad()
            if X_tensor.grad is not None:
                X_tensor.grad.zero_()
            
            output = model(X_tensor[i:i+1])
            output[0, class_idx].backward(retain_graph=True)
            
            grad = X_tensor.grad[i].detach().cpu().numpy()
            class_importance[class_name].append(np.abs(grad * X_class[i]))
    
    class_importance_df = pd.DataFrame({'gene': gene_names})
    for class_name in class_names:
        if class_importance[class_name]:
            avg_importance = np.mean(class_importance[class_name], axis=0)
            class_importance_df[f'{class_name}_importance'] = avg_importance
    
    return class_importance_df


# ============================================================================
# Gene Importance Analysis
# ============================================================================

def compute_gene_importance(model, X, y, gene_names, device='cpu'):
    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)
    X_tensor.requires_grad = True
    
    all_gradients = []
    for i in range(len(X)):
        model.zero_grad()
        if X_tensor.grad is not None:
            X_tensor.grad.zero_()
        
        output = model(X_tensor[i:i+1])
        pred_class = output.argmax(dim=1).item()
        output[0, pred_class].backward(retain_graph=True)
        all_gradients.append(X_tensor.grad[i].detach().cpu().numpy())
    
    gradients = np.array(all_gradients)
    
    importance_df = pd.DataFrame({
        'gene': gene_names,
        'importance_gradient': np.abs(gradients).mean(axis=0),
        'importance_grad_input': np.abs(gradients * X).mean(axis=0),
    })
    importance_df = importance_df.sort_values('importance_grad_input', ascending=False)
    importance_df['rank'] = range(1, len(importance_df) + 1)
    
    return importance_df


# ============================================================================
# Data Loading (改进版 - 支持任意分类)
# ============================================================================

def load_data(expr_path, sample_file=None):
    """
    加载数据，支持任意数量的分类（改进版 - 更好的错误处理）
    
    Parameters:
    -----------
    expr_path : str
        表达矩阵CSV路径 (基因×样本)
    sample_file : str
        样本分组文件路径，格式：
        Group\tSample
        ClassA\tSample1
        ClassA\tSample2
        ClassB\tSample3
        ...
        
        如果为None，则尝试从样本名称推断（不推荐）
    
    Returns:
    --------
    X : np.ndarray
        表达矩阵 (样本×基因)
    y : np.ndarray
        标签 (0, 1, 2, ...)
    gene_names : list
        基因名称
    class_names : list
        类别名称（按照y的编码顺序）
    sample_to_group : dict
        {sample_name: class_name}
    """
    print("=" * 60)
    print("Loading data...")
    print("=" * 60)
    
    # 读取表达矩阵
    expr_df = pd.read_csv(expr_path, index_col=0)
    print(f"Expression matrix: {expr_df.shape[0]} genes x {expr_df.shape[1]} samples")
    
    gene_names = expr_df.index.tolist()
    all_samples = expr_df.columns.tolist()
    
    if sample_file:
        # 从sample_file读取分组信息
        print(f"Reading sample grouping from: {sample_file}")
        
        sample_info = []
        
        # 首先，读取所有行进行调试
        try:
            with open(sample_file, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
            
            print(f"Total lines in file: {len(all_lines)}")
            
            # 检测分隔符
            first_data_line = None
            for line in all_lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    first_data_line = line
                    break
            
            if first_data_line:
                if '\t' in first_data_line:
                    delimiter = '\t'
                    print(f"Detected delimiter: TAB")
                elif ',' in first_data_line:
                    delimiter = ','
                    print(f"Detected delimiter: COMMA")
                else:
                    delimiter = None
                    print(f"Warning: No clear delimiter detected")
            else:
                raise ValueError("File appears to be empty or contains only comments")
            
            # 判断是否有表头
            has_header = False
            if first_data_line:
                first_parts = first_data_line.split(delimiter) if delimiter else [first_data_line]
                if len(first_parts) >= 2:
                    # 检查第一行是否看起来像表头
                    if any(keyword in first_parts[0].lower() for keyword in ['group', 'class', 'type', 'phenotype']):
                        has_header = True
                        print(f"Detected header line: {first_data_line}")
            
            # 解析文件
            line_count = 0
            skipped_lines = []
            
            for line_num, line in enumerate(all_lines, 1):
                line = line.strip()
                
                # 跳过空行
                if not line:
                    continue
                
                # 跳过注释行
                if line.startswith('#'):
                    continue
                
                # 跳过表头（只跳过第一个数据行，如果它是表头）
                if has_header and line_count == 0:
                    line_count += 1
                    continue
                
                # 分割
                if delimiter:
                    parts = [p.strip() for p in line.split(delimiter)]
                else:
                    # 尝试空格分割
                    parts = [p.strip() for p in line.split()]
                
                if len(parts) >= 2:
                    group_name = parts[0]
                    sample_name = parts[1]
                    sample_info.append((group_name, sample_name))
                    line_count += 1
                else:
                    skipped_lines.append(f"Line {line_num}: {line} (only {len(parts)} parts)")
            
            print(f"Successfully parsed {line_count} sample entries")
            
            if skipped_lines:
                print(f"Skipped {len(skipped_lines)} lines:")
                for skip_info in skipped_lines[:5]:  # 只显示前5个
                    print(f"  {skip_info}")
                if len(skipped_lines) > 5:
                    print(f"  ... and {len(skipped_lines) - 5} more")
        
        except Exception as e:
            print(f"Error reading sample file: {e}")
            print(f"File path: {sample_file}")
            raise
        
        if not sample_info:
            print("\n" + "=" * 60)
            print("ERROR: No valid sample information found!")
            print("=" * 60)
            print("\nPlease check your sample file format:")
            print("\nExpected format (with TAB or COMMA separator):")
            print("  Group\tSample")
            print("  ClassA\tSample1")
            print("  ClassA\tSample2")
            print("  ClassB\tSample3")
            print("\nOr without header:")
            print("  ClassA\tSample1")
            print("  ClassA\tSample2")
            print("  ClassB\tSample3")
            print("\nFirst few lines of your file:")
            try:
                with open(sample_file, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i < 5:
                            print(f"  Line {i+1}: {repr(line.strip())}")
            except:
                pass
            print("=" * 60)
            raise ValueError(f"No valid sample information found in {sample_file}")
        
        # 提取类别名称（保持输入文件中的顺序）
        class_names = []
        seen = set()
        for group_name, _ in sample_info:
            if group_name not in seen:
                class_names.append(group_name)
                seen.add(group_name)
        
        # 创建类别到索引的映射
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        
        print(f"Detected {len(class_names)} classes: {class_names}")
        
        # 匹配样本
        valid_samples = []
        valid_labels = []
        sample_to_group = {}
        not_found_samples = []
        
        for group_name, sample_name in sample_info:
            if sample_name in all_samples:
                valid_samples.append(sample_name)
                valid_labels.append(class_to_idx[group_name])
                sample_to_group[sample_name] = group_name
            else:
                not_found_samples.append(sample_name)
        
        if not_found_samples:
            print(f"\nWarning: {len(not_found_samples)} samples not found in expression matrix:")
            for sample in not_found_samples[:10]:
                print(f"  - {sample}")
            if len(not_found_samples) > 10:
                print(f"  ... and {len(not_found_samples) - 10} more")
        
        if not valid_samples:
            print("\n" + "=" * 60)
            print("ERROR: No matching samples found!")
            print("=" * 60)
            print("\nSamples in sample file:")
            for i, (group, sample) in enumerate(sample_info[:10]):
                print(f"  {group}\t{sample}")
            if len(sample_info) > 10:
                print(f"  ... and {len(sample_info) - 10} more")
            print("\nSamples in expression matrix (first 10):")
            for sample in all_samples[:10]:
                print(f"  {sample}")
            if len(all_samples) > 10:
                print(f"  ... and {len(all_samples) - 10} more")
            print("=" * 60)
            raise ValueError("No matching samples found between sample file and expression matrix!")
        
        # 提取对应的表达数据
        X = expr_df[valid_samples].values.T  # 转置为 样本×基因
        y = np.array(valid_labels)
        
        print(f"Matched {len(valid_samples)} samples")
        
    else:
        # 如果没有sample_file，尝试从样本名称推断（不推荐）
        print("Warning: No sample file provided. Attempting to infer classes from sample names...")
        print("  Recommendation: Please provide a sample grouping file for accurate classification.")
        
        # 简单策略：用下划线或其他分隔符分割
        labels = []
        for name in all_samples:
            # 尝试提取第一个下划线前的部分
            if '_' in name:
                label = name.split('_')[0]
            elif '-' in name:
                label = name.split('-')[0]
            else:
                label = 'Unknown'
            labels.append(label)
        
        # 使用LabelEncoder编码
        le = LabelEncoder()
        y = le.fit_transform(labels)
        class_names = le.classes_.tolist()
        
        X = expr_df.T.values
        sample_to_group = {s: l for s, l in zip(all_samples, labels)}
        
        print(f"Inferred {len(class_names)} classes: {class_names}")
        print(f"  Warning: This may not be accurate. Please verify!")
    
    # 处理NaN值
    X = np.nan_to_num(X, nan=0.0)
    
    # 打印样本分布
    print(f"\nSample distribution:")
    for i, class_name in enumerate(class_names):
        count = np.sum(y == i)
        print(f"  {class_name}: {count} samples")
    
    return X, y, gene_names, class_names, sample_to_group


# ============================================================================
# Main Function
# ============================================================================

def main(expr_path, output_dir='./output_wgcna', sample_file=None,
         n_features=100, samples_per_class=50, gan_epochs=600,
         embed_dim=64, num_heads=4, window_size=7, dropout=0.3,
         epochs=300, batch_size=16, learning_rate=1e-4, weight_decay=1e-3,
         patience=50, use_mixup=False, mixup_alpha=0.2, label_smoothing=0.05,
         n_overall_hub=20, n_phenotype_hub=10):
    """
    Main pipeline with WGCNA integration - 支持任意分类数
    """
    
    os.makedirs(output_dir, exist_ok=True)
    input_dir = os.path.join(output_dir, 'Input')
    os.makedirs(input_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # ========== Step 1: Load Data ==========
    X_original, y_original, gene_names, class_names, sample_to_group = load_data(expr_path, sample_file)
    
    # 创建类别颜色映射
    class_colors = create_class_color_mapping(class_names)
    print(f"\nClass color mapping:")
    for class_name, color in class_colors.items():
        print(f"  {class_name}: {color}")
    
    # ========== Step 2: Feature Selection ==========
    print("\n" + "=" * 60)
    print("Step 1: Feature Selection")
    print("=" * 60)
    
    selector = FeatureSelector(n_features=n_features, method='knn_combined')
    X_selected = selector.fit_transform(X_original, y_original, gene_names)
    selected_gene_names = selector.selected_names
    
    # ========== Step 3: WGCNA Analysis ==========
    print("\n" + "=" * 60)
    print("Step 2: WGCNA Analysis")
    print("=" * 60)
    
    # Prepare trait data (one-hot encoding)
    trait_data = pd.DataFrame(index=range(len(y_original)))
    for i, class_name in enumerate(class_names):
        trait_data[class_name] = (y_original == i).astype(float)
    
    # Run WGCNA
    wgcna = PythonWGCNA(min_module_size=max(5, n_features // 20), merge_cut_height=0.25)
    wgcna_results = wgcna.fit(X_selected, trait_data, gene_names=selected_gene_names)
    
    # ========== Step 4: Data Augmentation ==========
    print("\n" + "=" * 60)
    print("Step 3: Data Augmentation")
    print("=" * 60)
    
    augmenter = ImprovedAugmenter(num_classes=len(class_names), device=device)
    augmenter.fit(X_selected, y_original, gan_epochs=gan_epochs)
    X_augmented, y_augmented, aug_stats = augmenter.augment(X_selected, y_original, samples_per_class)
    
    # ========== Step 5: Train Classifier ==========
    print("\n" + "=" * 60)
    print("Step 4: Train ACmix-Swin Classifier")
    print("=" * 60)
    
    model, history, all_preds, all_labels, all_probs, X_test, y_test = train_classifier(
        X_augmented, y_augmented, len(class_names), device,
        embed_dim=embed_dim,
        num_heads=num_heads,
        window_size=window_size,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        use_mixup=use_mixup,
        mixup_alpha=mixup_alpha,
        label_smoothing=label_smoothing
    )
    
    final_accuracy = accuracy_score(all_labels, all_preds) * 100
    print(f"\nFinal Test Accuracy: {final_accuracy:.2f}%")
    
    # ========== Step 6: Gene Importance ==========
    print("\n" + "=" * 60)
    print("Step 5: Gene Importance Analysis")
    print("=" * 60)
    
    importance_df = compute_gene_importance(model, X_test, y_test, selected_gene_names, device)
    
    # ========== Step 7: Generate Network Files ==========
    print("\n" + "=" * 60)
    print("Step 6: Generate Network Files for R Visualization")
    print("=" * 60)
    
    generator = NetworkFileGenerator(output_dir=input_dir)
    
    # Hub gene selection
    sample_groups = [class_names[l] for l in y_original]
    overall_hub, phenotype_hub, gene_scores = generator.generate_hub_genes(
        importance_df, wgcna_results, selected_gene_names, class_names,
        n_overall_hub=n_overall_hub, n_phenotype_hub=n_phenotype_hub
    )
    
    # Generate network files
    generator.generate_node_edge_layout(
        overall_hub, phenotype_hub, gene_scores, class_names,
        wgcna_results, X_selected, sample_groups
    )
    
    # Generate WGCNA files
    generator.generate_wgcna_files(wgcna_results, class_names)
    
    # Generate metabolite types
    generator.generate_metabolite_types(overall_hub, phenotype_hub)
    
    # ========== Step 8: Visualization ==========
    print("\n" + "=" * 60)
    print("Step 7: Generating Visualizations")
    print("=" * 60)
    
    # Training curves
    plot_training_curves(history, os.path.join(output_dir, 'training_curves.pdf'))
    
    # WGAN training (if available)
    if hasattr(augmenter, 'gan_history') and augmenter.gan_history:
        plot_wgan_training(augmenter.gan_history, os.path.join(output_dir, 'wgan_training.pdf'))
    
    # Confusion matrix (使用动态颜色)
    plot_confusion_matrix(all_labels, all_preds, class_names,
                         os.path.join(output_dir, 'confusion_matrix.pdf'))
    
    # ACmix weights
    plot_acmix_weights(model, os.path.join(output_dir, 'acmix_weights.pdf'))
    
    # Augmentation summary
    plot_augmentation_summary(aug_stats, os.path.join(output_dir, 'augmentation_summary.pdf'))
    
    # Data distribution - original (使用类别颜色)
    X_selected_scaled = augmenter.scaler.transform(selector.transform(X_original))
    plot_data_distribution(X_selected_scaled, y_original, class_names,
                          f'Original Data (n={len(y_original)})',
                          os.path.join(output_dir, 'data_original.pdf'),
                          class_colors=class_colors)
    
    # Data distribution - augmented
    plot_data_distribution(X_augmented, y_augmented, class_names,
                          f'Augmented Data (n={len(y_augmented)})',
                          os.path.join(output_dir, 'data_augmented.pdf'),
                          class_colors=class_colors)
    
    # Data comparison
    plot_data_comparison(X_selected_scaled, y_original, X_augmented, y_augmented,
                        class_names, os.path.join(output_dir, 'data_comparison.pdf'),
                        class_colors=class_colors)
    
    # Distance analysis
    plot_class_distance_analysis(X_selected_scaled, y_original, X_augmented, y_augmented,
                                 class_names, os.path.join(output_dir, 'distance_analysis.pdf'))
    
    # Gene importance
    plot_gene_importance(importance_df, top_n=30,
                        save_path=os.path.join(output_dir, 'gene_importance.pdf'))
    
    # Gene importance by class (使用类别颜色)
    class_importance_df = compute_gene_importance_by_class(
        model, X_test, y_test, selected_gene_names, class_names, device
    )
    plot_gene_importance_by_class(class_importance_df, class_names, top_n=20,
                                  save_path=os.path.join(output_dir, 'gene_importance_by_class.pdf'),
                                  class_colors=class_colors)
    
    # Gene heatmap (使用类别颜色)
    model.eval()
    X_tensor = torch.FloatTensor(X_test).to(device)
    X_tensor.requires_grad = True
    all_grads = []
    for i in range(len(X_test)):
        model.zero_grad()
        if X_tensor.grad is not None:
            X_tensor.grad.zero_()
        output = model(X_tensor[i:i+1])
        pred_class = output.argmax(dim=1).item()
        output[0, pred_class].backward(retain_graph=True)
        all_grads.append(X_tensor.grad[i].detach().cpu().numpy())
    sample_gene_contrib = pd.DataFrame(
        np.abs(np.array(all_grads) * X_test),
        columns=selected_gene_names
    )
    plot_gene_heatmap(sample_gene_contrib, y_test, selected_gene_names, class_names,
                     top_n=30, save_path=os.path.join(output_dir, 'gene_heatmap.pdf'),
                     class_colors=class_colors)
    
    # WGCNA visualizations
    plot_wgcna_modules(wgcna_results, os.path.join(output_dir, 'wgcna_module_trait.pdf'))
    plot_module_correlation(wgcna_results, os.path.join(output_dir, 'wgcna_module_correlation.pdf'))
    
    # ========== Step 9: Save Results ==========
    print("\n" + "=" * 60)
    print("Step 8: Save Results")
    print("=" * 60)
    
    # Training history
    pd.DataFrame(history).to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)
    print(f"  Saved: training_history.csv")
    
    # Gene importance
    importance_df.to_csv(os.path.join(output_dir, 'gene_importance.csv'), index=False)
    print(f"  Saved: gene_importance.csv")
    
    # Gene importance by class
    class_importance_df.to_csv(os.path.join(output_dir, 'gene_importance_by_class.csv'), index=False)
    print(f"  Saved: gene_importance_by_class.csv")
    
    # Sample gene contribution
    sample_gene_contrib.to_csv(os.path.join(output_dir, 'sample_gene_contribution.csv'), index=False)
    print(f"  Saved: sample_gene_contribution.csv")
    
    # Gene scores with WGCNA info
    gene_scores.to_csv(os.path.join(output_dir, 'gene_scores_combined.csv'), index=False)
    print(f"  Saved: gene_scores_combined.csv")
    
    # Predictions
    pred_df = pd.DataFrame({
        'True_Label': [class_names[i] for i in all_labels],
        'Predicted_Label': [class_names[i] for i in all_preds],
        'Correct': all_labels == all_preds
    })
    for i, name in enumerate(class_names):
        pred_df[f'Prob_{name}'] = all_probs[:, i]
    pred_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    print(f"  Saved: predictions.csv")
    
    # Selected features
    pd.DataFrame({
        'gene': selected_gene_names,
        'original_index': selector.selected_indices
    }).to_csv(os.path.join(output_dir, 'selected_features.csv'), index=False)
    print(f"  Saved: selected_features.csv")
    
    # Class color mapping
    pd.DataFrame({
        'class': list(class_colors.keys()),
        'color': list(class_colors.values())
    }).to_csv(os.path.join(output_dir, 'class_colors.csv'), index=False)
    print(f"  Saved: class_colors.csv")
    
    # Model
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'class_colors': class_colors,
        'selected_features': selected_gene_names,
        'final_accuracy': final_accuracy,
        'acmix_weights': model.get_acmix_weights()
    }, os.path.join(output_dir, 'model.pth'))
    print(f"  Saved: model.pth")
    
    # ========== Summary ==========
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"\nNumber of Classes: {len(class_names)}")
    print(f"Classes: {class_names}")
    print(f"\nTest Accuracy: {final_accuracy:.2f}%")
    print(f"\nOutput files saved to: {output_dir}/")
    print(f"\n★ 支持任意数量的分类!")
    print(f"  当前分析: {len(class_names)}个类别")
    
    return model, wgcna_results, importance_df, final_accuracy


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='ACmix-Swin + WGCNA Integration (支持任意分类数)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Example Usage:
  # 基本用法（必须提供sample_file）
  python acmix_swin_wgcna_v2.py --expr exp.csv --samples samples.txt
  
  # 自定义模型参数
  python acmix_swin_wgcna_v2.py --expr exp.csv --samples samples.txt \\
      --embed_dim 32 --dropout 0.6 --lr 5e-5 --use_mixup
  
Sample File Format (samples.txt):
  Group    Sample
  ClassA   Sample1
  ClassA   Sample2
  ClassB   Sample3
  ClassB   Sample4
  ClassC   Sample5
  ...
  
注意：
  - sample_file是必须的，用于正确识别分类
  - 支持任意数量的分类（2类、3类、4类、...）
  - 类别名称会按照文件中出现的顺序保留
        '''
    )
    
    # 数据参数
    parser.add_argument('--expr', type=str, required=True, help='Expression matrix CSV path')
    parser.add_argument('--samples', type=str, required=True, help='Sample grouping file (REQUIRED)')
    parser.add_argument('--output', type=str, default='./output_wgcna', help='Output directory')
    
    # 特征选择参数
    parser.add_argument('--n_features', type=int, default=100, help='Number of features to select')
    
    # 数据增强参数
    parser.add_argument('--samples_per_class', type=int, default=50, help='Samples per class after augmentation')
    parser.add_argument('--gan_epochs', type=int, default=600, help='GAN training epochs')
    
    # 模型架构参数
    parser.add_argument('--embed_dim', type=int, default=64, help='Embedding dimension (32, 64, 128)')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--window_size', type=int, default=7, help='Swin window size')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (0.1-0.6)')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=300, help='Max training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay (L2 regularization)')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--use_mixup', action='store_true', help='Enable Mixup augmentation during training')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='Mixup alpha parameter')
    parser.add_argument('--label_smoothing', type=float, default=0.05, help='Label smoothing factor')
    
    # Hub基因参数
    parser.add_argument('--n_overall_hub', type=int, default=20, help='Number of overall hub genes')
    parser.add_argument('--n_phenotype_hub', type=int, default=10, help='Number of hub genes per phenotype')
    
    args = parser.parse_args()
    
    # 检查必需参数
    if not args.samples:
        print("\nError: --samples parameter is REQUIRED for correct classification!")
        print("Please provide a sample grouping file.\n")
        parser.print_help()
        exit(1)
    
    main(
        expr_path=args.expr,
        output_dir=args.output,
        sample_file=args.samples,
        n_features=args.n_features,
        samples_per_class=args.samples_per_class,
        gan_epochs=args.gan_epochs,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        window_size=args.window_size,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        use_mixup=args.use_mixup,
        mixup_alpha=args.mixup_alpha,
        label_smoothing=args.label_smoothing,
        n_overall_hub=args.n_overall_hub,
        n_phenotype_hub=args.n_phenotype_hub
    )
