#!/usr/bin/env python3
"""
Script de benchmark comparatif entre CapyMOA et River.
Génère des figures et tableaux pour comparer les performances des deux bibliothèques.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration des styles
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Palette de couleurs fixe pour les deux bibliothèques
LIB_COLORS = {
    "CapyMOA": "#1f77b4",  # bleu
    "River": "#ff7f0e",    # orange
}


def load_csv_files(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Charge tous les fichiers CSV de résultats."""
    csv_files = {}
    
    for csv_file in sorted(data_dir.glob("*.csv")):
        # Ignorer les fichiers de résultats finaux si on les génère
        if csv_file.name.startswith("benchmark_comparison"):
            continue
        
        try:
            df = pd.read_csv(csv_file)
            # Identifier le type (CapyMOA ou River) et la tâche (robuste)
            name = csv_file.stem
            name_lower = name.lower()
            
            # Bibliothèque
            library = "CapyMOA" if "capymoa" in name_lower else "River"
            
            # Tâche: détection robuste par motifs
            if "regression" in name_lower:
                task = "Regression"
            elif "multiclass" in name_lower:
                task = "Multiclass Classification"
            elif "binary" in name_lower and "class" in name_lower:
                task = "Binary Classification"
            elif "anomaly" in name_lower:
                task = "Anomaly Detection"
            else:
                # Fallback: nettoyage de préfixe et underscores
                base = name_lower
                base = base.replace("capymoa__", "capymoa_")
                if base.startswith("capymoa_"):
                    base = base[len("capymoa_"):]
                task = base.replace("_", " ").title()
            
            # Harmoniser les colonnes temps/mémoire si différents noms
            # Chercher "Time in s" (exact ou variations)
            if "Time in s" not in df.columns:
                for cand in ["Time in s", "Time_s", "time_s", "time", "elapsed_s", "Elapsed s", "Time in S"]:
                    if cand in df.columns:
                        df["Time in s"] = df[cand]
                        break
            
            # Chercher "Memory in Mb" (exact ou variations)
            # CapyMOA utilise "Memory in Mb", River aussi
            if "Memory in Mb" not in df.columns:
                # Chercher d'abord les variations exactes de "Memory in Mb"
                for cand in ["Memory in Mb", "Memory in MB", "memory in mb", "Memory_In_Mb"]:
                    if cand in df.columns:
                        df["Memory in Mb"] = df[cand]
                        break
                # Si pas trouvé, chercher d'autres variantes
                if "Memory in Mb" not in df.columns:
                    for cand in ["Memory_MB", "memory_mb", "memory", "RSS_MB", "rss_mb", "MemoryMB"]:
                        if cand in df.columns:
                            df["Memory in Mb"] = df[cand]
                            break
            
            key = f"{library}_{task}"
            csv_files[key] = df
            logger.info(f"Loaded {csv_file.name}: {len(df)} rows, library={library}, task={task}")
            
        except Exception as e:
            logger.warning(f"Failed to load {csv_file.name}: {e}")
    
    return csv_files


def normalize_dataset_names(df: pd.DataFrame, library: str) -> pd.DataFrame:
    """Normalise les noms de datasets pour faciliter la comparaison.
    
    Args:
        df: DataFrame avec colonne 'dataset'
        library: 'CapyMOA' ou 'River' pour savoir quel mapping appliquer
    """
    df = df.copy()
    
    if 'dataset' not in df.columns:
        return df
    
    # Mapping explicite pour normaliser les noms de datasets entre CapyMOA et River
    # On normalise vers un format commun (CamelCase comme River)
    # CapyMOA utilise souvent des noms en minuscules, River utilise CamelCase
    dataset_mapping = {
        # Régression
        "chickweights": "ChickWeights",
        "trumpapproval": "TrumpApproval",
        # Ajouter d'autres mappings si nécessaire pour d'autres types de problèmes
    }
    
    # Appliquer le mapping (fonctionne pour CapyMOA et River)
    # Si le nom n'est pas dans le mapping, on le garde tel quel
    df['dataset'] = df['dataset'].map(dataset_mapping).fillna(df['dataset'])
    
    return df


def normalize_model_names(df: pd.DataFrame, library: str) -> pd.DataFrame:
    """Normalise les noms de modèles pour faciliter la comparaison.
    
    Args:
        df: DataFrame avec colonne 'model'
        library: 'CapyMOA' ou 'River' pour savoir quel mapping appliquer
    """
    df = df.copy()
    
    # Mapping unidirectionnel: CapyMOA -> nom commun (basé sur River pour lisibilité)
    # On ne normalise QUE les noms CapyMOA vers un nom commun
    # Les noms River restent inchangés (sauf s'ils ont un équivalent CapyMOA)
    if library == "CapyMOA":
        name_mapping = {
            "AdaptiveRandomForestClassifier": "Adaptive Random Forest",
            "HoeffdingTree": "Hoeffding Tree",
            "HoeffdingAdaptiveTree": "Hoeffding Adaptive Tree",
            "NaiveBayes": "Naive Bayes",
            "KNN": "k-Nearest Neighbors",
            "StreamingRandomPatches": "Streaming Random Patches",
            "OnlineAdwinBagging": "ADWIN Bagging",
            "LeveragingBagging": "Leveraging Bagging",
            "OnlineBagging": "Bagging",
            "OzaBoost": "OzaBoost (CapyMOA)",  # Garder distinct pour éviter confusion avec AdaBoost River
            "SGDClassifier": "Logistic regression",
            "PassiveAggressiveClassifier": "Passive-Aggressive Classifier",
        }
        df['model_normalized'] = df['model'].map(name_mapping).fillna(df['model'])
    else:
        # River: garder les noms originaux
        df['model_normalized'] = df['model']
    
    return df


def filter_multiclass_models(df: pd.DataFrame, library: str) -> pd.DataFrame:
    """Filtre les modèles pour la classification multiclasse selon la bibliothèque.
    
    Args:
        df: DataFrame avec les résultats (doit avoir model_normalized après normalisation)
        library: 'CapyMOA' ou 'River'
    
    Returns:
        DataFrame filtré avec seulement les modèles autorisés
    """
    if 'model_normalized' not in df.columns:
        logger.warning("model_normalized column not found, cannot filter multiclass models")
        return df
    
    # Modèles autorisés pour River (noms normalisés - River garde ses noms originaux)
    river_allowed_models = {
        'Hoeffding Adaptive Tree',
        'Hoeffding Tree',
        'k-Nearest Neighbors',
        'Bagging',
        'Leveraging Bagging',
        'ADWIN Bagging',
        'Naive Bayes',
        'AdaBoost',
        '[baseline] Last Class'
    }
    
    # Modèles autorisés pour CapyMOA (noms normalisés après transformation)
    # Note: Certains noms CapyMOA sont normalisés (ex: HoeffdingTree -> Hoeffding Tree)
    # D'autres gardent leur nom original (ex: NoChange, SAMKNN, etc.)
    capymoa_allowed_models = {
        # Noms normalisés (transformés par normalize_model_names)
        'Hoeffding Adaptive Tree',  # HoeffdingAdaptiveTree -> Hoeffding Adaptive Tree
        'Hoeffding Tree',  # HoeffdingTree -> Hoeffding Tree
        'k-Nearest Neighbors',  # KNN -> k-Nearest Neighbors
        'Bagging',  # OnlineBagging -> Bagging
        'Leveraging Bagging',  # LeveragingBagging -> Leveraging Bagging
        'ADWIN Bagging',  # OnlineAdwinBagging -> ADWIN Bagging
        'Naive Bayes',  # NaiveBayes -> Naive Bayes
        # Noms non normalisés (gardent leur nom original)
        'OnlineSmoothBoost',  # Pas dans le mapping, garde son nom original
        'SAMKNN',
        'Streaming Random Patches',  # StreamingRandomPatches -> Streaming Random Patches
        'WeightedkNN',
        '[baseline] Last Class'
    }
    
    if library == "River":
        allowed = river_allowed_models
    elif library == "CapyMOA":
        allowed = capymoa_allowed_models
    else:
        # Si bibliothèque inconnue, ne pas filtrer
        return df
    
    # Filtrer les modèles en utilisant model_normalized
    filtered_df = df[df['model_normalized'].isin(allowed)].copy()
    
    if len(filtered_df) < len(df):
        logger.info(f"Filtered {len(df) - len(filtered_df)} models for {library} multiclass classification")
        logger.debug(f"Allowed models for {library}: {sorted(allowed)}")
        logger.debug(f"Available models: {sorted(df['model_normalized'].unique())}")
    
    return filtered_df


def get_final_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Extrait les métriques finales pour chaque modèle/dataset.
    
    Pour chaque modèle/dataset, prend la moyenne des derniers checkpoints
    (derniers 10% ou au moins 3 checkpoints) pour avoir des métriques plus stables.
    Si une métrique est NaN dans tous les checkpoints, elle reste NaN.
    """
    final_metrics = []
    
    for (model, dataset), group in df.groupby(['model', 'dataset']):
        # Trier par step
        group_sorted = group.sort_values('step')
        
        # Prendre les derniers checkpoints (derniers 10% ou au moins 3)
        n_checkpoints = len(group_sorted)
        n_to_use = max(3, int(n_checkpoints * 0.1))
        last_checkpoints = group_sorted.tail(n_to_use)
        
        # Calculer la moyenne des métriques numériques
        final_row = {}
        
        # Colonnes non-métriques : prendre la dernière valeur
        non_metric_cols = ['step', 'track', 'model', 'dataset', 'Memory in Mb', 'Time in s', 'Valid updates']
        for col in non_metric_cols:
            if col in last_checkpoints.columns:
                final_row[col] = last_checkpoints[col].iloc[-1]  # Dernière valeur
        
        # Métriques : prendre la moyenne (ignore NaN automatiquement)
        metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1', 'MicroF1', 'MacroF1', 'MAE', 'RMSE']
        for col in metric_cols:
            if col in last_checkpoints.columns:
                # Calculer la moyenne en ignorant NaN
                values = last_checkpoints[col].dropna()
                if len(values) > 0:
                    # Pour les métriques F1, MicroF1, MacroF1, on peut avoir des 0 valides
                    # Mais si toutes les valeurs sont 0, c'est suspect - garder quand même
                    final_row[col] = values.mean()
                else:
                    # Toutes les valeurs sont NaN
                    final_row[col] = np.nan
        
        final_metrics.append(final_row)
    
    return pd.DataFrame(final_metrics)


def compare_libraries(csv_files: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Compare les résultats entre CapyMOA et River."""
    
    # Organiser les données par tâche
    tasks = {}
    for key, df in csv_files.items():
        parts = key.split("_", 1)
        if len(parts) == 2:
            library, task = parts
            if task not in tasks:
                tasks[task] = {}
            tasks[task][library] = df
    
    # Créer le répertoire de sortie
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Générer les comparaisons pour chaque tâche
    for task_name, task_data in tasks.items():
        logger.info(f"Processing task: {task_name}")
        
        # Normaliser les noms de datasets et de modèles (en passant la bibliothèque pour éviter les fausses correspondances)
        for library in task_data:
            task_data[library] = normalize_dataset_names(task_data[library], library)
            task_data[library] = normalize_model_names(task_data[library], library)
            
            # Filtrer les modèles pour la classification multiclasse
            if task_name == "Multiclass Classification":
                task_data[library] = filter_multiclass_models(task_data[library], library)
        
        # Extraire les métriques finales
        final_metrics = {}
        for library, df in task_data.items():
            final_metrics[library] = get_final_metrics(df)
        
        # Générer les visualisations (nécessite au moins 2 bibliothèques pour la comparaison)
        if len(task_data) >= 2:
            generate_task_comparison(task_name, task_data, final_metrics, output_dir)
        
        # Générer les visualisations par dataset (temps et mémoire) - fonctionne même avec une seule bibliothèque
        generate_time_memory_by_dataset(task_name, task_data, final_metrics, output_dir)


def generate_task_comparison(
    task_name: str,
    task_data: dict[str, pd.DataFrame],
    final_metrics: dict[str, pd.DataFrame],
    output_dir: Path
) -> None:
    """Génère les visualisations pour une tâche spécifique."""
    
    # Identifier les métriques disponibles
    capymoa_df = task_data.get("CapyMOA")
    river_df = task_data.get("River")
    
    if capymoa_df is None or river_df is None:
        logger.warning(f"Missing data for {task_name}, skipping visualizations")
        return
    
    # Métriques communes
    common_metrics = set(capymoa_df.columns) & set(river_df.columns)
    
    # Détecter si c'est une tâche multiclasse
    is_multiclass = "multiclass" in task_name.lower()
    
    # Pour la classification multiclasse, on utilise MicroF1 et MacroF1, PAS F1
    # Pour la classification binaire, on utilise F1, Precision, Recall
    metrics_to_plot = []
    if is_multiclass:
        # Multiclass: utiliser seulement MicroF1 et MacroF1 (pas F1)
        for m in ['Accuracy', 'MAE', 'RMSE', 'MicroF1', 'MacroF1']:
            if m in common_metrics:
                metrics_to_plot.append(m)
    else:
        # Binary/Anomaly: utiliser F1, Precision, Recall
        for m in ['Accuracy', 'MAE', 'RMSE', 'F1', 'Precision', 'Recall']:
            if m in common_metrics:
                metrics_to_plot.append(m)
    
    # Si on n'a pas de métriques à tracer, essayer de trouver des alternatives
    if not metrics_to_plot:
        # Fallback: utiliser Accuracy si disponible
        if 'Accuracy' in common_metrics:
            metrics_to_plot.append('Accuracy')
    
    if not metrics_to_plot:
        logger.warning(f"No common metrics found for {task_name}")
        return
    
    # Créer un PDF pour cette tâche
    pdf_path = output_dir / f"benchmark_comparison_{task_name.lower().replace(' ', '_')}.pdf"
    task_safe_name = task_name.lower().replace(' ', '_')
    
    with PdfPages(pdf_path) as pdf:
        # 1. Comparaison des métriques finales par modèle
        for metric in metrics_to_plot:
            fig = compare_metric_by_model(final_metrics, metric, task_name)
            if fig:
                pdf.savefig(fig, bbox_inches='tight')
                # Sauvegarder aussi en PNG
                png_path = output_dir / f"{task_safe_name}_metric_comparison_{metric.lower()}.png"
                fig.savefig(png_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved PNG: {png_path}")
                plt.close(fig)
        
        # 2. Évolution des métriques au fil du temps
        for metric in metrics_to_plot:
            fig = plot_metric_evolution(task_data, metric, task_name)
            if fig:
                pdf.savefig(fig, bbox_inches='tight')
                # Sauvegarder aussi en PNG
                png_path = output_dir / f"{task_safe_name}_metric_evolution_{metric.lower()}.png"
                fig.savefig(png_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved PNG: {png_path}")
                plt.close(fig)
        
        # 3. Comparaison temps/mémoire
        fig = compare_time_memory(final_metrics, task_name)
        if fig:
            pdf.savefig(fig, bbox_inches='tight')
            # Sauvegarder aussi en PNG
            png_path = output_dir / f"{task_safe_name}_time_memory_comparison.png"
            fig.savefig(png_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved PNG: {png_path}")
            plt.close(fig)
        
        # 4. Tableau récapitulatif
        fig = create_summary_table(final_metrics, metrics_to_plot, task_name)
        if fig:
            pdf.savefig(fig, bbox_inches='tight')
            # Sauvegarder aussi en PNG
            png_path = output_dir / f"{task_safe_name}_summary_table.png"
            fig.savefig(png_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved PNG: {png_path}")
            plt.close(fig)
        
        # 5. Heatmap de comparaison
        fig = create_comparison_heatmap(final_metrics, metrics_to_plot, task_name)
        if fig:
            pdf.savefig(fig, bbox_inches='tight')
            # Sauvegarder aussi en PNG
            png_path = output_dir / f"{task_safe_name}_comparison_heatmap.png"
            fig.savefig(png_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved PNG: {png_path}")
            plt.close(fig)
    
    logger.info(f"Generated PDF: {pdf_path}")
    
    # Générer aussi un CSV récapitulatif
    summary_csv = create_summary_csv(final_metrics, metrics_to_plot, task_name, output_dir)
    if summary_csv:
        logger.info(f"Generated summary CSV: {summary_csv}")


def compare_metric_by_model(
    final_metrics: dict[str, pd.DataFrame],
    metric: str,
    task_name: str
) -> plt.Figure | None:
    """Compare une métrique entre les bibliothèques par modèle."""
    
    # Préparer les données
    comparison_data = []
    
    for library, df in final_metrics.items():
        if metric not in df.columns:
            continue
        
        for _, row in df.iterrows():
            model = row.get('model_normalized', row.get('model', 'Unknown'))
            dataset = row.get('dataset', 'Unknown')
            value = row[metric]
            
            comparison_data.append({
                'Library': library,
                'Model': model,
                'Dataset': dataset,
                'Metric': metric,
                'Value': value
            })
    
    if not comparison_data:
        return None
    
    comp_df = pd.DataFrame(comparison_data)
    
    # Créer la figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Graphique 1: Comparaison par modèle (moyenne sur tous les datasets)
    # Filtrer les NaN avant de calculer la moyenne pour éviter que NaN soit traité comme 0
    comp_df_clean = comp_df[comp_df['Value'].notna()].copy()
    
    if len(comp_df_clean) == 0:
        logger.warning(f"No valid {metric} values found for plotting")
        return None
    
    model_avg = comp_df_clean.groupby(['Library', 'Model'])['Value'].mean().reset_index()
    
    # Pivot pour faciliter la comparaison
    pivot_avg = model_avg.pivot(index='Model', columns='Library', values='Value')
    
    # Filtrer les modèles qui n'ont aucune valeur valide
    pivot_avg = pivot_avg.dropna(how='all')
    
    if len(pivot_avg) == 0:
        logger.warning(f"No valid {metric} values after filtering")
        return None
    
    sort_col = 'CapyMOA' if 'CapyMOA' in pivot_avg.columns else (pivot_avg.columns[0] if len(pivot_avg.columns) else None)
    if sort_col and sort_col in pivot_avg.columns:
        # Trier en gérant les NaN (les mettre à la fin)
        pivot_avg = pivot_avg.sort_values(by=sort_col, ascending=False, na_position='last')
    
    x = np.arange(len(pivot_avg))
    width = 0.35
    
    ax1 = axes[0]
    # Dessiner barres avec fallback: si un modèle n'existe que dans une lib, barre centrée
    capy_present = 'CapyMOA' in pivot_avg.columns
    river_present = 'River' in pivot_avg.columns
    capy_vals = pivot_avg['CapyMOA'].values if capy_present else np.array([np.nan] * len(pivot_avg))
    river_vals = pivot_avg['River'].values if river_present else np.array([np.nan] * len(pivot_avg))
    capy_bars = []
    river_bars = []
    for i in range(len(pivot_avg)):
        c = capy_vals[i]
        r = river_vals[i]
        c_ok = not pd.isna(c)
        r_ok = not pd.isna(r)
        if c_ok and r_ok:
            capy_bars.append(ax1.bar(i - width/2, c, width, label='CapyMOA', alpha=0.8, color=LIB_COLORS["CapyMOA"]))
            river_bars.append(ax1.bar(i + width/2, r, width, label='River', alpha=0.8, color=LIB_COLORS["River"]))
        elif c_ok:
            capy_bars.append(ax1.bar(i, c, width, label='CapyMOA', alpha=0.8, color=LIB_COLORS["CapyMOA"]))
        elif r_ok:
            river_bars.append(ax1.bar(i, r, width, label='River', alpha=0.8, color=LIB_COLORS["River"]))
    # Légende unique (une seule fois)
    from matplotlib.patches import Patch
    handles = []
    if capy_present and not np.all(pd.isna(capy_vals)):
        handles.append(Patch(color=LIB_COLORS["CapyMOA"], label='CapyMOA'))
    if river_present and not np.all(pd.isna(river_vals)):
        handles.append(Patch(color=LIB_COLORS["River"], label='River'))
    if handles:
        ax1.legend(handles=handles, loc='best')
    
    ax1.set_xlabel('Model')
    ax1.set_ylabel(f'{metric} (average)')
    ax1.set_title(f'{metric} Comparison by Model - {task_name}')
    ax1.set_xticks(x)
    ax1.set_xticklabels(pivot_avg.index, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Graphique 2: Box plot par bibliothèque
    ax2 = axes[1]
    libraries = comp_df['Library'].unique()
    # Trier les bibliothèques pour garantir un ordre cohérent
    libraries_sorted = sorted(libraries, key=lambda x: ('CapyMOA', 'River').index(x) if x in ('CapyMOA', 'River') else 999)
    data_to_plot = [comp_df[comp_df['Library'] == lib]['Value'].values for lib in libraries_sorted]
    
    bp = ax2.boxplot(data_to_plot, tick_labels=libraries_sorted, patch_artist=True)
    # Assigner les couleurs dans l'ordre cohérent avec le premier graphique
    for i, patch in enumerate(bp['boxes']):
        lib_name = libraries_sorted[i]
        color = LIB_COLORS.get(lib_name, '#808080')  # Gris par défaut si bibliothèque inconnue
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel(metric)
    ax2.set_title(f'{metric} Distribution by Library - {task_name}')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_metric_evolution(
    task_data: dict[str, pd.DataFrame],
    metric: str,
    task_name: str
) -> plt.Figure | None:
    """Trace l'évolution d'une métrique au fil du temps."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Graphique 1: Évolution moyenne
    ax1 = axes[0]
    
    for library, df in task_data.items():
        if metric not in df.columns:
            continue
        
        # Grouper par step et calculer la moyenne
        evolution = df.groupby('step')[metric].mean().reset_index()
        color = LIB_COLORS.get(library, None)
        ax1.plot(
            evolution['step'],
            evolution[metric],
            label=library,
            linewidth=2,
            marker='o',
            markersize=4,
            color=color
        )
    
    ax1.set_xlabel('Step')
    ax1.set_ylabel(metric)
    ax1.set_title(f'{metric} Evolution - {task_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Graphique 2: Évolution par dataset (si plusieurs datasets)
    ax2 = axes[1]
    
    datasets = set()
    for df in task_data.values():
        if 'dataset' in df.columns:
            datasets.update(df['dataset'].unique())
    
    if len(datasets) > 1:
        for library, df in task_data.items():
            if metric not in df.columns:
                continue
            
            for dataset in datasets:
                dataset_df = df[df['dataset'] == dataset]
                if len(dataset_df) > 0:
                    evolution = dataset_df.groupby('step')[metric].mean().reset_index()
                    ax2.plot(evolution['step'], evolution[metric], 
                            label=f'{library} - {dataset}', linewidth=1.5, alpha=0.7)
        
        ax2.set_xlabel('Step')
        ax2.set_ylabel(metric)
        ax2.set_title(f'{metric} Evolution by Dataset - {task_name}')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.axis('off')
        ax2.text(0.5, 0.5, 'Single dataset', ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    return fig


def compare_time_memory(
    final_metrics: dict[str, pd.DataFrame],
    task_name: str
) -> plt.Figure | None:
    """Compare le temps d'exécution et la mémoire utilisée."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Préparer les données
    time_data = []
    memory_data = []
    
    for library, df in final_metrics.items():
        if 'Time in s' in df.columns:
            for _, row in df.iterrows():
                time_data.append({
                    'Library': library,
                    'Model': row.get('model_normalized', row.get('model', 'Unknown')),
                    'Time': row['Time in s']
                })
        
        if 'Memory in Mb' in df.columns:
            for _, row in df.iterrows():
                memory_data.append({
                    'Library': library,
                    'Model': row.get('model_normalized', row.get('model', 'Unknown')),
                    'Memory': row['Memory in Mb']
                })
    
    # Graphique 1: Temps d'exécution
    if time_data:
        time_df = pd.DataFrame(time_data)
        time_avg = time_df.groupby(['Library', 'Model'])['Time'].mean().reset_index()
        
        pivot_time = time_avg.pivot(index='Model', columns='Library', values='Time')
        if len(pivot_time) > 0:
            sort_col = 'CapyMOA' if 'CapyMOA' in pivot_time.columns else (pivot_time.columns[0] if len(pivot_time.columns) else None)
            if sort_col:
                pivot_time = pivot_time.sort_values(by=sort_col, ascending=False).head(20)
            
            x = np.arange(len(pivot_time))
            width = 0.35
            
            ax1 = axes[0]
            capy_present = 'CapyMOA' in pivot_time.columns
            river_present = 'River' in pivot_time.columns
            capy_vals = pivot_time['CapyMOA'].values if capy_present else np.array([np.nan] * len(pivot_time))
            river_vals = pivot_time['River'].values if river_present else np.array([np.nan] * len(pivot_time))
            capy_bars = []
            river_bars = []
            for i in range(len(pivot_time)):
                c = capy_vals[i]
                r = river_vals[i]
                c_ok = not pd.isna(c)
                r_ok = not pd.isna(r)
                if c_ok and r_ok:
                    capy_bars.append(ax1.barh(i - width/2, c, width, label='CapyMOA', alpha=0.8, color=LIB_COLORS["CapyMOA"]))
                    river_bars.append(ax1.barh(i + width/2, r, width, label='River', alpha=0.8, color=LIB_COLORS["River"]))
                elif c_ok:
                    capy_bars.append(ax1.barh(i, c, width, label='CapyMOA', alpha=0.8, color=LIB_COLORS["CapyMOA"]))
                elif r_ok:
                    river_bars.append(ax1.barh(i, r, width, label='River', alpha=0.8, color=LIB_COLORS["River"]))
            
            ax1.set_xlabel('Time (seconds)')
            ax1.set_ylabel('Model')
            ax1.set_title(f'Execution Time Comparison - {task_name}')
            ax1.set_yticks(x)
            ax1.set_yticklabels(pivot_time.index)
            # Légende unique
            from matplotlib.patches import Patch
            handles = []
            if capy_present and not np.all(pd.isna(capy_vals)):
                handles.append(Patch(color=LIB_COLORS["CapyMOA"], label='CapyMOA'))
            if river_present and not np.all(pd.isna(river_vals)):
                handles.append(Patch(color=LIB_COLORS["River"], label='River'))
            if handles:
                ax1.legend(handles=handles, loc='best')
            ax1.grid(True, alpha=0.3)
    
    # Graphique 2: Mémoire
    if memory_data:
        memory_df = pd.DataFrame(memory_data)
        memory_avg = memory_df.groupby(['Library', 'Model'])['Memory'].mean().reset_index()
        
        pivot_memory = memory_avg.pivot(index='Model', columns='Library', values='Memory')
        if len(pivot_memory) > 0:
            sort_col = 'CapyMOA' if 'CapyMOA' in pivot_memory.columns else (pivot_memory.columns[0] if len(pivot_memory.columns) else None)
            if sort_col:
                pivot_memory = pivot_memory.sort_values(by=sort_col, ascending=False).head(20)
            
            x = np.arange(len(pivot_memory))
            width = 0.35
            
            ax2 = axes[1]
            capy_present = 'CapyMOA' in pivot_memory.columns
            river_present = 'River' in pivot_memory.columns
            capy_vals = pivot_memory['CapyMOA'].values if capy_present else np.array([np.nan] * len(pivot_memory))
            river_vals = pivot_memory['River'].values if river_present else np.array([np.nan] * len(pivot_memory))
            capy_bars = []
            river_bars = []
            for i in range(len(pivot_memory)):
                c = capy_vals[i]
                r = river_vals[i]
                c_ok = not pd.isna(c)
                r_ok = not pd.isna(r)
                if c_ok and r_ok:
                    capy_bars.append(ax2.barh(i - width/2, c, width, label='CapyMOA', alpha=0.8, color=LIB_COLORS["CapyMOA"]))
                    river_bars.append(ax2.barh(i + width/2, r, width, label='River', alpha=0.8, color=LIB_COLORS["River"]))
                elif c_ok:
                    capy_bars.append(ax2.barh(i, c, width, label='CapyMOA', alpha=0.8, color=LIB_COLORS["CapyMOA"]))
                elif r_ok:
                    river_bars.append(ax2.barh(i, r, width, label='River', alpha=0.8, color=LIB_COLORS["River"]))
            
            ax2.set_xlabel('Memory (MB)')
            ax2.set_ylabel('Model')
            ax2.set_title(f'Memory Usage Comparison - {task_name}')
            ax2.set_yticks(x)
            ax2.set_yticklabels(pivot_memory.index)
            # Légende unique
            from matplotlib.patches import Patch
            handles = []
            if capy_present and not np.all(pd.isna(capy_vals)):
                handles.append(Patch(color=LIB_COLORS["CapyMOA"], label='CapyMOA'))
            if river_present and not np.all(pd.isna(river_vals)):
                handles.append(Patch(color=LIB_COLORS["River"], label='River'))
            if handles:
                ax2.legend(handles=handles, loc='best')
            ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_summary_table(
    final_metrics: dict[str, pd.DataFrame],
    metrics: list[str],
    task_name: str
) -> plt.Figure | None:
    """Crée un tableau récapitulatif des performances."""
    
    # Préparer les données pour le tableau
    summary_rows = []
    
    for library, df in final_metrics.items():
        for _, row in df.iterrows():
            # Utiliser le nom original du modèle, pas le normalisé, pour éviter les confusions
            model = row.get('model', 'Unknown')
            dataset = row.get('dataset', 'Unknown')
            
            row_data = {
                'Library': library,
                'Model': model,
                'Dataset': dataset
            }
            
            for metric in metrics:
                if metric in row:
                    row_data[metric] = f"{row[metric]:.4f}" if pd.notna(row[metric]) else "N/A"
                else:
                    row_data[metric] = "N/A"
            
            if 'Time in s' in row:
                row_data['Time (s)'] = f"{row['Time in s']:.2f}" if pd.notna(row['Time in s']) else "N/A"
            else:
                row_data['Time (s)'] = "N/A"
            
            if 'Memory in Mb' in row:
                row_data['Memory (MB)'] = f"{row['Memory in Mb']:.2f}" if pd.notna(row['Memory in Mb']) else "N/A"
            else:
                row_data['Memory (MB)'] = "N/A"
            
            summary_rows.append(row_data)
    
    if not summary_rows:
        return None
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Créer la figure avec le tableau
    fig, ax = plt.subplots(figsize=(20, max(8, len(summary_df) * 0.3)))
    ax.axis('tight')
    ax.axis('off')
    
    # Créer le tableau
    table = ax.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    
    # Style du tableau
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Colorier les lignes selon la bibliothèque (CapyMOA = bleu clair, River = orange clair)
    for i in range(1, len(summary_df) + 1):
        library = summary_df.iloc[i-1]['Library']
        row_color = '#e3f2fd' if library == 'CapyMOA' else '#fff3e0'  # Bleu très clair / Orange très clair
        for j in range(len(summary_df.columns)):
            table[(i, j)].set_facecolor(row_color)
    
    plt.title(f'Summary Table - {task_name}', fontsize=16, fontweight='bold', pad=20)
    
    return fig


def create_comparison_heatmap(
    final_metrics: dict[str, pd.DataFrame],
    metrics: list[str],
    task_name: str
) -> plt.Figure | None:
    """Crée une heatmap de comparaison."""
    
    # Préparer les données
    comparison_data = []
    
    for library, df in final_metrics.items():
        for _, row in df.iterrows():
            model = row.get('model_normalized', row.get('model', 'Unknown'))
            
            for metric in metrics:
                if metric in row and pd.notna(row[metric]):
                    comparison_data.append({
                        'Library': library,
                        'Model': model,
                        'Metric': metric,
                        'Value': row[metric]
                    })
    
    if not comparison_data:
        return None
    
    comp_df = pd.DataFrame(comparison_data)
    
    # Créer une matrice de comparaison
    pivot_data = comp_df.pivot_table(
        index=['Model', 'Metric'],
        columns='Library',
        values='Value',
        aggfunc='mean'
    )
    
    # Normaliser les valeurs pour la heatmap (0-1)
    pivot_normalized = pivot_data.copy()
    for col in pivot_normalized.columns:
        col_min = pivot_normalized[col].min()
        col_max = pivot_normalized[col].max()
        if col_max > col_min:
            pivot_normalized[col] = (pivot_normalized[col] - col_min) / (col_max - col_min)
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=(12, max(8, len(pivot_normalized) * 0.3)))
    
    sns.heatmap(
        pivot_normalized,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        center=0.5,
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Normalized Value'},
        ax=ax
    )
    
    ax.set_title(f'Performance Heatmap - {task_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Library')
    ax.set_ylabel('Model - Metric')
    
    plt.tight_layout()
    return fig


def create_summary_csv(
    final_metrics: dict[str, pd.DataFrame],
    metrics: list[str],
    task_name: str,
    output_dir: Path
) -> Path | None:
    """Crée un CSV récapitulatif."""
    
    summary_rows = []
    
    for library, df in final_metrics.items():
        for _, row in df.iterrows():
            summary_row = {
                'Library': library,
                'Model': row.get('model_normalized', row.get('model', 'Unknown')),
                'Dataset': row.get('dataset', 'Unknown'),
            }
            
            for metric in metrics:
                if metric in row:
                    summary_row[metric] = row[metric] if pd.notna(row[metric]) else None
                else:
                    summary_row[metric] = None
            
            if 'Time in s' in row:
                summary_row['Time_s'] = row['Time in s'] if pd.notna(row['Time in s']) else None
            else:
                summary_row['Time_s'] = None
            
            if 'Memory in Mb' in row:
                summary_row['Memory_MB'] = row['Memory in Mb'] if pd.notna(row['Memory in Mb']) else None
            else:
                summary_row['Memory_MB'] = None
            
            summary_rows.append(summary_row)
    
    if not summary_rows:
        return None
    
    summary_df = pd.DataFrame(summary_rows)
    csv_path = output_dir / f"benchmark_summary_{task_name.lower().replace(' ', '_')}.csv"
    summary_df.to_csv(csv_path, index=False)
    
    return csv_path


def generate_time_memory_by_dataset(
    task_name: str,
    task_data: dict[str, pd.DataFrame],
    final_metrics: dict[str, pd.DataFrame],
    output_dir: Path
) -> None:
    """Génère des visualisations de temps et mémoire par dataset pour chaque type de problème.
    
    Crée des graphiques en barres horizontales pour:
    - Temps de calcul moyen par bibliothèque
    - Mémoire moyenne par bibliothèque
    
    Pour chaque dataset et chaque type de problème (classification binaire, multiclass, régression, détection d'anomalies).
    """
    
    # Récupérer tous les datasets uniques
    datasets = set()
    for df in task_data.values():
        if 'dataset' in df.columns:
            datasets.update(df['dataset'].unique())
    
    if not datasets:
        logger.warning(f"No datasets found for {task_name}, skipping time/memory by dataset visualizations")
        return
    
    # Créer un PDF pour cette tâche avec les visualisations par dataset
    pdf_path = output_dir / f"time_memory_by_dataset_{task_name.lower().replace(' ', '_')}.pdf"
    task_safe_name = task_name.lower().replace(' ', '_')
    
    with PdfPages(pdf_path) as pdf:
        for dataset in sorted(datasets):
            logger.info(f"Generating time/memory plots for {task_name} - {dataset}")
            
            # Filtrer les données pour ce dataset
            dataset_final_metrics = {}
            for library, df in final_metrics.items():
                dataset_df = df[df['dataset'] == dataset].copy()
                if len(dataset_df) > 0:
                    dataset_final_metrics[library] = dataset_df
            
            if not dataset_final_metrics:
                continue
            
            # Générer les graphiques pour ce dataset
            fig = plot_time_memory_by_library(dataset_final_metrics, task_name, dataset)
            if fig:
                pdf.savefig(fig, bbox_inches='tight')
                # Sauvegarder aussi en PNG avec un nom de fichier approprié
                dataset_safe = dataset.lower().replace(' ', '_').replace('/', '_')
                png_path = output_dir / f"{task_safe_name}_time_memory_{dataset_safe}.png"
                fig.savefig(png_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved PNG: {png_path}")
                plt.close(fig)
    
    logger.info(f"Generated time/memory by dataset PDF: {pdf_path}")


def plot_time_memory_by_library(
    final_metrics: dict[str, pd.DataFrame],
    task_name: str,
    dataset: str
) -> plt.Figure | None:
    """Crée des graphiques en barres horizontales pour le temps et la mémoire par bibliothèque.
    
    Args:
        final_metrics: Dictionnaire {library: DataFrame} avec les métriques finales
        task_name: Nom de la tâche (ex: "Regression", "Binary Classification", etc.)
        dataset: Nom du dataset
    
    Returns:
        Figure matplotlib avec deux sous-graphiques (temps et mémoire)
    """
    
    # Préparer les données pour le temps
    time_data = []
    memory_data = []
    
    for library, df in final_metrics.items():
        if 'Time in s' in df.columns:
            for _, row in df.iterrows():
                model = row.get('model_normalized', row.get('model', 'Unknown'))
                time_data.append({
                    'Library': library,
                    'Model': model,
                    'Time': row['Time in s']
                })
        
        if 'Memory in Mb' in df.columns:
            for _, row in df.iterrows():
                model = row.get('model_normalized', row.get('model', 'Unknown'))
                memory_data.append({
                    'Library': library,
                    'Model': model,
                    'Memory': row['Memory in Mb']
                })
    
    if not time_data and not memory_data:
        return None
    
    # Créer la figure avec deux sous-graphiques
    fig, axes = plt.subplots(1, 2, figsize=(16, max(8, len(set([d['Model'] for d in time_data + memory_data])) * 0.4)))
    
    # Graphique 1: Temps de calcul moyen par bibliothèque
    if time_data:
        time_df = pd.DataFrame(time_data)
        # Calculer la moyenne par bibliothèque et par modèle
        time_avg = time_df.groupby(['Library', 'Model'])['Time'].mean().reset_index()
        
        # Pivoter pour avoir les bibliothèques en colonnes
        pivot_time = time_avg.pivot(index='Model', columns='Library', values='Time')
        
        if len(pivot_time) > 0:
            # Trier par la première bibliothèque disponible
            sort_col = 'CapyMOA' if 'CapyMOA' in pivot_time.columns else (
                'River' if 'River' in pivot_time.columns else pivot_time.columns[0]
            )
            if sort_col and sort_col in pivot_time.columns:
                pivot_time = pivot_time.sort_values(by=sort_col, ascending=False)
            
            x = np.arange(len(pivot_time))
            width = 0.35
            
            ax1 = axes[0]
            capy_present = 'CapyMOA' in pivot_time.columns
            river_present = 'River' in pivot_time.columns
            
            capy_vals = pivot_time['CapyMOA'].values if capy_present else np.array([np.nan] * len(pivot_time))
            river_vals = pivot_time['River'].values if river_present else np.array([np.nan] * len(pivot_time))
            
            # Dessiner les barres
            for i in range(len(pivot_time)):
                c = capy_vals[i] if capy_present else np.nan
                r = river_vals[i] if river_present else np.nan
                c_ok = not pd.isna(c) if capy_present else False
                r_ok = not pd.isna(r) if river_present else False
                
                if c_ok and r_ok:
                    ax1.barh(i - width/2, c, width, label='CapyMOA' if i == 0 else '', 
                            alpha=0.8, color=LIB_COLORS["CapyMOA"])
                    ax1.barh(i + width/2, r, width, label='River' if i == 0 else '', 
                            alpha=0.8, color=LIB_COLORS["River"])
                elif c_ok:
                    ax1.barh(i, c, width, label='CapyMOA' if i == 0 else '', 
                            alpha=0.8, color=LIB_COLORS["CapyMOA"])
                elif r_ok:
                    ax1.barh(i, r, width, label='River' if i == 0 else '', 
                            alpha=0.8, color=LIB_COLORS["River"])
            
            ax1.set_xlabel('Temps (secondes)', fontsize=12)
            ax1.set_ylabel('Modèles', fontsize=12)
            ax1.set_title(f'Temps de Calcul - {dataset}\n{task_name}', fontsize=14, fontweight='bold')
            ax1.set_yticks(x)
            ax1.set_yticklabels(pivot_time.index, fontsize=9)
            
            # Légende
            from matplotlib.patches import Patch
            handles = []
            if capy_present and not np.all(pd.isna(capy_vals)):
                handles.append(Patch(color=LIB_COLORS["CapyMOA"], label='Librairie CapyMDA'))
            if river_present and not np.all(pd.isna(river_vals)):
                handles.append(Patch(color=LIB_COLORS["River"], label='River'))
            if handles:
                ax1.legend(handles=handles, loc='best', fontsize=10)
            ax1.grid(True, alpha=0.3, axis='x')
        else:
            axes[0].axis('off')
            axes[0].text(0.5, 0.5, 'Pas de données de temps disponibles', 
                        ha='center', va='center', transform=axes[0].transAxes)
    else:
        axes[0].axis('off')
        axes[0].text(0.5, 0.5, 'Pas de données de temps disponibles', 
                    ha='center', va='center', transform=axes[0].transAxes)
    
    # Graphique 2: Mémoire moyenne par bibliothèque
    if memory_data:
        memory_df = pd.DataFrame(memory_data)
        # Calculer la moyenne par bibliothèque et par modèle
        memory_avg = memory_df.groupby(['Library', 'Model'])['Memory'].mean().reset_index()
        
        # Pivoter pour avoir les bibliothèques en colonnes
        pivot_memory = memory_avg.pivot(index='Model', columns='Library', values='Memory')
        
        if len(pivot_memory) > 0:
            # Trier par la première bibliothèque disponible
            sort_col = 'CapyMOA' if 'CapyMOA' in pivot_memory.columns else (
                'River' if 'River' in pivot_memory.columns else pivot_memory.columns[0]
            )
            if sort_col and sort_col in pivot_memory.columns:
                pivot_memory = pivot_memory.sort_values(by=sort_col, ascending=False)
            
            x = np.arange(len(pivot_memory))
            width = 0.35
            
            ax2 = axes[1]
            capy_present = 'CapyMOA' in pivot_memory.columns
            river_present = 'River' in pivot_memory.columns
            
            capy_vals = pivot_memory['CapyMOA'].values if capy_present else np.array([np.nan] * len(pivot_memory))
            river_vals = pivot_memory['River'].values if river_present else np.array([np.nan] * len(pivot_memory))
            
            # Dessiner les barres
            for i in range(len(pivot_memory)):
                c = capy_vals[i] if capy_present else np.nan
                r = river_vals[i] if river_present else np.nan
                c_ok = not pd.isna(c) if capy_present else False
                r_ok = not pd.isna(r) if river_present else False
                
                if c_ok and r_ok:
                    ax2.barh(i - width/2, c, width, label='CapyMOA' if i == 0 else '', 
                            alpha=0.8, color=LIB_COLORS["CapyMOA"])
                    ax2.barh(i + width/2, r, width, label='River' if i == 0 else '', 
                            alpha=0.8, color=LIB_COLORS["River"])
                elif c_ok:
                    ax2.barh(i, c, width, label='CapyMOA' if i == 0 else '', 
                            alpha=0.8, color=LIB_COLORS["CapyMOA"])
                elif r_ok:
                    ax2.barh(i, r, width, label='River' if i == 0 else '', 
                            alpha=0.8, color=LIB_COLORS["River"])
            
            ax2.set_xlabel('Mémoire (MB)', fontsize=12)
            ax2.set_ylabel('Modèles', fontsize=12)
            ax2.set_title(f'Mémoire Moyenne (MB) - {dataset}\n{task_name}', fontsize=14, fontweight='bold')
            ax2.set_yticks(x)
            ax2.set_yticklabels(pivot_memory.index, fontsize=9)
            
            # Légende
            from matplotlib.patches import Patch
            handles = []
            if capy_present and not np.all(pd.isna(capy_vals)):
                handles.append(Patch(color=LIB_COLORS["CapyMOA"], label='Librairie CapyMDA'))
            if river_present and not np.all(pd.isna(river_vals)):
                handles.append(Patch(color=LIB_COLORS["River"], label='River'))
            if handles:
                ax2.legend(handles=handles, loc='best', fontsize=10)
            ax2.grid(True, alpha=0.3, axis='x')
        else:
            axes[1].axis('off')
            axes[1].text(0.5, 0.5, 'Pas de données de mémoire disponibles', 
                        ha='center', va='center', transform=axes[1].transAxes)
    else:
        axes[1].axis('off')
        axes[1].text(0.5, 0.5, 'Pas de données de mémoire disponibles', 
                    ha='center', va='center', transform=axes[1].transAxes)
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Compare CapyMOA and River benchmarks")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help="Directory containing CSV result files (default: current directory)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Output directory for generated figures and tables (default: benchmark_results)"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    logger.info(f"Loading CSV files from {data_dir}")
    csv_files = load_csv_files(data_dir)
    
    if not csv_files:
        logger.error("No CSV files found!")
        return
    
    logger.info(f"Found {len(csv_files)} CSV files")
    
    logger.info(f"Generating comparisons in {output_dir}")
    compare_libraries(csv_files, output_dir)
    
    logger.info("Benchmark comparison complete!")


if __name__ == "__main__":
    main()

