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
            if "Time in s" not in df.columns:
                for cand in ["Time_s", "time_s", "time", "elapsed_s", "Elapsed s"]:
                    if cand in df.columns:
                        df["Time in s"] = df[cand]
                        break
            if "Memory in Mb" not in df.columns:
                for cand in ["Memory_MB", "memory_mb", "memory", "RSS_MB", "rss_mb"]:
                    if cand in df.columns:
                        df["Memory in Mb"] = df[cand]
                        break
            
            key = f"{library}_{task}"
            csv_files[key] = df
            logger.info(f"Loaded {csv_file.name}: {len(df)} rows, library={library}, task={task}")
            
        except Exception as e:
            logger.warning(f"Failed to load {csv_file.name}: {e}")
    
    return csv_files


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


def get_final_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Extrait les métriques finales (dernier checkpoint) pour chaque modèle/dataset."""
    final_metrics = []
    
    for (model, dataset), group in df.groupby(['model', 'dataset']):
        # Prendre le dernier checkpoint (step maximum)
        final_row = group.loc[group['step'].idxmax()]
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
        
        if len(task_data) < 2:
            logger.warning(f"Not enough libraries for {task_name}, skipping...")
            continue
        
        # Normaliser les noms de modèles (en passant la bibliothèque pour éviter les fausses correspondances)
        for library in task_data:
            task_data[library] = normalize_model_names(task_data[library], library)
        
        # Extraire les métriques finales
        final_metrics = {}
        for library, df in task_data.items():
            final_metrics[library] = get_final_metrics(df)
        
        # Générer les visualisations
        generate_task_comparison(task_name, task_data, final_metrics, output_dir)


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
    metrics_to_plot = [m for m in ['Accuracy', 'MAE', 'RMSE', 'MicroF1', 'MacroF1'] if m in common_metrics]
    
    if not metrics_to_plot:
        logger.warning(f"No common metrics found for {task_name}")
        return
    
    # Créer un PDF pour cette tâche
    pdf_path = output_dir / f"benchmark_comparison_{task_name.lower().replace(' ', '_')}.pdf"
    
    with PdfPages(pdf_path) as pdf:
        # 1. Comparaison des métriques finales par modèle
        for metric in metrics_to_plot:
            fig = compare_metric_by_model(final_metrics, metric, task_name)
            if fig:
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
        
        # 2. Évolution des métriques au fil du temps
        for metric in metrics_to_plot:
            fig = plot_metric_evolution(task_data, metric, task_name)
            if fig:
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
        
        # 3. Comparaison temps/mémoire
        fig = compare_time_memory(final_metrics, task_name)
        if fig:
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        
        # 4. Tableau récapitulatif
        fig = create_summary_table(final_metrics, metrics_to_plot, task_name)
        if fig:
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        
        # 5. Heatmap de comparaison
        fig = create_comparison_heatmap(final_metrics, metrics_to_plot, task_name)
        if fig:
            pdf.savefig(fig, bbox_inches='tight')
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
    model_avg = comp_df.groupby(['Library', 'Model'])['Value'].mean().reset_index()
    
    # Pivot pour faciliter la comparaison
    pivot_avg = model_avg.pivot(index='Model', columns='Library', values='Value')
    sort_col = 'CapyMOA' if 'CapyMOA' in pivot_avg.columns else (pivot_avg.columns[0] if len(pivot_avg.columns) else None)
    if sort_col:
        pivot_avg = pivot_avg.sort_values(by=sort_col, ascending=False)
    
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
    data_to_plot = [comp_df[comp_df['Library'] == lib]['Value'].values for lib in libraries]
    
    bp = ax2.boxplot(data_to_plot, tick_labels=libraries, patch_artist=True)
    colors = ['#1f77b4', '#ff7f0e']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
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

