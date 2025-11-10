from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from river_config import TRACKS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_dataset(dataset: Any, dataset_name: str, track_name: str, output_dir: Path) -> None:
    """Sauvegarde un dataset River en format CSV local.
    
    Args:
        dataset: Dataset River (itérable de (x, y))
        dataset_name: Nom du dataset
        track_name: Nom du track (pour organisation)
        output_dir: Répertoire de sortie
    """
    # Créer le répertoire pour ce track
    track_dir = output_dir / track_name.replace(" ", "_").lower()
    track_dir.mkdir(parents=True, exist_ok=True)
    
    # Nom du fichier
    safe_name = dataset_name.replace(" ", "_").replace("/", "_").lower()
    output_path = track_dir / f"{safe_name}.csv"
    
    # Si le fichier existe déjà, on skip (pour éviter de re-télécharger)
    if output_path.exists():
        logger.info(f"Dataset {dataset_name} déjà sauvegardé, skip...")
        return
    
    logger.info(f"Sauvegarde de {dataset_name} ({track_name})...")
    
    # Collecter toutes les données
    rows = []
    feature_names = None
    
    try:
        # Itérer sur le dataset
        for x, y in tqdm(dataset, desc=f"Collecting {dataset_name}", leave=False):
            # x est un dict de features, y est la target
            row = dict(x)
            row["target"] = y
            
            # Stocker les noms de features au premier passage
            if feature_names is None:
                feature_names = list(x.keys())
            
            rows.append(row)
        
        # Créer un DataFrame
        if not rows:
            logger.warning(f"Dataset {dataset_name} est vide!")
            return
        
        df = pd.DataFrame(rows)
        
        # Réorganiser les colonnes: features d'abord, target à la fin
        cols = [c for c in df.columns if c != "target"] + ["target"]
        df = df[cols]
        
        # Sauvegarder en CSV
        df.to_csv(output_path, index=False)
        logger.info(f"✓ Sauvegardé: {output_path} ({len(df)} lignes, {len(df.columns)-1} features)")
        
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde de {dataset_name}: {e}")
        # Sauvegarder un fichier vide ou avec erreur pour indiquer le problème
        error_path = track_dir / f"{safe_name}_ERROR.txt"
        with open(error_path, "w") as f:
            f.write(f"Erreur lors de la sauvegarde: {e}\n")


def save_all_datasets(output_dir: str = "./datasets") -> None:
    """Sauvegarde tous les datasets de tous les tracks.
    
    Args:
        output_dir: Répertoire de sortie pour les datasets
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Sauvegarde de tous les datasets dans {output_path.absolute()}")
    
    # Télécharger tous les datasets d'abord
    logger.info("Téléchargement des datasets...")
    for track in TRACKS:
        for dataset in track.datasets:
            try:
                dataset.download(verbose=False)
            except Exception as e:
                logger.warning(f"Erreur lors du téléchargement de {dataset.__class__.__name__}: {e}")
    
    # Sauvegarder chaque dataset
    total_datasets = sum(len(track.datasets) for track in TRACKS)
    logger.info(f"Traitement de {total_datasets} datasets...")
    
    for track in TRACKS:
        track_name = track.name
        logger.info(f"\n{'='*60}")
        logger.info(f"Track: {track_name}")
        logger.info(f"{'='*60}")
        
        for dataset in track.datasets:
            dataset_name = dataset.__class__.__name__
            try:
                # Créer une copie du dataset pour éviter les problèmes de réutilisation
                # Certains datasets peuvent être des générateurs qui se consomment
                save_dataset(dataset, dataset_name, track_name, output_path)
            except Exception as e:
                logger.error(f"Erreur avec {dataset_name}: {e}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Terminé! Datasets sauvegardés dans {output_path.absolute()}")
    
    # Créer un fichier récapitulatif
    summary_path = output_path / "datasets_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Résumé des datasets sauvegardés\n")
        f.write("=" * 60 + "\n\n")
        for track in TRACKS:
            f.write(f"Track: {track.name}\n")
            f.write("-" * 60 + "\n")
            for dataset in track.datasets:
                dataset_name = dataset.__class__.__name__
                safe_name = dataset_name.replace(" ", "_").replace("/", "_").lower()
                track_dir_name = track.name.replace(" ", "_").lower()
                file_path = output_path / track_dir_name / f"{safe_name}.csv"
                if file_path.exists():
                    df = pd.read_csv(file_path)
                    f.write(f"  - {dataset_name}: {file_path.name} ({len(df)} lignes, {len(df.columns)-1} features)\n")
                else:
                    f.write(f"  - {dataset_name}: NON SAUVEGARDÉ\n")
            f.write("\n")
    
    logger.info(f"Résumé sauvegardé dans {summary_path}")


if __name__ == "__main__":
    save_all_datasets()

