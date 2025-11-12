from __future__ import annotations

import logging
import io
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# River can be unavailable in some environments; guard the import
try:
    from river import datasets as river_datasets
    RIVER_AVAILABLE = True
except Exception as _e:
    RIVER_AVAILABLE = False
    _RIVER_IMPORT_ERROR = _e

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# EXTERNAL ANOMALY DATASETS
# =========================
def _ensure_anomaly_dir(output_root: Path) -> Path:
    """Ensure anomaly directory exists and return its path."""
    anomaly_dir = output_root / "anomaly"
    anomaly_dir.mkdir(parents=True, exist_ok=True)
    return anomaly_dir

def download_nab_nyc_taxi(output_root: Path) -> Path | None:
    """Download NAB (NYC Taxi) and build a binary CSV with columns [value, target].
    
    target = 1 if timestamp falls inside any labeled anomaly window, else 0.
    """
    import io
    import zipfile
    from urllib.request import urlopen
    import json as _json

    anomaly_dir = _ensure_anomaly_dir(output_root)
    out_csv = anomaly_dir / "nab_nyc_taxi.csv"
    if out_csv.exists():
        logger.info(f"NAB NYC Taxi déjà présent: {out_csv.name}")
        return out_csv

    logger.info("Téléchargement NAB (NYC Taxi) depuis GitHub...")
    url = "https://github.com/numenta/NAB/archive/refs/heads/master.zip"
    try:
        with urlopen(url) as resp:
            data = resp.read()
    except Exception as e:
        logger.warning(f"Échec download NAB: {e}")
        return None

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            # Paths inside the zip
            csv_member = None
            labels_member = None
            for name in zf.namelist():
                if name.endswith("data/realKnownCause/nyc_taxi.csv"):
                    csv_member = name
                if name.endswith("labels/combined_windows.json"):
                    labels_member = name
            if csv_member is None or labels_member is None:
                logger.warning("Fichiers requis introuvables dans NAB zip")
                return None
            with zf.open(csv_member) as f_csv, zf.open(labels_member) as f_lbl:
                df = pd.read_csv(f_csv, parse_dates=["timestamp"])
                windows = _json.load(io.TextIOWrapper(f_lbl, encoding="utf-8"))
                key = "realKnownCause/nyc_taxi.csv"
                win_list = windows.get(key, [])
                # Build boolean mask
                def is_anomaly(ts):
                    for s, e in win_list:
                        if pd.to_datetime(s) <= ts <= pd.to_datetime(e):
                            return 1
                    return 0
                df["target"] = df["timestamp"].apply(is_anomaly)
                # Keep numeric feature(s) only: 'value'
                out = df[["value", "target"]].copy()
                out.to_csv(out_csv, index=False)
                logger.info(f"✓ Sauvegardé: {out_csv} ({len(out)} lignes, 1 feature)")
                return out_csv
    except Exception as e:
        logger.error(f"Erreur traitement NAB: {e}")
        return None

def download_kdd99_10pct_binary(output_root: Path) -> Path | None:
    """Download KDD Cup 1999 10% and create a binary anomaly CSV.
    
    - target = 1 if label != 'normal.' else 0
    - Keep only numeric columns (drop 3 categorical: protocol_type, service, flag)
    """
    import gzip
    import csv as _csv
    from urllib.request import urlopen

    anomaly_dir = _ensure_anomaly_dir(output_root)
    out_csv = anomaly_dir / "kdd99_10pct_binary.csv"
    if out_csv.exists():
        logger.info(f"KDD99 10% déjà présent: {out_csv.name}")
        return out_csv

    url = "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz"
    logger.info("Téléchargement KDD99 10% depuis UCI...")
    try:
        with urlopen(url) as resp:
            gz_data = resp.read()
    except Exception as e:
        logger.warning(f"Échec download KDD99: {e}")
        return None

    # Parse gz content
    try:
        rows = []
        with gzip.GzipFile(fileobj=io.BytesIO(gz_data)) as gz:
            reader = _csv.reader(io.TextIOWrapper(gz, encoding="utf-8", newline=""))
            for r in reader:
                if not r:
                    continue
                # Last column = label
                label = r[-1].strip()
                target = 0 if label == "normal." else 1
                # Drop 3 categorical columns (indices 1,2,3) -> keep numeric
                vals = [v for i, v in enumerate(r[:-1]) if i not in (1, 2, 3)]
                # Convert to float where possible
                feats = []
                for v in vals:
                    try:
                        feats.append(float(v))
                    except Exception:
                        # Rare non-numerics: default to 0.0
                        feats.append(0.0)
                rows.append((*feats, target))

        # Build header: f0..fN, target
        if not rows:
            logger.warning("KDD99 vidé ou non lisible")
            return None
        n_feats = len(rows[0]) - 1
        cols = [f"f{i}" for i in range(n_feats)] + ["target"]
        df = pd.DataFrame(rows, columns=cols)
        df.to_csv(out_csv, index=False)
        logger.info(f"✓ Sauvegardé: {out_csv} ({len(df)} lignes, {n_feats} features)")
        return out_csv
    except Exception as e:
        logger.error(f"Erreur traitement KDD99: {e}")
        return None


def save_dataset(dataset, dataset_name: str, track_name: str, output_dir: Path) -> None:
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
    
    # Télécharger le dataset d'abord
    try:
        dataset.download(verbose=False)
    except Exception as e:
        logger.warning(f"Erreur lors du téléchargement de {dataset_name}: {e}")
        return
    
    # Sauvegarder de manière incrémentale pour éviter les problèmes de mémoire
    import csv
    
    # Pour les datasets avec features variables (comme MaliciousURL), on doit d'abord collecter toutes les features
    # On fait une première passe pour collecter toutes les features possibles
    logger.info(f"  Première passe: collecte des features pour {dataset_name}...")
    all_features = set()
    sample_count = 0
    max_samples_for_feature_collection = 10000  # Limiter pour éviter de tout charger en mémoire
    
    try:
        # Première passe: collecter toutes les features possibles
        for x, y in dataset:
            all_features.update(x.keys())
            sample_count += 1
            if sample_count >= max_samples_for_feature_collection:
                logger.info(f"  {sample_count} échantillons analysés pour collecter les features")
                break
        
        # Si on a collecté toutes les features, continuer à itérer
        # Sinon, on a déjà tout parcouru, il faut réinitialiser le dataset
        if sample_count < max_samples_for_feature_collection:
            # Le dataset est fini, on doit le réinitialiser
            logger.info(f"  Dataset {dataset_name} entièrement parcouru lors de la collecte des features")
            dataset.download(verbose=False)  # Réinitialiser si possible
        
        # Convertir en liste triée pour avoir un ordre cohérent
        feature_names = sorted([f for f in all_features if f != "target"])
        cols = feature_names + ["target"]
        
        logger.info(f"  {len(feature_names)} features uniques trouvées pour {dataset_name}")
        
    except Exception as e:
        logger.warning(f"  Erreur lors de la collecte des features pour {dataset_name}: {e}")
        # Fallback: utiliser les features de la première ligne
        try:
            dataset.download(verbose=False)  # Réinitialiser
            first_x, first_y = next(iter(dataset))
            feature_names = sorted([f for f in first_x.keys() if f != "target"])
            cols = feature_names + ["target"]
        except Exception:
            logger.error(f"  Impossible de récupérer les features pour {dataset_name}")
            raise
    
    # Réinitialiser le dataset pour la deuxième passe
    try:
        dataset.download(verbose=False)
    except Exception:
        pass  # Certains datasets ne peuvent pas être réinitialisés
    
    row_count = 0
    error_count = 0
    
    try:
        # Ouvrir le fichier CSV en mode écriture
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=cols)
            writer.writeheader()
            
            # Deuxième passe: écrire toutes les données
            try:
                for x, y in tqdm(dataset, desc=f"Collecting {dataset_name}", leave=False):
                    try:
                        # x est un dict de features, y est la target
                        row = {col: x.get(col, 0.0) for col in feature_names}  # Utiliser 0.0 pour les features manquantes
                        row["target"] = y
                        
                        # Écrire la ligne directement
                        writer.writerow(row)
                        row_count += 1
                        
                        # Afficher un log tous les 10000 éléments pour suivre la progression
                        if row_count % 10000 == 0:
                            logger.info(f"  {dataset_name}: {row_count} lignes sauvegardées...")
                    
                    except Exception as e:
                        error_count += 1
                        if error_count <= 5:  # Logger seulement les 5 premières erreurs
                            logger.warning(f"  Erreur lors du traitement d'une ligne de {dataset_name}: {e}")
                        continue
                        
            except StopIteration:
                # C'est normal, le dataset est fini
                pass
            except Exception as e:
                logger.error(f"Erreur lors de l'itération sur {dataset_name}: {e}")
                raise
        
        if row_count == 0:
            logger.warning(f"Dataset {dataset_name} est vide!")
            output_path.unlink()  # Supprimer le fichier vide
            return
        
        if error_count > 0:
            logger.warning(f"Dataset {dataset_name}: {error_count} erreurs rencontrées, {row_count} lignes sauvegardées")
        
        logger.info(f"✓ Sauvegardé: {output_path} ({row_count} lignes, {len(feature_names)} features)")
        
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde de {dataset_name}: {e}")
        # Sauvegarder un fichier vide ou avec erreur pour indiquer le problème
        error_path = track_dir / f"{safe_name}_ERROR.txt"
        with open(error_path, "w") as f:
            f.write(f"Erreur lors de la sauvegarde: {e}\n")


def main():
    """Télécharge les datasets spécifiques demandés."""
    output_dir = Path("./datasets")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Téléchargement des datasets spécifiques depuis river.datasets...")
    
    if RIVER_AVAILABLE:
        # Datasets de régression: Taxis et Elec
        regression_datasets = {}
        
        # Charger Taxis directement depuis datasets
        try:
            regression_datasets["Taxis"] = river_datasets.Taxis()
            logger.info("✓ Dataset Taxis chargé")
        except (AttributeError, ImportError) as e:
            logger.warning(f"Dataset Taxis non disponible: {e}")
        
        # Charger Elec directement depuis datasets (essayer différentes variantes)
        try:
            regression_datasets["Elec"] = river_datasets.Elec()
            logger.info("✓ Dataset Elec chargé")
        except (AttributeError, ImportError):
            try:
                regression_datasets["Elec"] = river_datasets.Elec2()
                logger.info("✓ Dataset Elec2 chargé (utilisé comme Elec)")
            except (AttributeError, ImportError) as e:
                logger.warning(f"Dataset Elec/Elec2 non disponible: {e}")
        
        # Dataset de classification multiclasse: Music
        multiclass_datasets = {}
        try:
            multiclass_datasets["Music"] = river_datasets.Music()
            logger.info("✓ Dataset Music chargé")
        except (AttributeError, ImportError) as e:
            logger.warning(f"Dataset Music non disponible: {e}")
        
        # Datasets d'anomalie: CreditCard, TREC07
        anomaly_datasets = {}
        
        # Charger CreditCard
        try:
            anomaly_datasets["CreditCard"] = river_datasets.CreditCard()
            logger.info("✓ Dataset CreditCard chargé")
        except (AttributeError, ImportError) as e:
            logger.warning(f"Dataset CreditCard non disponible: {e}")
        
        # Charger TREC07
        try:
            anomaly_datasets["TREC07"] = river_datasets.TREC07()
            logger.info("✓ Dataset TREC07 chargé")
        except (AttributeError, ImportError) as e:
            logger.warning(f"Dataset TREC07 non disponible: {e}")
        
        # Sauvegarder les datasets de régression
        if regression_datasets:
            logger.info("\n" + "="*60)
            logger.info("Track: Regression")
            logger.info("="*60)
            for dataset_name, dataset in regression_datasets.items():
                try:
                    save_dataset(dataset, dataset_name, "Regression", output_dir)
                except Exception as e:
                    logger.error(f"Erreur avec {dataset_name}: {e}")
        
        # Sauvegarder les datasets de classification multiclasse
        if multiclass_datasets:
            logger.info("\n" + "="*60)
            logger.info("Track: Multiclass classification")
            logger.info("="*60)
            for dataset_name, dataset in multiclass_datasets.items():
                try:
                    save_dataset(dataset, dataset_name, "Multiclass classification", output_dir)
                except Exception as e:
                    logger.error(f"Erreur avec {dataset_name}: {e}")
        
        # Sauvegarder les datasets d'anomalie
        if anomaly_datasets:
            logger.info("\n" + "="*60)
            logger.info("Track: Anomaly detection")
            logger.info("="*60)
            for dataset_name, dataset in anomaly_datasets.items():
                try:
                    # Utiliser "Anomaly" comme nom de track pour créer le dossier "anomaly"
                    save_dataset(dataset, dataset_name, "Anomaly", output_dir)
                except Exception as e:
                    logger.error(f"Erreur avec {dataset_name}: {e}")
    else:
        logger.info("River indisponible - on passe les téléchargements River.")
        logger.debug(f"Raison: {_RIVER_IMPORT_ERROR!r}")
    
    # Datasets d'anomalie externes (hors River)
    logger.info("\n" + "="*60)
    logger.info("Track: Anomaly detection (EXTERNES)")
    logger.info("="*60)
    try:
        download_nab_nyc_taxi(output_dir)
    except Exception as e:
        logger.error(f"Erreur NAB NYC Taxi: {e}")
    try:
        download_kdd99_10pct_binary(output_dir)
    except Exception as e:
        logger.error(f"Erreur KDD99 10%: {e}")
    
    logger.info("\n" + "="*60)
    logger.info("Terminé!")


if __name__ == "__main__":
    main()

