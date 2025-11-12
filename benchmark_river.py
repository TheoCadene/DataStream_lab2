from __future__ import annotations

import copy
import inspect
import itertools
import json
import logging
import multiprocessing
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from river_config import MODELS, N_CHECKPOINTS, TRACKS
from tqdm import tqdm

from river import base, metrics
import warnings

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Filtrer les warnings RuntimeWarning de River (ex: division par zéro dans GaussianScorer)
# Ces warnings sont fréquents avec certains détecteurs d'anomalie mais n'empêchent pas l'exécution
warnings.filterwarnings("ignore", category=RuntimeWarning, module="river.proba.gaussian")
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")


class AnomalyScorerWrapper(base.Classifier):
    """Wrapper pour convertir les scores d'anomalie en prédictions binaires."""
    
    def __init__(self, anomaly_model, threshold=0.5):
        super().__init__()
        self.anomaly_model = anomaly_model
        self.threshold = threshold
        
        # Détecter les signatures des méthodes pour gérer les différents types de détecteurs
        score_sig = inspect.signature(self.anomaly_model.score_one)
        self.score_needs_y = 'y' in score_sig.parameters
        
        learn_sig = inspect.signature(self.anomaly_model.learn_one)
        self.learn_needs_y = 'y' in learn_sig.parameters
    
    def predict_one(self, x):
        """Convertit le score d'anomalie en prédiction binaire."""
        # Certains détecteurs nécessitent y dans score_one, mais on ne l'a pas en prédiction
        # On utilise None ou on essaie sans y
        try:
            if self.score_needs_y:
                # Si score_one nécessite y, on essaie avec y=None ou on utilise une valeur par défaut
                # Mais en prédiction, on n'a pas y, donc on essaie sans
                try:
                    score = self.anomaly_model.score_one(x, y=None)
                except (TypeError, ValueError):
                    # Si ça échoue, on essaie sans y (certains acceptent y optionnel)
                    score = self.anomaly_model.score_one(x)
            else:
                score = self.anomaly_model.score_one(x)
        except Exception as e:
            # Fallback: si score_one échoue, on retourne False (normal)
            logger.warning(f"Error in score_one: {e}, returning False")
            return False
        
        # Si score > threshold, c'est une anomalie (True/1), sinon normal (False/0)
        return score > self.threshold
    
    def predict_proba_one(self, x):
        """Retourne les probabilités pour chaque classe."""
        try:
            if self.score_needs_y:
                try:
                    score = self.anomaly_model.score_one(x, y=None)
                except (TypeError, ValueError):
                    score = self.anomaly_model.score_one(x)
            else:
                score = self.anomaly_model.score_one(x)
        except Exception as e:
            logger.warning(f"Error in score_one for proba: {e}, returning 0.5")
            score = 0.5
        
        # Normaliser le score entre 0 et 1 pour obtenir une probabilité
        # Score élevé = probabilité élevée d'anomalie
        prob_anomaly = min(1.0, max(0.0, score))
        return {True: prob_anomaly, False: 1.0 - prob_anomaly}
    
    def learn_one(self, x, y):
        """Passe l'apprentissage au modèle d'anomalie sous-jacent."""
        if self.learn_needs_y:
            self.anomaly_model.learn_one(x, y)
        else:
            # Certains détecteurs d'anomalie n'acceptent pas y dans learn_one
            self.anomaly_model.learn_one(x)
        return self
    
    def clone(self):
        """Clone le wrapper et le modèle sous-jacent."""
        return AnomalyScorerWrapper(self.anomaly_model.clone(), self.threshold)


class AnomalyTrack:
    """Track personnalisé pour la détection d'anomalie."""
    
    def __init__(self, datasets):
        self.name = "Anomaly detection"
        self.datasets = datasets
    
    def run(self, model, dataset, n_checkpoints=50):
        """Exécute le benchmark pour un modèle d'anomalie."""
        import time
        import sys
        
        # Détecter les signatures des méthodes du modèle
        score_sig = inspect.signature(model.score_one)
        score_needs_y = 'y' in score_sig.parameters
        
        learn_sig = inspect.signature(model.learn_one)
        learn_needs_y = 'y' in learn_sig.parameters
        
        # Détecter si c'est LocalOutlierFactor (très lent sur grands datasets)
        model_name = model.__class__.__name__
        is_lof = 'LocalOutlierFactor' in model_name
        
        # Métriques pour la classification binaire
        accuracy = metrics.Accuracy()
        f1 = metrics.F1()
        
        # Calculer l'intervalle pour les checkpoints
        n_samples = getattr(dataset, 'n_samples', None)
        if n_samples is None:
            # Si n_samples n'est pas disponible, utiliser une valeur par défaut
            n_samples = 10000  # Valeur par défaut raisonnable
        
        # Limiter LocalOutlierFactor à 2000 échantillons pour éviter les temps d'exécution très longs
        max_samples_lof = 2000
        if is_lof and n_samples > max_samples_lof:
            logger.warning(f"LocalOutlierFactor est très lent. Limitation à {max_samples_lof} échantillons (au lieu de {n_samples})")
            n_samples = max_samples_lof
        
        interval = max(1, n_samples // n_checkpoints) if n_samples else 1000
        
        step = 0
        start_time = time.perf_counter()
        last_checkpoint = 0
        threshold = 0.5  # Seuil pour convertir score en prédiction binaire
        
        # Limiter le nombre d'itérations pour LocalOutlierFactor
        max_iterations = n_samples if is_lof and n_samples <= 2000 else None
        
        for x, y_true in dataset:
            # Arrêter après max_iterations pour LOF
            if max_iterations is not None and step >= max_iterations:
                break
            # Obtenir le score d'anomalie
            try:
                if score_needs_y:
                    score = model.score_one(x, y=y_true)
                else:
                    score = model.score_one(x)
            except Exception as e:
                logger.warning(f"Error in score_one at step {step}: {e}")
                score = 0.0  # Valeur par défaut
            
            # Convertir le score en prédiction binaire
            y_pred = score > threshold
            
            # Mettre à jour les métriques
            accuracy.update(y_true, y_pred)
            f1.update(y_true, y_pred)
            
            # Apprendre
            try:
                if learn_needs_y:
                    model.learn_one(x, y_true)
                else:
                    model.learn_one(x)
            except Exception as e:
                logger.warning(f"Error in learn_one at step {step}: {e}")
            
            step += 1
            
            # Checkpoint
            if step - last_checkpoint >= interval or step == n_samples:
                elapsed = time.perf_counter() - start_time
                yield {
                    "Step": step,
                    "Accuracy": accuracy,  # Garder l'objet Metric pour que run_dataset puisse appeler .get()
                    "F1": f1,  # Garder l'objet Metric
                    "Time": timedelta(seconds=elapsed),
                    "Memory": sys.getsizeof(model) if hasattr(sys, 'getsizeof') else 0,
                }
                last_checkpoint = step


class CSVDataset:
    """Wrapper pour utiliser un dataset CSV local avec River."""
    
    def __init__(self, csv_path: Path, name: str):
        self.csv_path = csv_path
        self.name = name
        self._df = pd.read_csv(csv_path)
        if "target" not in self._df.columns:
            raise ValueError(f"CSV {csv_path} must have a 'target' column")
        self.n_samples = len(self._df)
    
    def __iter__(self):
        for _, row in self._df.iterrows():
            y = row["target"]
            x = row.drop(labels=["target"]).to_dict()
            yield x, y
    
    def download(self, verbose: bool = False):
        """Méthode de compatibilité avec les datasets River - pas besoin de télécharger."""
        pass


def _load_local_datasets_if_available() -> dict[str, list[Any]]:
    """Charge les datasets CSV locaux s'ils existent."""
    datasets_dir = Path(__file__).parent / "datasets"
    if not datasets_dir.exists():
        return {}
    
    local_datasets = {}
    
    # Mapping des noms de tracks
    track_mapping = {
        "multiclass_classification": "Multiclass classification",
        "binary_classification": "Binary classification",
        "regression": "Regression",
        "anomaly": "Anomaly detection",
        "anomaly_detection": "Anomaly detection",  # Alias pour compatibilité
    }
    
    for folder_name, track_name in track_mapping.items():
        track_dir = datasets_dir / folder_name
        if track_dir.exists():
            datasets = []
            for csv_file in sorted(track_dir.glob("*.csv")):
                # Ignorer les fichiers d'erreur
                if csv_file.name.endswith("_ERROR.txt"):
                    continue
                try:
                    dataset = CSVDataset(csv_file, csv_file.stem)
                    datasets.append(dataset)
                    logger.info(f"Loaded local dataset: {csv_file.name} for {track_name}")
                except Exception as e:
                    logger.warning(f"Failed to load {csv_file}: {e}")
            if datasets:
                local_datasets[track_name] = datasets
    
    return local_datasets


def _prefetch_all_datasets():
    """Download datasets serially to avoid concurrent extraction errors in workers."""
    local_datasets = _load_local_datasets_if_available()
    
    # Ajouter le track d'anomalie si des datasets sont disponibles
    if "Anomaly detection" in local_datasets:
        anomaly_track = AnomalyTrack(local_datasets["Anomaly detection"])
        TRACKS.append(anomaly_track)
        logger.info(f"Added Anomaly detection track with {len(anomaly_track.datasets)} datasets")
    
    for track in TRACKS:
        track_name = track.name
        # Si on a des datasets locaux pour ce track, on les utilise
        if track_name in local_datasets:
            logger.info(f"Using local datasets for {track_name}")
            track.datasets = local_datasets[track_name]
            continue
        
        # Sinon, essayer de télécharger les datasets River
        for dataset in track.datasets:
            try:
                dataset.download(verbose=False)
            except Exception as e:
                error_msg = str(e)
                if "404" in error_msg or "Not Found" in error_msg:
                    logger.warning(f"Dataset {dataset.__class__.__name__} not available (404), skipping...")
                else:
                    logger.warning(f"Failed to download {dataset.__class__.__name__}: {e}")

def _init_worker():
    """Initialize worker process - reload TRACKS and MODELS from river_config."""
    global TRACKS, MODELS
    # Re-import to get the latest TRACKS and MODELS in each worker process
    from river_config import MODELS as _MODELS, TRACKS as _TRACKS
    MODELS = _MODELS
    TRACKS = list(_TRACKS)  # Make a copy to avoid sharing issues
    
    # Also reload local datasets if available (redefine function locally to avoid import issues)
    datasets_dir = Path(__file__).parent / "datasets"
    if datasets_dir.exists():
        local_datasets = {}
        track_mapping = {
            "multiclass_classification": "Multiclass classification",
            "binary_classification": "Binary classification",
            "regression": "Regression",
            "anomaly": "Anomaly detection",
            "anomaly_detection": "Anomaly detection",
        }
        for folder_name, track_name in track_mapping.items():
            track_dir = datasets_dir / folder_name
            if track_dir.exists():
                datasets = []
                for csv_file in sorted(track_dir.glob("*.csv")):
                    if csv_file.name.endswith("_ERROR.txt"):
                        continue
                    try:
                        dataset = CSVDataset(csv_file, csv_file.stem)
                        datasets.append(dataset)
                    except Exception:
                        pass
                if datasets:
                    local_datasets[track_name] = datasets
        
        # Add anomaly track if datasets available
        if local_datasets and "Anomaly detection" in local_datasets:
            anomaly_track = AnomalyTrack(local_datasets["Anomaly detection"])
            TRACKS.append(anomaly_track)
        # Update track datasets with local ones if available
        for track in TRACKS:
            track_name = track.name
            if track_name in local_datasets:
                track.datasets = local_datasets[track_name]

def run_dataset(model_str, no_dataset, no_track):
    model_name = model_str
    
    # Ensure TRACKS is accessible (should be set by _init_worker)
    if no_track >= len(TRACKS):
        logger.warning(f"Track index {no_track} out of range (max: {len(TRACKS)-1})")
        return []
    
    track = TRACKS[no_track]
    
    # Vérifier que le dataset existe
    if no_dataset >= len(track.datasets):
        logger.warning(f"Dataset index {no_dataset} out of range for track {track.name}")
        return []
    
    dataset = track.datasets[no_dataset]
    
    # Obtenir le nom du dataset
    if isinstance(dataset, CSVDataset):
        dataset_name = dataset.name
    else:
        try:
            dataset_name = dataset.__class__.__name__
        except AttributeError:
            dataset_name = str(dataset)
    
    # Pour les tracks standards River, utiliser clone()
    # Pour le track d'anomalie personnalisé, cloner le modèle
    if track.name == "Anomaly detection":
        # Les modèles d'anomalie doivent être clonés comme les autres
        model = MODELS[track.name][model_name].clone()
    else:
        MODELS["Binary classification"].update(MODELS["Multiclass classification"])
        model = MODELS[track.name][model_name].clone()
    print(f"Processing {model_str} on {dataset_name}")

    results = []
    track = copy.deepcopy(track)
    time = 0.0
    
    try:
        for i in tqdm(
            track.run(model, dataset, n_checkpoints=N_CHECKPOINTS),
            total=N_CHECKPOINTS,
            desc=f"{model_str} on {dataset_name}",
        ):
            time += i["Time"].total_seconds()
            res = {
                "step": i["Step"],
                "track": track.name,
                "model": model_name,
                "dataset": dataset_name,
            }
            for k, v in i.items():
                if isinstance(v, metrics.base.Metric):
                    res[k] = v.get()
            res["Memory in Mb"] = i["Memory"] / 1024**2
            res["Time in s"] = time
            results.append(res)
    except Exception as e:
        logger.error(f"Error running {model_str} on {dataset_name}: {e}")
        # Retourner les résultats partiels si disponibles
        pass
    
    return results


def run_track(models: list[str], no_track: int, n_workers: int = None):
    # Use real processes for true CPU parallelism (not threads limited by GIL)
    # Each process will clone models independently
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=n_workers, initializer=_init_worker)
    track = TRACKS[no_track]
    runs = list(itertools.product(models, range(len(track.datasets)), [no_track]))
    results = []

    try:
        for val in pool.starmap(run_dataset, runs):
            results.extend(val)
    finally:
        pool.close()
        pool.join()
    
    csv_name = track.name.replace(" ", "_").lower()
    pd.DataFrame(results).to_csv(f"./{csv_name}.csv", index=False)


if __name__ == "__main__":
    # Pre-download datasets to prevent EOFError from concurrent unzip in workers
    _prefetch_all_datasets()

    details = {}
    # Create details for each model – run all tracks except those explicitly skipped
    for i, track in enumerate(TRACKS):
        # Skip only Regression if you want to exclude it
        # Remove Binary classification from skip list to generate results
        if track.name == "Multiclass classification":
            continue
        if track.name == "Binary classification":
            continue
        if track.name == "Regression":
            continue
        details[track.name] = {"Dataset": {}, "Model": {}}
        for dataset in track.datasets:
            dataset_name = dataset.name if isinstance(dataset, CSVDataset) else dataset.__class__.__name__
            details[track.name]["Dataset"][dataset_name] = repr(dataset)
        for model_name, model in MODELS[track.name].items():
            details[track.name]["Model"][model_name] = repr(model)
        with open("details.json", "w") as f:
            json.dump(details, f, indent=2)
        run_track(models=MODELS[track.name].keys(), no_track=i)