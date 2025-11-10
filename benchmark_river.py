from __future__ import annotations

import copy
import itertools
import json
import logging
import multiprocessing
import multiprocessing.dummy as mp_threads
from pathlib import Path
from typing import Any

import pandas as pd
from river_config import MODELS, N_CHECKPOINTS, TRACKS
from tqdm import tqdm

from river import metrics

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


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

def run_dataset(model_str, no_dataset, no_track):
    model_name = model_str
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


def run_track(models: list[str], no_track: int, n_workers: int = 50):
    # Use threads to avoid pickling issues when returning results/exceptions
    pool = mp_threads.Pool(processes=n_workers)
    track = TRACKS[no_track]
    runs = list(itertools.product(models, range(len(track.datasets)), [no_track]))
    results = []

    for val in pool.starmap(run_dataset, runs):
        results.extend(val)
    csv_name = track.name.replace(" ", "_").lower()
    pd.DataFrame(results).to_csv(f"./{csv_name}.csv", index=False)


if __name__ == "__main__":
    # Pre-download datasets to prevent EOFError from concurrent unzip in workers
    _prefetch_all_datasets()

    details = {}
    # Create details for each model – run ONLY Regression track
    for i, track in enumerate(TRACKS):
        if track.name == "Regression" or track.name == "Binary classification":
            continue
        details[track.name] = {"Dataset": {}, "Model": {}}
        for dataset in track.datasets:
            dataset_name = dataset.name if isinstance(dataset, CSVDataset) else dataset.__class__.__name__
            details[track.name]["Dataset"][dataset_name] = repr(dataset)
        for model_name, model in MODELS[track.name].items():
            details[track.name]["Model"][model_name] = repr(model)
        with open("details.json", "w") as f:
            json.dump(details, f, indent=2)
        run_track(models=MODELS[track.name].keys(), no_track=i, n_workers=50)