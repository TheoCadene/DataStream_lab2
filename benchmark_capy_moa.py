from __future__ import annotations

import itertools
import json
import logging
import multiprocessing
import time
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd
from tqdm import tqdm

from river import metrics

from capy_moa_config import MODELS, N_CHECKPOINTS, TRACKS, CapyMoaAdapter

try:
	import psutil  # optional, for memory usage
except Exception:  # pragma: no cover
	psutil = None

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


@dataclass
class CheckpointResult:
	step: int
	track: str
	model: str
	dataset: str
	metrics: dict[str, float]
	memory_mb: float
	elapsed_s: float


def _iterate_dataset(dataset: Any) -> Iterable[tuple[dict[str, Any], Any]]:
	# River datasets are iterable (x, y); forward directly
	for x, y in dataset:
		yield x, y


def _get_memory_mb() -> float:
	if psutil is None:
		return 0.0
	process = psutil.Process()
	return float(process.memory_info().rss) / (1024**2)


def run_dataset(model_str: str, no_dataset: int, no_track: int) -> list[dict[str, Any]]:
	track = TRACKS[no_track]
	dataset = track["datasets"][no_dataset]
	track_name = track["name"]
	model_factory = MODELS[track_name][model_str]
	model = model_factory()
	adapter = CapyMoaAdapter(model, classes=[False, True] if "classification" in track_name.lower() else None)
	print(f"Processing {model_str} on {dataset.__class__.__name__}")

	results: list[dict[str, Any]] = []
	# Choose metrics
	metric_objs: list[metrics.base.Metric] = [m.copy() for m in track["metrics"]]
	metric_names = [m.__class__.__name__ for m in metric_objs]

	# Determine checkpoints
	total = getattr(dataset, "n_samples", None)
	if total is not None and total > 0:
		interval = max(1, int((total + N_CHECKPOINTS - 1) // N_CHECKPOINTS))
	else:
		interval = 1000  # fallback chunk size for unknown-length streams

	step = 0
	last_checkpoint_step = 0
	start_time = time.perf_counter()

	for x, y in tqdm(_iterate_dataset(dataset), desc=f"{model_str} on {dataset.__class__.__name__}"):
		# Predict, update metrics, then learn (prequential evaluation)
		try:
			y_pred = adapter.predict_one(x)
		except Exception:
			y_pred = None
		for m in metric_objs:
			try:
				m.update(y, y_pred)
			except Exception:
				pass
		adapter.learn_one(x, y)

		step += 1
		if step - last_checkpoint_step >= interval:
			elapsed = time.perf_counter() - start_time
			mem_mb = _get_memory_mb()
			row = {
				"step": step,
				"track": track_name,
				"model": model_str,
				"dataset": dataset.__class__.__name__,
				"Memory in Mb": mem_mb,
				"Time in s": elapsed,
			}
			for name, m in zip(metric_names, metric_objs):
				try:
					row[name] = m.get()
				except Exception:
					pass
			results.append(row)
			last_checkpoint_step = step

	# Ensure we emit a final checkpoint if none were produced
	if not results:
		elapsed = time.perf_counter() - start_time
		mem_mb = _get_memory_mb()
		row = {
			"step": step,
			"track": track_name,
			"model": model_str,
			"dataset": dataset.__class__.__name__,
			"Memory in Mb": mem_mb,
			"Time in s": elapsed,
		}
		for name, m in zip(metric_names, metric_objs):
			try:
				row[name] = m.get()
			except Exception:
				pass
		results.append(row)

	return results


def run_track(models: list[str], no_track: int, n_workers: int = 50) -> None:
	pool = multiprocessing.Pool(processes=n_workers)
	track = TRACKS[no_track]
	runs = list(itertools.product(models, range(len(track["datasets"])), [no_track]))
	results: list[dict[str, Any]] = []

	for val in pool.starmap(run_dataset, runs):
		results.extend(val)
	csv_name = track["name"].replace(" ", "_").lower()
	pd.DataFrame(results).to_csv(f"./capymoa_{csv_name}.csv", index=False)


if __name__ == "__main__":
	# Merge binary and multiclass models for convenience, mirroring the River script behavior
	combined_classification = {**MODELS.get("Binary classification", {}), **MODELS.get("Multiclass classification", {})}

	details: dict[str, dict[str, dict[str, str]]] = {}
	for i, track in enumerate(TRACKS):
		track_name = track["name"]
		if "classification" in track_name.lower():
			model_bank = combined_classification
		else:
			model_bank = MODELS.get("Regression", {})

		details[track_name] = {"Dataset": {}, "Model": {}}
		for dataset in track["datasets"]:
			details[track_name]["Dataset"][dataset.__class__.__name__] = repr(dataset)
		for model_name, factory in model_bank.items():
			try:
				instance = factory()
				details[track_name]["Model"][model_name] = repr(instance)
			except Exception:
				details[track_name]["Model"][model_name] = "<construction failed>"

		with open("capymoa_details.json", "w") as f:
			json.dump(details, f, indent=2)

		run_track(models=list(model_bank.keys()), no_track=i, n_workers=50)
