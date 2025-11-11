from __future__ import annotations

import inspect
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

from capy_moa_config import MODELS, N_CHECKPOINTS, TRACKS, CapyMoaAdapter, get_failed_models

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


def _create_metric_instance(metric: metrics.base.Metric) -> metrics.base.Metric:
	"""Create a new instance of a metric (River metrics don't have copy())."""
	# River metrics are typically simple and don't need complex copying
	# Just create a new instance of the same class
	return metric.__class__()


def _coerce_regression_pred_to_float(pred: Any) -> float | None:
	"""Try to coerce various prediction types to a float for regression metrics."""
	# numpy types and arrays
	try:
		import numpy as _np  # local import to avoid global dependency if unused
		if isinstance(pred, _np.ndarray):
			if pred.ndim == 0:
				return float(pred.item())
			if pred.size > 0:
				return float(pred.reshape(-1)[0])
	except Exception:
		pass
	# Simple float() cast (works for Python numeric, numpy scalars, jpype numerics)
	try:
		return float(pred)  # type: ignore[arg-type]
	except Exception:
		pass
	# Try common numeric accessors from Java/JPype-wrapped objects
	for accessor in ("doubleValue", "floatValue", "get", "value"):
		try:
			method = getattr(pred, accessor, None)
			if callable(method):
				val = method()
				return float(val)
		except Exception:
			continue
	# Last resort: string cast then float
	try:
		return float(str(pred))
	except Exception:
		return None


def _is_finite_number(val: Any) -> bool:
	"""Check that val is a finite number (int/float, not NaN/inf)."""
	try:
		import math as _math
		return isinstance(val, (int, float)) and _math.isfinite(float(val))
	except Exception:
		return False

def _create_capymoa_schema(first_x: dict[str, Any], is_classification: bool) -> Any:
	"""Create a CapyMOA schema from the first data sample.
	
	Creates a temporary CSV file and uses CSVStream to generate the schema,
	which is the proper way to create a schema in CapyMOA.
	"""
	try:
		import capymoa.stream as cm_stream
		import tempfile
		import os
		import csv
		
		# Get feature names and types
		feature_names = list(first_x.keys())
		if not feature_names:
			return None
		
		# Create a temporary CSV file with header and one row
		with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
			temp_path = f.name
			writer = csv.writer(f)
			
			# Write header
			header = list(feature_names) + ['target']
			writer.writerow(header)
			
			# Write one data row (using first_x values and a dummy target)
			row = [first_x[name] for name in feature_names] + [0 if is_classification else 0.0]
			writer.writerow(row)
		
		try:
			# Create CSVStream and get its schema
			stream = cm_stream.CSVStream(temp_path, class_index=-1)
			schema = stream.get_schema()
			return schema
		finally:
			# Clean up temporary file
			try:
				os.unlink(temp_path)
			except Exception:
				pass
	except Exception as e:
		logger.warning(f"Failed to create CapyMOA schema: {e}")
		import traceback
		logger.debug(traceback.format_exc())
		return None

def run_dataset(model_str: str, no_dataset: int, no_track: int) -> list[dict[str, Any]]:
	"""Run a model on a dataset using CapyMOA streams directly."""
	track = TRACKS[no_track]
	dataset = track["datasets"][no_dataset]
	track_name = track["name"]
	model_factory = MODELS[track_name][model_str]
	
	# Get CSV path from dataset
	from capy_moa_config import CSVStreamDataset
	if isinstance(dataset, CSVStreamDataset):
		csv_path = dataset.csv_path
	else:
		raise ValueError(f"Dataset {dataset} is not a CSVStreamDataset")
	
	# Adapter to unify CapyMOA and baseline models
	from capy_moa_config import CapyMoaAdapter
	
	dataset_name = getattr(dataset, "name", dataset.__class__.__name__)
	print(f"Processing {model_str} on {dataset_name}")

	# Use CapyMOA CSVStream directly
	import capymoa.stream as cm_stream
	import csv
	import tempfile
	import os
	
	# For multiclass with string labels, we need to convert them to numeric indices
	# because CSVStream has issues with string labels
	label_to_index = {}  # Map original labels to numeric indices
	index_to_label = {}  # Map numeric indices back to original labels
	use_temp_csv = False
	temp_csv_path = None
	
	is_classification = "classification" in track_name.lower()
	
	if is_classification:
		# Check if labels are strings (multiclass) or numeric (binary)
		try:
			with open(csv_path, 'r') as f:
				reader = csv.reader(f)
				header = next(reader)  # Skip header
				# Read first few rows to check label type
				sample_labels = []
				for i, row in enumerate(reader):
					if i >= 10:  # Check first 10 rows
						break
					if len(row) > 0:
						label = row[-1].strip()  # Last column is target
						sample_labels.append(label)
						# Try to convert to float/int
						try:
							float(label)
							# Numeric label - no conversion needed
						except ValueError:
							# String label - need to convert to indices
							use_temp_csv = True
							break
			
			# If we need conversion, read all labels and create mapping
			if use_temp_csv:
				all_labels = set()
				with open(csv_path, 'r') as f:
					reader = csv.reader(f)
					next(reader)  # Skip header
					for row in reader:
						if len(row) > 0:
							all_labels.add(row[-1].strip())
				
				# Create label to index mapping
				sorted_labels = sorted(list(all_labels))
				label_to_index = {label: idx for idx, label in enumerate(sorted_labels)}
				index_to_label = {idx: label for label, idx in label_to_index.items()}
				
				# Create temporary CSV with numeric labels
				temp_csv_path = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='')
				with open(csv_path, 'r') as f_in, open(temp_csv_path.name, 'w', newline='') as f_out:
					reader = csv.reader(f_in)
					writer = csv.writer(f_out)
					# Write header
					writer.writerow(next(reader))
					# Convert labels to indices
					for row in reader:
						if len(row) > 0:
							label = row[-1].strip()
							row[-1] = str(label_to_index[label])
							writer.writerow(row)
				
				csv_path_to_use = temp_csv_path.name
			else:
				csv_path_to_use = str(csv_path)
		except Exception as e:
			# If detection/conversion fails, use original CSV
			logger.warning(f"Failed to detect/convert labels: {e}, using original CSV")
			csv_path_to_use = str(csv_path)
			use_temp_csv = False
	else:
		csv_path_to_use = str(csv_path)
	
	# Create stream from CSV
	stream = cm_stream.CSVStream(csv_path_to_use, class_index=-1)
	schema = stream.get_schema()
	
	# Create model with schema
	sig = inspect.signature(model_factory)
	if len(sig.parameters) > 0:
		# Model needs schema
		try:
			raw_model = model_factory(schema)
			if raw_model is None:
				raise ValueError(f"Model factory returned None for {model_str}")
			# Wrap with adapter so we can use predict_one/learn_one uniformly
			model = CapyMoaAdapter(raw_model, schema=schema)
			if model is None:
				raise ValueError(f"Model factory returned None for {model_str}")
		except Exception as e:
			error_msg = str(e)
			if "SIGBUS" in error_msg or "fatal error" in error_msg.lower():
				raise RuntimeError(f"Java crash detected for {model_str}. Skipping this model.")
			raise ValueError(f"Failed to create model {model_str} with schema: {e}")
	else:
		# Model doesn't need schema
		try:
			raw_model = model_factory()
			if raw_model is None:
				raise ValueError(f"Model factory returned None for {model_str}")
			# Wrap with adapter even for baseline models
			model = CapyMoaAdapter(raw_model, schema=schema)
		except Exception as e:
			error_msg = str(e)
			if "SIGBUS" in error_msg or "fatal error" in error_msg.lower():
				raise RuntimeError(f"Java crash detected for {model_str}. Skipping this model.")
			raise ValueError(f"Failed to create model {model_str}: {e}")

	results: list[dict[str, Any]] = []
	# Choose metrics - create new instances instead of copying
	metric_objs: list[metrics.base.Metric] = [_create_metric_instance(m) for m in track["metrics"]]
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

	# Prepare feature names from schema for dict conversion
	feature_names = []
	try:
		num_attrs = schema.get_num_attributes()
		for i in range(num_attrs):
			attr = schema.get_attribute(i)
			feature_names.append(attr.name())
	except Exception:
		feature_names = []
	
	# Numpy for array handling
	import numpy as np
	
	# Iterate directly over the CSV rows to avoid MOA instance shape quirks
	import csv as _csv
	is_regression_track = "regression" in track_name.lower()
	with open(csv_path_to_use, 'r') as _f:
		_reader = _csv.reader(_f)
		try:
			_header = next(_reader)
		except StopIteration:
			_header = []
		# Assume last column is target
		valid_updates_since_ckpt = 0
		for row_vals in tqdm(_reader, desc=f"{model_str} on {dataset_name}", total=total if total else None):
			if not row_vals:
				continue
			# y extraction
			try:
				if is_regression_track:
					y = float(row_vals[-1])
				else:
					# Labels may already be numeric in csv_path_to_use (after conversion)
					label_str = row_vals[-1]
					try:
						y = int(label_str)
					except ValueError:
						low = label_str.strip().lower()
						if low in ('true', '1', 'yes'):
							y = True
						elif low in ('false', '0', 'no'):
							y = False
						else:
							y = label_str
			except Exception:
				y = None
			
			# x dict aligned to schema feature names
			try:
				values = row_vals[:-1]
				num_schema_feats = len(feature_names)
				# Adjust length to schema
				if len(values) >= num_schema_feats:
					values = values[:num_schema_feats]
				else:
					values = values + ["0"] * (num_schema_feats - len(values))
				def _to_float(v):
					try:
						return float(v)
					except Exception:
						return 0.0
				x_dict = {feature_names[i]: _to_float(values[i]) for i in range(num_schema_feats)}
			except Exception:
				x_dict = None
			
			# Predict using the unified adapter interface
			y_pred = None
			if x_dict is not None:
				try:
					y_pred = model.predict_one(x_dict)
					if is_regression_track:
						# Normalize regression outputs to a scalar float
						if isinstance(y_pred, (list, tuple)):
							y_pred = float(y_pred[0]) if y_pred else None
						else:
							y_pred = _coerce_regression_pred_to_float(y_pred)
				except Exception as e:
					if step <= 3:
						logger.warning(f"Prediction error on step {step}: {e}")
					y_pred = None
			
			# Update metrics
			if y is not None and y_pred is not None:
				# For regression, require finite numeric values
				if is_regression_track:
					if not _is_finite_number(y) or not _is_finite_number(y_pred):
						pass
					else:
						for m in metric_objs:
							try:
								m.update(y, y_pred)
							except Exception as e:
								if step == 1:
									logger.debug(f"Metric update error: {e}")
						valid_updates_since_ckpt += 1
				else:
					# Classification: update directly
					for m in metric_objs:
						try:
							m.update(y, y_pred)
						except Exception as e:
							if step == 1:
								logger.debug(f"Metric update error: {e}")
					valid_updates_since_ckpt += 1
			
			# Learn
			if x_dict is not None and y is not None:
				try:
					model.learn_one(x_dict, y)
				except Exception as e:
					if step == 1:
						logger.debug(f"Training error on step {step}: {e}")
			
			step += 1
			if step - last_checkpoint_step >= interval:
				elapsed = time.perf_counter() - start_time
				mem_mb = _get_memory_mb()
				row = {
					"step": step,
					"track": track_name,
					"model": model_str,
					"dataset": dataset_name,
					"Memory in Mb": mem_mb,
					"Time in s": elapsed,
					"Valid updates": valid_updates_since_ckpt,
				}
				for name, m in zip(metric_names, metric_objs):
					try:
						row[name] = m.get()
					except Exception:
						pass
				results.append(row)
				last_checkpoint_step = step
				valid_updates_since_ckpt = 0

	# Ensure we emit a final checkpoint if none were produced
	if not results:
		elapsed = time.perf_counter() - start_time
		mem_mb = _get_memory_mb()
		row = {
			"step": step,
			"track": track_name,
			"model": model_str,
			"dataset": dataset_name,
			"Memory in Mb": mem_mb,
			"Time in s": elapsed,
		}
		for name, m in zip(metric_names, metric_objs):
			try:
				row[name] = m.get()
			except Exception:
				pass
		results.append(row)
	
	# Clean up temporary CSV if created (after all iterations are done)
	if use_temp_csv and temp_csv_path and os.path.exists(temp_csv_path.name):
		try:
			os.unlink(temp_csv_path.name)
		except Exception:
			pass

	return results


def _init_worker():
	"""Initialize worker process - each process needs its own JVM."""
	# Set JAVA_OPTS if not already set
	import os
	if 'JAVA_OPTS' not in os.environ:
		os.environ['JAVA_OPTS'] = '-Xmx2g -Xss1m'
	
	# Import CapyMOA to initialize JVM in this process
	try:
		import capymoa as cm
		# Force JVM initialization by accessing a simple attribute
		_ = cm.__version__ if hasattr(cm, '__version__') else None
	except Exception:
		pass  # JVM will be initialized when needed


def _run_dataset_wrapper(args):
	"""Wrapper for run_dataset to handle errors in multiprocessing.
	
	Ensures all necessary imports are available in worker processes.
	"""
	try:
		# Re-import to ensure globals are available in worker process
		from capy_moa_config import MODELS, TRACKS
		# Call the actual function
		return run_dataset(*args)
	except Exception as e:
		import logging
		logger = logging.getLogger(__name__)
		logger.error(f"Error in run_dataset {args}: {e}")
		return []  # Return empty list on error


def run_track(models: list[str], no_track: int, n_workers: int = 1) -> None:
	"""Run track with models. 
	
	Args:
		models: List of model names to test
		no_track: Track index
		n_workers: Number of parallel workers (default: 1)
			- n_workers=1: Sequential execution (safest)
			- n_workers>1: Parallel execution (each process has its own JVM)
			- Recommended: n_workers = min(4, number of CPU cores) to avoid memory issues
	
	Results are saved by model class (classification, regression, anomaly).
	"""
	track = TRACKS[no_track]
	runs = list(itertools.product(models, range(len(track["datasets"])), [no_track]))
	results: list[dict[str, Any]] = []

	if n_workers == 1:
		# Sequential execution (safest for Java/CapyMOA)
		for run_args in tqdm(runs, desc=f"Running {track['name']} track"):
			try:
				val = run_dataset(*run_args)
				results.extend(val)
			except Exception as e:
				logger.error(f"Error in run_dataset {run_args}: {e}")
				continue
	else:
		# Parallel execution - each process gets its own JVM
		print(f"  Using {n_workers} parallel workers (each with its own JVM)")
		with multiprocessing.Pool(processes=n_workers, initializer=_init_worker) as pool:
			# Use imap for better progress tracking
			pool_results = list(tqdm(
				pool.imap(_run_dataset_wrapper, runs),
				total=len(runs),
				desc=f"Running {track['name']} track (parallel)"
			))
			# Flatten results
			for val in pool_results:
				if val:
					results.extend(val)
	
	# Save results by model class
	name_lower = track["name"].lower()
	if "binary classification" in name_lower:
		csv_name = "capymoa_binary_classification.csv"
	elif "regression" in name_lower:
		csv_name = "capymoa_regression.csv"
	elif "anomaly" in name_lower:
		csv_name = "capymoa_anomaly.csv"
	elif "multiclass classification" in name_lower:
		csv_name = "capymoa_multiclass_classification.csv"
	else:
		csv_name = track["name"].replace(" ", "_").lower() + ".csv"
	
	# Append to existing file if it exists, otherwise create new
	existing_df = None
	try:
		existing_df = pd.read_csv(csv_name)
	except FileNotFoundError:
		pass
	
	if existing_df is not None and len(existing_df) > 0:
		# Append to existing
		new_df = pd.DataFrame(results)
		combined_df = pd.concat([existing_df, new_df], ignore_index=True)
		combined_df.to_csv(csv_name, index=False)
	else:
		# Create new file
		pd.DataFrame(results).to_csv(csv_name, index=False)


if __name__ == "__main__":
	import argparse
	
	# Parse command line arguments
	parser = argparse.ArgumentParser(description="Benchmark CapyMOA models on datasets")
	parser.add_argument(
		"--workers", "-w",
		type=int,
		default=4,
		help="Number of parallel workers (default: 1, sequential). Each worker has its own JVM. Recommended: 2-4."
	)
	args = parser.parse_args()
	
	# Validate and limit number of workers
	import os
	max_recommended = min(4, os.cpu_count() or 4)
	if args.workers > max_recommended:
		print(f"⚠️  Warning: {args.workers} workers requested, but {max_recommended} is recommended")
		print(f"   Each worker uses ~2GB RAM (JVM). Limiting to {max_recommended} workers.")
		args.workers = max_recommended
	
	if args.workers < 1:
		print("⚠️  Number of workers must be >= 1. Using 1 worker.")
		args.workers = 1
	
	from capy_moa_config import DISABLE_CAPYMOA
	capymoa_available = False

	if not DISABLE_CAPYMOA:
			try:
				import capymoa as cm
				print(f"✓ CapyMOA est installé (version: {getattr(cm, '__version__', 'unknown')})")
				capymoa_available = True
			except ImportError:
				print("❌ CapyMOA n'est pas installé!")
				print("   Installez-le avec: pip install capymoa")
			except Exception as e:
				print(f"⚠️  Erreur lors de l'import de CapyMOA: {e}")
				print("   Le benchmark continuera avec les modèles baseline uniquement.")
	else:
		print("ℹ️  CapyMOA est désactivé (DISABLE_CAPYMOA=True)")
	
	# Check for failed models (collected during module import)
	failed = get_failed_models()
	
	# Display model counts
	total_models = sum(len(m) for m in MODELS.values())
	print(f"\nModèles chargés: {total_models}")
	print(f"  - Binary classification: {len(MODELS.get('Binary classification', {}))}")
	print(f"  - Multiclass classification: {len(MODELS.get('Multiclass classification', {}))}")
	print(f"  - Regression: {len(MODELS.get('Regression', {}))}")
	print(f"  - Anomaly detection: {len(MODELS.get('Anomaly detection', {}))}")
	
	if not capymoa_available and total_models == 0:
		print("\n⚠️  Aucun modèle disponible! Le benchmark ne peut pas s'exécuter.")
		exit(1)
	
	if failed:
		print("\n" + "="*60)
		print("MODÈLES QUI ONT ÉCHOUÉ À L'INSTANCIATION:")
		print("="*60)
		for model_name, error in failed:
			print(f"  ❌ {model_name}")
			# Truncate long error messages
			error_display = error[:150] + "..." if len(error) > 150 else error
			print(f"     Erreur: {error_display}")
		print(f"\nTotal: {len(failed)} modèle(s) échoué(s)")
		print("="*60 + "\n")
	else:
		if total_models == 0:
			print("\n⚠️  Aucun modèle CapyMOA n'a pu être chargé!")
			print("   Vérifiez que CapyMOA est installé et que les modèles peuvent être instanciés.")
		else:
			print("\n✓ Tous les modèles ont été instanciés avec succès!\n")
	
	# Merge binary and multiclass models for convenience, mirroring the River script behavior
	combined_classification = {**MODELS.get("Binary classification", {}), **MODELS.get("Multiclass classification", {})}

	details: dict[str, dict[str, dict[str, str]]] = {}
	
	# Run all tracks: Binary classification, Multiclass classification, and Regression
	print(f"\n{'='*60}")
	print(f"EXÉCUTION DU BENCHMARK POUR {len(TRACKS)} TRACK(S)")
	if args.workers > 1:
		print(f"Mode parallèle: {args.workers} workers")
		print(f"  ⚠️  Chaque worker a sa propre JVM (consommation mémoire élevée)")
		print(f"  💡 Recommandé: 2-4 workers pour éviter les problèmes de mémoire")
	else:
		print(f"Mode séquentiel (1 worker)")
	print(f"{'='*60}\n")
	
	for i, track in enumerate(TRACKS):
		track_name = track["name"]

		if track_name == "Regression":
			continue


		print(f"\n{'='*60}")
		print(f"Track {i+1}/{len(TRACKS)}: {track_name}")
		print(f"{'='*60}")
		
		if "classification" in track_name.lower():
			model_bank = combined_classification
		elif "regression" in track_name.lower():
			model_bank = MODELS.get("Regression", {})
		elif "anomaly" in track_name.lower():
			model_bank = MODELS.get("Anomaly detection", {})
		else:
			model_bank = {}
		
		print(f"Nombre de modèles à tester: {len(model_bank)}")
		print(f"Nombre de datasets: {len(track['datasets'])}")
		
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

		# Use specified number of workers (default: 1 for safety)
		# Each worker process will have its own JVM
		run_track(models=list(model_bank.keys()), no_track=i, n_workers=args.workers)
	
	print(f"\n{'='*60}")
	print("BENCHMARK TERMINÉ!")
	print(f"{'='*60}\n")
