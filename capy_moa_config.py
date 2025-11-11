from __future__ import annotations

import inspect
from typing import Any, Callable, Dict

import os
from pathlib import Path
from typing import Iterable

import pandas as pd

# Option to disable CapyMOA models (useful if Java crashes)
# Set to True to only use baseline models
# If you get SIGBUS errors, try:
#   1. Set DISABLE_CAPYMOA = True (use only baselines)
#   2. Or set JAVA_OPTS before running: export JAVA_OPTS="-Xmx2g -Xss1m"
DISABLE_CAPYMOA = False  # Changed to True to avoid Java crashes

# Block JPype/CapyMOA imports if disabled to prevent JVM initialization
if DISABLE_CAPYMOA:
	import sys
	# Create fake modules to prevent any import of capymoa/jpype
	class FakeModule:
		def __getattr__(self, name):
			raise ImportError(f"CapyMOA is disabled (DISABLE_CAPYMOA=True)")
	
	sys.modules['capymoa'] = FakeModule()
	sys.modules['jpype'] = FakeModule()
	cm = None
	print("⚠️  CapyMOA models are disabled (DISABLE_CAPYMOA=True)")
	print("   Only baseline models will be used.")
else:
	try:
		# Try to configure Java memory if not already set
		import os
		if 'JAVA_OPTS' not in os.environ:
			# Set conservative memory settings to avoid SIGBUS on macOS ARM64
			os.environ['JAVA_OPTS'] = '-Xmx2g -Xss1m'
			print("ℹ️  Setting JAVA_OPTS=-Xmx2g -Xss1m (reduce if you still get crashes)")
		
		# Let CapyMOA handle JVM startup completely - it needs to set up the classpath
		# for MOA libraries. Don't start JVM manually here.
		import capymoa as cm  # type: ignore
	except Exception as e:  # pragma: no cover
		cm = None
		error_msg = str(e)
		if "SIGBUS" in error_msg or "fatal error" in error_msg.lower():
			print("\n" + "="*60)
			print("❌ Java crash detected during CapyMOA import!")
			print("="*60)
			print("Solutions:")
			print("  1. Set DISABLE_CAPYMOA = True in capy_moa_config.py")
			print("  2. Or set JAVA_OPTS before running:")
			print("     export JAVA_OPTS='-Xmx1g -Xss512k'")
			print("     python benchmark_capy_moa.py")
			print("  3. Or try a different Java version (Java 17+ recommended)")
			print("="*60 + "\n")
		else:
			print(f"⚠️  Failed to import CapyMOA: {e}")
			print("   The benchmark will continue with baseline models only.")

# Reuse River tracks' datasets to keep parity with your River benchmark
try:
	from river_config import TRACKS as RIVER_TRACKS  # type: ignore
except Exception:
	RIVER_TRACKS = []

from river import metrics

N_CHECKPOINTS = 50


class CSVStreamDataset:
	"""Simple streaming dataset backed by a CSV saved by save_datasets.py.

	- Expects a 'target' column for labels.
	- Iterates as (x_dict, y) pairs like River datasets.
	"""

	def __init__(self, csv_path: Path, name: str | None = None, target_col: str = "target") -> None:
		self.csv_path = Path(csv_path)
		self.target_col = target_col
		self.name = name or self.csv_path.stem
		# Load eagerly to expose n_samples; acceptable for moderate sizes
		self._df = pd.read_csv(self.csv_path)
		if self.target_col not in self._df.columns:
			raise ValueError(f"{self.csv_path}: missing target column '{self.target_col}'")
		self.n_samples = len(self._df)

	def __iter__(self) -> Iterable[tuple[dict[str, Any], Any]]:
		for _, row in self._df.iterrows():
			y = row[self.target_col]
			x = row.drop(labels=[self.target_col]).to_dict()
			yield x, y

	def __repr__(self) -> str:
		return f"CSVStreamDataset(name={self.name!r}, path={str(self.csv_path)!r}, n_samples={self.n_samples})"


def _load_local_tracks_if_available(base_dir: Path) -> list[dict[str, Any]] | None:
	"""Build TRACKS from ./datasets directory produced by save_datasets.py."""
	if not base_dir.exists():
		return None

	subdirs = {
		"binary_classification": ("Binary classification", lambda: [metrics.Accuracy()]),
		"multiclass_classification": ("Multiclass classification", lambda: [metrics.Accuracy()]),
		"regression": ("Regression", lambda: [metrics.MAE(), metrics.RMSE()]),
		"anomaly_detection": ("Anomaly detection", lambda: [metrics.Accuracy()]),  # Use accuracy for anomaly (anomaly vs normal)
	}

	tracks: list[dict[str, Any]] = []
	for folder_name, (track_name, metrics_factory) in subdirs.items():
		track_path = base_dir / folder_name
		if not track_path.exists():
			continue
		datasets: list[Any] = []
		for entry in sorted(track_path.glob("*.csv")):
			# Derive a readable dataset name from filename
			nice_name = entry.stem
			datasets.append(CSVStreamDataset(entry, name=nice_name))
		if datasets:
			tracks.append({
				"name": track_name,
				"kind": "Regression" if "Regression" in track_name else "Classification",
				"datasets": datasets,
				"metrics": metrics_factory(),  # Create new instances for each track
			})

	return tracks if tracks else None


class CapyMoaAdapter:
	"""Unify CapyMOA (or similar) learners to a simple learn_one/predict_one interface.

	This wrapper tries, in order:
	- learn_one/predict_one (River-style)
	- train/predict (CapyMOA-style with MOA instances)
	- partial_fit/predict (sklearn-style)
	- fit_one/predict_one (alternative River-style)
	"""

	def __init__(self, model: Any, classes: list[Any] | None = None, schema: Any = None) -> None:
		self.model = model
		self._classes = classes
		self._schema = schema
		
		# Check for different API styles
		self._has_learn_one = hasattr(model, "learn_one") and callable(getattr(model, "learn_one"))
		self._has_predict_one = hasattr(model, "predict_one") and callable(getattr(model, "predict_one"))
		self._has_train = hasattr(model, "train") and callable(getattr(model, "train"))
		self._has_predict = hasattr(model, "predict") and callable(getattr(model, "predict"))
		self._has_partial_fit = hasattr(model, "partial_fit") and callable(getattr(model, "partial_fit"))
		self._has_fit_one = hasattr(model, "fit_one") and callable(getattr(model, "fit_one"))
		
		# For CapyMOA models, we need to cache feature order from schema
		self._feature_names = None
		if self._schema is not None and self._has_train:
			try:
				import capymoa.instance as cm_instance
				import numpy as np
				# Get feature names from schema
				num_attrs = self._schema.get_num_attributes()
				self._feature_names = []
				for i in range(num_attrs):
					attr = self._schema.get_attribute(i)
					self._feature_names.append(attr.name())
			except Exception:
				pass

	def _dict_to_moa_instance(self, x: dict[str, Any], y: Any = None) -> Any:
		"""Convert a dict to a CapyMOA MOA instance.
		
		If y is None, creates an unlabeled instance for prediction.
		If y is provided, creates a labeled instance for training.
		
		Tries NumpyStream first (stable 2D arrays), then falls back to CSVStream.
		"""
		if self._schema is None or self._feature_names is None:
			raise ValueError("Schema and feature names required for CapyMOA instance conversion")
		
		import capymoa.stream as cm_stream
		import numpy as np
		# Try NumpyStream first
		try:
			x_vec = [float(x.get(name, 0.0)) for name in self._feature_names]
			x_arr = np.array(x_vec, dtype=float).reshape(1, -1)  # 2D
			# Dummy target: numeric 0.0; CapyMOA ignores for prediction
			t_arr = np.array([[float(y) if y is not None else 0.0]], dtype=float)
			stream = cm_stream.NumpyStream(x_arr, t_arr)
			instance = stream.next_instance()
			instance._schema = self._schema
			return instance
		except Exception:
			# Fallback to temporary CSV approach
			import tempfile
			import csv
			import os
			with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
				temp_path = f.name
				writer = csv.writer(f)
				header = list(self._feature_names) + ['target']
				writer.writerow(header)
				row = [x.get(name, 0.0) for name in self._feature_names]
				row.append(y if y is not None else 0)
				writer.writerow(row)
				# Add second dummy row
				dummy_row = [0.0] * len(self._feature_names) + [0]
				writer.writerow(dummy_row)
			try:
				stream = cm_stream.CSVStream(temp_path, class_index=-1)
				instance = stream.next_instance()
				instance._schema = self._schema
				return instance
			finally:
				try:
					os.unlink(temp_path)
				except Exception:
					pass

	def predict_one(self, x: dict[str, Any]):
		if self._has_predict_one:
			return self.model.predict_one(x)
		if self._has_predict:
			# CapyMOA models need MOA instances
			if self._has_train and self._schema is not None:
				instance = self._dict_to_moa_instance(x, y=None)
				return self.model.predict(instance)
			# Try direct predict (sklearn-style)
			try:
				return self.model.predict(x)
			except TypeError:
				try:
					return self.model.predict([x])[0]
				except Exception:
					raise AttributeError("Model does not implement a prediction method")
		raise AttributeError("Model does not implement a prediction method")

	def learn_one(self, x: dict[str, Any], y: Any):
		if self._has_learn_one:
			self.model.learn_one(x, y)
			return self
		if self._has_train:
			# CapyMOA models use train() with MOA instances
			if self._schema is not None:
				instance = self._dict_to_moa_instance(x, y)
				self.model.train(instance)
				return self
		if self._has_fit_one:
			self.model.fit_one(x, y)
			return self
		if self._has_partial_fit:
			# sklearn-like API
			kwargs = {}
			if self._classes is not None and "classes" in inspect.signature(self.model.partial_fit).parameters:
				kwargs["classes"] = self._classes
			try:
				self.model.partial_fit(x, y, **kwargs)
			except TypeError:
				self.model.partial_fit([x], [y], **kwargs)
			return self
		raise AttributeError("Model does not implement a learning method")


# Track failed models for reporting
_failed_models: list[tuple[str, str]] = []  # (model_name, error_message)

def _maybe_with_schema(name: str, model_class: Any) -> tuple[str, Callable[[Any], Any]] | None:
	"""Create a factory for CapyMOA models that require a schema.
	
	Returns a factory that takes a schema and creates the model.
	"""
	if cm is None:
		return None
	try:
		# Check if the class exists
		if not hasattr(model_class, '__call__'):
			return None
		# Return a factory that takes a schema
		def factory(schema: Any):
			# Try with seed if available
			if 'seed' in inspect.signature(model_class).parameters:
				return model_class(schema=schema, seed=42)
			return model_class(schema=schema)
		return name, factory
	except Exception as e:
		# Track failed models for reporting
		error_msg = str(e)
		_failed_models.append((name, error_msg))
		return None

def _maybe(name: str, factory: Callable[[], Any]) -> tuple[str, Callable[[], Any]] | None:
	"""Guard model factories if capymoa is not available or class is missing.
	
	For models that don't require schema, try to instantiate directly.
	For models that require schema, they will be handled separately.
	"""
	if cm is None:
		return None
	try:
		# probe by instantiating then disposing; return a real factory to defer cost later
		instance = factory()
		del instance
		return name, factory
	except Exception as e:
		# Don't track as failed yet - might need schema
		# We'll try with schema in _maybe_with_schema
		return None

def get_failed_models() -> list[tuple[str, str]]:
	"""Return list of models that failed to instantiate."""
	return _failed_models.copy()

def clear_failed_models():
	"""Clear the list of failed models."""
	_failed_models.clear()


# Build datasets and per-track default metrics mirroring River tracks
_LOCAL_DATASETS_DIR = Path(__file__).resolve().parent / "datasets"
TRACKS: list[dict[str, Any]] = _load_local_tracks_if_available(_LOCAL_DATASETS_DIR) or []

# Fallback to River tracks if no local datasets are found
if not TRACKS:
	for t in RIVER_TRACKS:
		track_name = t.name
		if "Regression" in track_name:
			default_metrics = [metrics.MAE(), metrics.RMSE()]
			kind = "Regression"
		else:
			default_metrics = [metrics.Accuracy()]
			kind = "Classification"
		TRACKS.append({
			"name": track_name,
			"kind": kind,
			"datasets": list(t.datasets),
			"metrics": default_metrics,
		})


# Define CapyMOA models per task; guarded to avoid import crashes when not installed
# Note: Some CapyMOA models may require a schema parameter. Those that cannot be
# instantiated without arguments will be filtered out by _maybe().
MODELS: Dict[str, Dict[str, Callable[[], Any]]] = {
	"Binary classification": {},
	"Multiclass classification": {},
	"Regression": {},
	"Anomaly detection": {},
}

if cm is not None:
	# Import classifier and regressor modules separately
	import capymoa.classifier as cm_classifier
	import capymoa.regressor as cm_regressor
	
	# Classification models - all CapyMOA classifiers
	# Try models that might work without schema first, then try with schema
	classifier_classes = [
		("AdaptiveRandomForestClassifier", cm_classifier.AdaptiveRandomForestClassifier),
		("CSMOTE", cm_classifier.CSMOTE),
		("DynamicWeightedMajority", cm_classifier.DynamicWeightedMajority),
		("EFDT", cm_classifier.EFDT),
		("Finetune", cm_classifier.Finetune),
		("HoeffdingAdaptiveTree", cm_classifier.HoeffdingAdaptiveTree),
		("HoeffdingTree", cm_classifier.HoeffdingTree),
		("KNN", cm_classifier.KNN),
		("LeveragingBagging", cm_classifier.LeveragingBagging),
		("MajorityClass", cm_classifier.MajorityClass),
		("NaiveBayes", cm_classifier.NaiveBayes),
		("NoChange", cm_classifier.NoChange),
		("OnlineAdwinBagging", cm_classifier.OnlineAdwinBagging),
		("OnlineBagging", cm_classifier.OnlineBagging),
		("OnlineSmoothBoost", cm_classifier.OnlineSmoothBoost),
		("OzaBoost", cm_classifier.OzaBoost),
		("PassiveAggressiveClassifier", cm_classifier.PassiveAggressiveClassifier),
		("SAMkNN", cm_classifier.SAMkNN),
		("SGDClassifier", cm_classifier.SGDClassifier),
		("ShrubsClassifier", cm_classifier.ShrubsClassifier),
		("StreamingGradientBoostedTrees", cm_classifier.StreamingGradientBoostedTrees),
		("StreamingRandomPatches", cm_classifier.StreamingRandomPatches),
		("WeightedkNN", cm_classifier.WeightedkNN),
	]
	
	for name, model_class in classifier_classes:
		# Try without schema first (fix closure issue)
		def make_factory_no_schema(mc):
			return lambda: mc()
		result = _maybe(name, make_factory_no_schema(model_class))
		if result is None:
			# Try with schema factory
			result = _maybe_with_schema(name, model_class)
		if result is not None:
			mod_name, factory = result
			MODELS["Binary classification"][mod_name] = factory
			MODELS["Multiclass classification"][mod_name] = factory

	# Regression models - all CapyMOA regressors
	regressor_classes = [
		("SOKNLBT", cm_regressor.SOKNLBT),
		("SOKNL", cm_regressor.SOKNL),
		("ORTO", cm_regressor.ORTO),
		("KNNRegressor", cm_regressor.KNNRegressor),
		("FIMTDD", cm_regressor.FIMTDD),
		("ARFFIMTDD", cm_regressor.ARFFIMTDD),
		("AdaptiveRandomForestRegressor", cm_regressor.AdaptiveRandomForestRegressor),
		("PassiveAggressiveRegressor", cm_regressor.PassiveAggressiveRegressor),
		("SGDRegressor", cm_regressor.SGDRegressor),
		("ShrubsRegressor", cm_regressor.ShrubsRegressor),
		("StreamingGradientBoostedRegression", cm_regressor.StreamingGradientBoostedRegression),
		("NoChange", cm_regressor.NoChange),
		("TargetMean", cm_regressor.TargetMean),
		("FadingTargetMean", cm_regressor.FadingTargetMean),
	]
	
	for name, model_class in regressor_classes:
		# Try without schema first (fix closure issue)
		def make_factory_no_schema(mc):
			return lambda: mc()
		result = _maybe(name, make_factory_no_schema(model_class))
		if result is None:
			# Try with schema factory
			result = _maybe_with_schema(name, model_class)
		if result is not None:
			mod_name, factory = result
			MODELS["Regression"][mod_name] = factory

	# Anomaly detection models - all CapyMOA anomaly detectors
	try:
		import capymoa.anomaly as cm_anomaly
		anomaly_classes = [
			("HalfSpaceTrees", cm_anomaly.HalfSpaceTrees),
			("OnlineIsolationForest", cm_anomaly.OnlineIsolationForest),
			("Autoencoder", cm_anomaly.Autoencoder),
			("StreamRHF", cm_anomaly.StreamRHF),
		]
		
		for name, model_class in anomaly_classes:
			# Try without schema first
			def make_factory_no_schema(mc):
				return lambda: mc()
			result = _maybe(name, make_factory_no_schema(model_class))
			if result is None:
				# Try with schema factory
				result = _maybe_with_schema(name, model_class)
			if result is not None:
				mod_name, factory = result
				MODELS["Anomaly detection"][mod_name] = factory
	except ImportError:
		# Anomaly module might not be available
		pass

# Always include simple baselines so the runner can operate even if capymoa isn't available
class LastClassBaseline:
	def __init__(self) -> None:
		self.last = None
	def predict_one(self, x):
		return self.last
	def learn_one(self, x, y):
		self.last = y
		return self

class MeanRegressorBaseline:
	def __init__(self) -> None:
		self.n = 0
		self.sum = 0.0
	def predict_one(self, x):
		return (self.sum / self.n) if self.n else 0.0
	def learn_one(self, x, y):
		self.n += 1
		self.sum += float(y)
		return self

MODELS["Binary classification"].setdefault("[baseline] Last Class", lambda: LastClassBaseline())
MODELS["Multiclass classification"].setdefault("[baseline] Last Class", lambda: LastClassBaseline())
MODELS["Regression"].setdefault("[baseline] Mean predictor", lambda: MeanRegressorBaseline())
