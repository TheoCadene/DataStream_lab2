from __future__ import annotations

import inspect
from typing import Any, Callable, Dict

try:
	import capymoa as cm  # type: ignore
except Exception:  # pragma: no cover
	cm = None  # lazy handling if capymoa is unavailable

# Reuse River tracks' datasets to keep parity with your River benchmark
try:
	from river_config import TRACKS as RIVER_TRACKS  # type: ignore
except Exception:
	RIVER_TRACKS = []

from river import metrics

N_CHECKPOINTS = 50


class CapyMoaAdapter:
	"""Unify CapyMOA (or similar) learners to a simple learn_one/predict_one interface.

	This wrapper tries, in order:
	- learn_one/predict_one
	- partial_fit/predict
	- fit_one/predict_one
	"""

	def __init__(self, model: Any, classes: list[Any] | None = None) -> None:
		self.model = model
		self._classes = classes
		self._has_learn_one = hasattr(model, "learn_one") and callable(getattr(model, "learn_one"))
		self._has_predict_one = hasattr(model, "predict_one") and callable(getattr(model, "predict_one"))
		self._has_partial_fit = hasattr(model, "partial_fit") and callable(getattr(model, "partial_fit"))
		self._has_predict = hasattr(model, "predict") and callable(getattr(model, "predict"))
		self._has_fit_one = hasattr(model, "fit_one") and callable(getattr(model, "fit_one"))

	def predict_one(self, x: dict[str, Any]):
		if self._has_predict_one:
			return self.model.predict_one(x)
		if self._has_predict:
			# Some models may accept a single-sample dict; others expect a list/array
			try:
				return self.model.predict(x)
			except TypeError:
				return self.model.predict([x])[0]
		raise AttributeError("Model does not implement a prediction method")

	def learn_one(self, x: dict[str, Any], y: Any):
		if self._has_learn_one:
			self.model.learn_one(x, y)
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


def _maybe(name: str, factory: Callable[[], Any]) -> tuple[str, Callable[[], Any]] | None:
	"""Guard model factories if capymoa is not available or class is missing."""
	if cm is None:
		return None
	try:
		# probe by instantiating then disposing; return a real factory to defer cost later
		instance = factory()
		del instance
		return name, factory
	except Exception:
		return None


# Build datasets and per-track default metrics mirroring River tracks
TRACKS: list[dict[str, Any]] = []
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
MODELS: Dict[str, Dict[str, Callable[[], Any]]] = {
	"Binary classification": {},
	"Multiclass classification": {},
	"Regression": {},
}

if cm is not None:
	# Classification models
	candidates: list[tuple[str, Callable[[], Any]] | None] = [
		_maybe("MOA Hoeffding Tree", lambda: cm.classifier.HoeffdingTreeClassifier()),
		_maybe("MOA Hoeffding Adaptive Tree", lambda: cm.classifier.HoeffdingAdaptiveTreeClassifier()),
		_maybe("MOA Naive Bayes", lambda: cm.classifier.NaiveBayes()),
		_maybe("MOA Adaptive Random Forest", lambda: cm.ensemble.ARFClassifier(seed=42)),
	]
	for item in filter(None, candidates):
		name, factory = item  # type: ignore[misc]
		MODELS["Binary classification"][name] = factory
		MODELS["Multiclass classification"][name] = factory

	# Regression models
	reg_candidates: list[tuple[str, Callable[[], Any]] | None] = [
		_maybe("MOA Hoeffding Tree Regressor", lambda: cm.regressor.HoeffdingTreeRegressor()),
		_maybe("MOA Hoeffding Adaptive Tree Regressor", lambda: cm.regressor.HoeffdingAdaptiveTreeRegressor(seed=42)),
		_maybe("MOA Adaptive Random Forest Regressor", lambda: cm.ensemble.ARFRegressor(seed=42)),
	]
	for item in filter(None, reg_candidates):
		name, factory = item  # type: ignore[misc]
		MODELS["Regression"][name] = factory

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
