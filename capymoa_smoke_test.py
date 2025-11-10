from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple, Iterable

import pandas as pd


def iter_csv(csv_path: Path, target_col: str = "target") -> Iterable[Tuple[Dict[str, Any], Any]]:
	df = pd.read_csv(csv_path)
	if target_col not in df.columns:
		raise ValueError(f"{csv_path}: colonne cible '{target_col}' introuvable")
	for _, row in df.iterrows():
		y = row[target_col]
		x = row.drop(labels=[target_col]).to_dict()
		yield x, y


def build_schema_from_first_row(cm, first_x: Dict[str, Any], is_classification: bool = True):
	import capymoa.stream as cm_stream
	attributes = []
	for name in first_x.keys():
		attributes.append(cm_stream.NumericAttribute(name=name))
	target = cm_stream.ClassLabel() if is_classification else cm_stream.NumericTarget()
	return cm_stream.Schema(attributes=attributes, target=target)


def main():
	# Dataset: prenez un dataset simple de classification binaire
	base_dir = Path(__file__).resolve().parent
	csv_path = base_dir / "datasets" / "binary_classification" / "bananas.csv"
	if not csv_path.exists():
		print(f"❌ Dataset introuvable: {csv_path}")
		sys.exit(1)

	# Affichez quelques infos JVM pour debug rapide
	java_home = os.environ.get("JAVA_HOME", "<non défini>")
	jvm_env = os.environ.get("JPYPE_JVM", "<non défini>")
	print(f"JAVA_HOME={java_home}")
	print(f"JPYPE_JVM={jvm_env}")

	# 1) Import CapyMOA (la JVM est gérée par le package si nécessaire)
	try:
		import capymoa as cm  # type: ignore
		print(f"✓ CapyMOA importé (version: {getattr(cm, '__version__', 'inconnue')})")
	except Exception as e:
		print(f"❌ Impossible d'importer CapyMOA: {e}")
		sys.exit(1)

	# 2) Préparez le flux et le schéma
	stream = iter_csv(csv_path)
	try:
		first_x, first_y = next(stream)
	except StopIteration:
		print("❌ Dataset vide.")
		sys.exit(1)
	# Remettez la première instance au début
	def again():
		yield first_x, first_y
		for x, y in iter_csv(csv_path):
			yield x, y
	stream = again()

	schema = build_schema_from_first_row(cm, first_x, is_classification=True)

	# 3) Instanciez un modèle simple
	try:
		model = cm.classifier.HoeffdingTree(schema=schema)
		print("✓ Modèle: HoeffdingTree")
	except Exception as e:
		print(f"❌ Échec d'instanciation du modèle: {e}")
		sys.exit(1)

	# 4) Entraînement rapide sur 1000 instances max, calcul d'une accuracy simple
	n = 0
	n_correct = 0
	max_steps = 1000
	for x, y in stream:
		try:
			y_pred = model.predict_one(x)
		except Exception:
			y_pred = None
		if y_pred is not None and y_pred == y:
			n_correct += 1
		try:
			model.learn_one(x, y)
		except Exception:
			pass
		n += 1
		if n >= max_steps:
			break

	accuracy = (n_correct / n) if n else 0.0
	print(f"✓ Fini: {n} instances, Accuracy approx: {accuracy:.4f}")


if __name__ == "__main__":
	main()


