#!/usr/bin/env python3
"""
Script minimal pour tester LeveragingBagging sur le dataset music (multiclass).
Affiche y_true et y_pred à chaque étape pour voir ce qui se passe.
"""

import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent))

import capymoa.stream as cm_stream
from capy_moa_config import CapyMoaAdapter, MODELS
from river import metrics

# Configuration
MODEL_NAME = "LeveragingBagging"
DATASET_NAME = "music"
DATASET_PATH = Path(__file__).parent / "datasets" / "multiclass_classification" / f"{DATASET_NAME}.csv"

print(f"🔧 Configuration:")
print(f"   Modèle: {MODEL_NAME}")
print(f"   Dataset: {DATASET_NAME}")
print(f"   Chemin: {DATASET_PATH}")
print()

# Vérifier que le fichier existe
if not DATASET_PATH.exists():
    print(f"❌ Erreur: Le fichier {DATASET_PATH} n'existe pas!")
    print(f"   Vérifiez que le dataset a été téléchargé.")
    sys.exit(1)

# Charger le stream (gérer les labels string pour multiclass)
print("📂 Chargement du stream...")
import csv
import tempfile
import os

csv_path = str(DATASET_PATH)
label_to_index = {}
index_to_label = {}
use_temp_csv = False
temp_csv_path = None

# Vérifier le format des labels (peut être multi-label avec dict)
print("   ℹ️  Analyse du format des labels...")
import ast

try:
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        # Lire quelques lignes pour vérifier le type de label
        first_label = None
        for i, row in enumerate(reader):
            if i >= 5:
                break
            if len(row) > 0:
                label_str = row[-1].strip()
                first_label = label_str
                break
        
        # Vérifier si c'est un dictionnaire (format multi-label)
        is_dict_format = False
        if first_label and first_label.startswith('{'):
            try:
                # Essayer de parser comme dict
                label_dict = ast.literal_eval(first_label)
                if isinstance(label_dict, dict):
                    is_dict_format = True
                    print("   ℹ️  Format multi-label détecté (dict), conversion en classes uniques...")
            except:
                pass
        
        # Si format dict, convertir en classes uniques
        if is_dict_format:
            use_temp_csv = True
            # Récupérer toutes les classes possibles
            all_class_names = set()
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if len(row) > 0:
                        try:
                            label_dict = ast.literal_eval(row[-1].strip())
                            if isinstance(label_dict, dict):
                                all_class_names.update(label_dict.keys())
                        except:
                            pass
            
            # Créer mapping: classe -> index
            sorted_classes = sorted(list(all_class_names))
            label_to_index = {cls: idx for idx, cls in enumerate(sorted_classes)}
            index_to_label = {idx: cls for cls, idx in label_to_index.items()}
            
            print(f"   ℹ️  {len(sorted_classes)} classes trouvées: {sorted_classes}")
            print("   ℹ️  Conversion: prendre la première classe True comme label")
            
            # Créer CSV temporaire
            temp_csv_path = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='')
            with open(csv_path, 'r') as f_in, open(temp_csv_path.name, 'w', newline='') as f_out:
                reader = csv.reader(f_in)
                writer = csv.writer(f_out)
                writer.writerow(next(reader))  # Header
                for row in reader:
                    if len(row) > 0:
                        try:
                            label_dict = ast.literal_eval(row[-1].strip())
                            if isinstance(label_dict, dict):
                                # Prendre la première classe qui est True
                                selected_class = None
                                for cls, is_true in label_dict.items():
                                    if is_true:
                                        selected_class = cls
                                        break
                                # Si aucune classe True, prendre la première classe
                                if selected_class is None:
                                    selected_class = sorted_classes[0]
                                row[-1] = str(label_to_index[selected_class])
                            else:
                                row[-1] = str(0)  # Fallback
                        except:
                            row[-1] = str(0)  # Fallback en cas d'erreur
                        writer.writerow(row)
            
            csv_path_to_use = temp_csv_path.name
        else:
            # Format normal (string ou numérique)
            try:
                float(first_label)
                # Numérique - pas de conversion
                csv_path_to_use = csv_path
                use_temp_csv = False
            except ValueError:
                # String - conversion nécessaire
                use_temp_csv = True
                print("   ℹ️  Labels string détectés, conversion en indices numériques...")
                all_labels = set()
                with open(csv_path, 'r') as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header
                    for row in reader:
                        if len(row) > 0:
                            all_labels.add(row[-1].strip())
                
                # Créer le mapping
                sorted_labels = sorted(list(all_labels))
                label_to_index = {label: idx for idx, label in enumerate(sorted_labels)}
                index_to_label = {idx: label for label, idx in label_to_index.items()}
                
                print(f"   ℹ️  {len(sorted_labels)} classes trouvées: {sorted_labels}")
                
                # Créer CSV temporaire
                temp_csv_path = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='')
                with open(csv_path, 'r') as f_in, open(temp_csv_path.name, 'w', newline='') as f_out:
                    reader = csv.reader(f_in)
                    writer = csv.writer(f_out)
                    writer.writerow(next(reader))  # Header
                    for row in reader:
                        if len(row) > 0:
                            label = row[-1].strip()
                            row[-1] = str(label_to_index[label])
                            writer.writerow(row)
                
                csv_path_to_use = temp_csv_path.name
except Exception as e:
    print(f"   ⚠️  Erreur lors de la détection des labels: {e}")
    import traceback
    traceback.print_exc()
    csv_path_to_use = csv_path
    use_temp_csv = False

# Créer le stream depuis le CSV
try:
    stream = cm_stream.CSVStream(csv_path_to_use, class_index=-1)
    schema = stream.get_schema()
    print(f"   ✓ Stream chargé")
    print(f"   ✓ Schema: {schema.get_num_attributes()} attributs")
    if use_temp_csv:
        print(f"   ✓ Labels convertis en indices numériques (0-{len(index_to_label)-1})")
except Exception as e:
    print(f"   ❌ Erreur lors du chargement du stream: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Créer le modèle
print(f"🤖 Création du modèle {MODEL_NAME}...")
try:
    model_factory = MODELS["Multiclass classification"][MODEL_NAME]
    
    # Créer le modèle avec le schema
    sig = __import__('inspect').signature(model_factory)
    if len(sig.parameters) > 0:
        raw_model = model_factory(schema)
    else:
        raw_model = model_factory()
    
    # Wrapper avec l'adapter
    model = CapyMoaAdapter(raw_model, schema=schema)
    print(f"   ✓ Modèle créé")
except Exception as e:
    print(f"   ❌ Erreur lors de la création du modèle: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Initialiser les métriques
accuracy = metrics.Accuracy()
microf1 = metrics.MicroF1()
macrof1 = metrics.MacroF1()

# Préparer les noms de features depuis le schema
feature_names = []
try:
    num_attrs = schema.get_num_attributes()
    for i in range(num_attrs):
        attr = schema.get_attribute(i)
        feature_names.append(attr.name())
except Exception:
    feature_names = []

print(f"📊 Démarrage de l'entraînement et des prédictions...")
print(f"   Features: {len(feature_names)}")
print()
print("=" * 80)
print(f"{'Step':<6} {'y_true':<15} {'y_pred':<15} {'Correct':<8} {'Accuracy':<10} {'MicroF1':<10} {'MacroF1':<10}")
print("=" * 80)

# Itérer sur le stream
step = 0
max_steps = 50  # Limiter à 50 étapes pour l'affichage

import numpy as np

# Lire le CSV directement pour avoir les labels originaux
with open(csv_path_to_use, 'r') as f:
    csv_reader = csv.reader(f)
    header = next(csv_reader)  # Skip header
    
    for row_vals in csv_reader:
        if not row_vals:
            continue
        
        step += 1
        
        # Extraire y (dernière colonne)
        try:
            y_str = row_vals[-1].strip()
            # Convertir en int (les labels sont déjà numériques dans csv_path_to_use)
            y_true = int(y_str)
            # Si on a fait une conversion, récupérer le label original
            if use_temp_csv and y_true in index_to_label:
                y_true_original = index_to_label[y_true]
            else:
                y_true_original = y_true
        except Exception:
            y_true = None
            y_true_original = None
        
        # Créer x_dict
        try:
            values = row_vals[:-1]
            num_schema_feats = len(feature_names)
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
        except Exception as e:
            print(f"   ⚠️  Erreur création x_dict à l'étape {step}: {e}")
            x_dict = None
        
        # Prédire
        y_pred = None
        if x_dict is not None:
            try:
                y_pred = model.predict_one(x_dict)
            except Exception as e:
                print(f"   ⚠️  Erreur prédiction à l'étape {step}: {e}")
                y_pred = None
        
        # Mettre à jour les métriques
        if y_true is not None and y_pred is not None:
            try:
                # Normaliser les valeurs pour les métriques
                # Convertir en int si possible
                if isinstance(y_true, str):
                    try:
                        y_true_norm = int(y_true)
                    except ValueError:
                        y_true_norm = y_true
                else:
                    y_true_norm = int(y_true) if isinstance(y_true, (int, float)) else y_true
                
                if isinstance(y_pred, str):
                    try:
                        y_pred_norm = int(y_pred)
                    except ValueError:
                        y_pred_norm = y_pred
                else:
                    y_pred_norm = int(y_pred) if isinstance(y_pred, (int, float)) else y_pred
                
                # Mettre à jour les métriques
                accuracy.update(y_true_norm, y_pred_norm)
                microf1.update(y_true_norm, y_pred_norm)
                macrof1.update(y_true_norm, y_pred_norm)
                
                # Afficher
                correct = "✓" if y_true_norm == y_pred_norm else "✗"
                acc_val = accuracy.get()
                micro_val = microf1.get()
                macro_val = macrof1.get()
                
                # Formater les valeurs
                acc_str = f"{acc_val:.4f}" if acc_val is not None else "N/A"
                micro_str = f"{micro_val:.4f}" if micro_val is not None else "N/A"
                macro_str = f"{macro_val:.4f}" if macro_val is not None else "N/A"
                
                # Afficher le label original si conversion faite
                y_true_display = y_true_original if use_temp_csv and y_true_original is not None else y_true_norm
                y_pred_display = index_to_label.get(y_pred_norm, y_pred_norm) if use_temp_csv and isinstance(y_pred_norm, int) and y_pred_norm in index_to_label else y_pred_norm
                
                print(f"{step:<6} {str(y_true_display):<15} {str(y_pred_display):<15} {correct:<8} {acc_str:<10} {micro_str:<10} {macro_str:<10}")
                
            except Exception as e:
                print(f"   ⚠️  Erreur métriques à l'étape {step}: {e}")
                print(f"       y_true={y_true}, y_pred={y_pred}")
        
        # Apprendre
        if x_dict is not None and y_true is not None:
            try:
                model.learn_one(x_dict, y_true)
            except Exception as e:
                if step <= 5:
                    print(f"   ⚠️  Erreur apprentissage à l'étape {step}: {e}")
        
        # Limiter le nombre d'étapes affichées
        if step >= max_steps:
            print()
            print(f"... (arrêt après {max_steps} étapes pour l'affichage)")
            break

print("=" * 80)
print()
print("📈 Métriques finales:")
print(f"   Accuracy: {accuracy.get():.4f}" if accuracy.get() is not None else "   Accuracy: N/A")
print(f"   MicroF1: {microf1.get():.4f}" if microf1.get() is not None else "   MicroF1: N/A")
print(f"   MacroF1: {macrof1.get():.4f}" if macrof1.get() is not None else "   MacroF1: N/A")
print()
# Nettoyer le fichier temporaire si créé
if use_temp_csv and temp_csv_path and os.path.exists(temp_csv_path.name):
    try:
        os.unlink(temp_csv_path.name)
    except Exception:
        pass

print("✅ Terminé!")

