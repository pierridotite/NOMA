# Static Model Baseline

Ce dossier contient le baseline "modèle statique" pour les expériences de Test-Time Adaptation sur MNIST.

## Objectif

Démontrer la dégradation de performance d'un modèle statique (non adaptatif) lorsqu'il rencontre une rupture de distribution :
- **Avant t0** : Données clean MNIST → haute précision
- **Après t0** : Données dégradées → chute de précision

Cette baseline servira de référence pour comparer les méthodes d'adaptation (LoRA/NOMA).

## Architecture du Modèle

MLP simple avec la structure suivante :
```
Input (784) → Linear(256) → ReLU → Dropout(0.2)
           → Linear(128) → ReLU → Dropout(0.2)
           → Linear(10) → Output
```

## Utilisation

### Installation

```bash
pip install -r requirements.txt
```

### Exécution

```bash
# Configuration par défaut
python static_model_baseline.py

# Configuration personnalisée
python static_model_baseline.py \
  --stream-dir ../sequencer/mnist_stream \
  --epochs 15 \
  --window-size 500 \
  --output-dir ./output \
  --save-model
```

### Options principales

| Option | Défaut | Description |
|--------|--------|-------------|
| `--stream-dir` | `./sequencer/mnist_stream` | Répertoire du flux MNIST |
| `--data-root` | `./sequencer/mnist_data` | Répertoire des données d'entraînement |
| `--epochs` | 10 | Nombre d'époques d'entraînement |
| `--batch-size` | 64 | Taille de batch |
| `--lr` | 1e-3 | Learning rate |
| `--seed` | 42 | Seed pour reproductibilité |
| `--hidden-sizes` | [256, 128] | Tailles des couches cachées |
| `--window-size` | 500 | Taille de fenêtre pour rolling accuracy |
| `--output-dir` | `./baseline/output` | Répertoire de sortie |
| `--save-model` | False | Sauvegarder le checkpoint du modèle |

## Sorties

Le script génère trois fichiers dans le répertoire de sortie :

### 1. `static_baseline_accuracy.png`

Visualisation comprenant :
- **Courbe principale** : Rolling accuracy en fonction du temps
- **Ligne verticale rouge** : Point de bascule t0
- **Zones colorées** : Phase clean (vert) et phase drift (rouge)
- **Courbe secondaire** : Intensité de la perturbation

### 2. `static_baseline_metrics.csv`

Métriques détaillées pour chaque sample :

| Colonne | Description |
|---------|-------------|
| `t` | Index temporel |
| `correct` | 1 si prédiction correcte, 0 sinon |
| `predicted` | Chiffre prédit (0-9) |
| `actual` | Label réel (0-9) |
| `phase` | 'clean' ou 'drift' |
| `intensity` | Intensité de perturbation [0, 1] |
| `rolling_accuracy` | Précision sur fenêtre glissante (%) |

### 3. `static_baseline_summary.json`

Résumé statistique :
- Paramètres d'entraînement
- Précision finale sur test clean
- Précision par phase (clean/drift)
- Chute de précision

## Résultats attendus

Avec la configuration par défaut et le flux gaussian_noise (intensity max 0.8) :

| Métrique | Valeur attendue |
|----------|-----------------|
| Précision sur test clean | ~97-98% |
| Précision phase clean (streaming) | ~97-98% |
| Précision phase drift (streaming) | ~60-80% |
| Chute de précision | ~20-35% |

La courbe devrait montrer clairement :
1. Un plateau stable avant t0
2. Une chute progressive après t0 (pendant la rampe)
3. Une stabilisation à un niveau inférieur après la rampe

## Comparaison avec l'adaptation

Cette baseline fournit la référence pour démontrer l'efficacité de l'adaptation :

```
Sans adaptation (ce baseline):  97% → ~70% (chute de ~27%)
Avec adaptation (LoRA/NOMA):    97% → ~90% (récupération partielle)
```

## Format de données

Le script attend un flux au format défini par `generate_stream.py` :
- Images : `[1, 28, 28]`, float32, normalisées [0, 1]
- Labels : int64, 0-9
- Chunks safetensors avec manifest.json

## Reproductibilité

Le script fixe toutes les seeds (numpy, torch, CUDA) pour garantir la reproductibilité.
La même seed donnera exactement les mêmes résultats sur le même hardware.
