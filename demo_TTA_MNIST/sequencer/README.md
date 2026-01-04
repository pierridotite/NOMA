# MNIST Stream Generator with Distribution Shift

Ce dossier contient un système complet de génération de flux MNIST séquentiels et reproductibles avec rupture de distribution, conçu pour les expériences de Test-Time Adaptation.

## Vue d'ensemble

Le simulateur génère un flux temporel d'images MNIST où :
- **Avant t0** : données "clean" originales MNIST
- **Après t0** : données progressivement dégradées par des perturbations contrôlées
- **Format** : chunks safetensors + manifest JSON pour consommation par NOMA et Python

## Architecture

### Composants principaux

1. **`generate_stream.py`** : Générateur principal
   - Charge MNIST (train ou test)
   - Crée un flux séquentiel déterministe
   - Applique des perturbations progressives après t0
   - Exporte en chunks safetensors

2. **`verify_stream.py`** : Vérificateur d'intégrité
   - Vérifie le format des chunks
   - Valide la continuité temporelle
   - Confirme la rupture de distribution à t0

3. **`visualize_stream.py`** : Outils de visualisation
   - Évolution de l'intensité de perturbation
   - Séquence de transition autour de t0
   - Comparaison statistique clean vs drift
   - Rapport complet avec graphiques

## Installation

```bash
# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation rapide

### 1. Générer un flux

```bash
# Configuration par défaut (60k samples, t0=30000)
python generate_stream.py --output-dir ./mnist_stream

# Configuration personnalisée
python generate_stream.py \
  --output-dir ./my_stream \
  --seed 42 \
  --t0 20000 \
  --chunk-size 1000 \
  --perturbation combined \
  --max-intensity 0.8 \
  --ramp-duration 5000 \
  --shuffle
```

**Options importantes** :
- `--seed` : Graine aléatoire (défaut: 42)
- `--use-test` : Utiliser le test set au lieu du train (10k vs 60k)
- `--flatten` : Format [784] au lieu de [1, 28, 28]
- `--shuffle` : Mélanger l'ordre des échantillons
- `--t0` : Point de bascule (défaut: 30000)
- `--perturbation` : Type de perturbation (gaussian_noise, brightness, contrast, combined)
- `--max-intensity` : Intensité maximale (0.0-1.0, défaut: 0.8)
- `--ramp-duration` : Durée de la montée en intensité (défaut: 5000)

### 2. Vérifier le flux

```bash
python verify_stream.py ./mnist_stream
```

Effectue les vérifications suivantes :
- Format et types des tenseurs
- Continuité des indices temporels
- Présence de la rupture à t0
- Statistiques par phase

### 3. Visualiser le flux

```bash
# Rapport complet (recommandé)
python visualize_stream.py ./mnist_stream --mode report

# Visualisations individuelles
python visualize_stream.py ./mnist_stream --mode intensity
python visualize_stream.py ./mnist_stream --mode transition
python visualize_stream.py ./mnist_stream --mode comparison
python visualize_stream.py ./mnist_stream --mode samples --times 0 10000 30000 40000
```

## Format de sortie

### Structure du répertoire

```
mnist_stream/
├── manifest.json              # Métadonnées complètes
├── chunk_0000.safetensors     # Échantillons 0-999
├── chunk_0001.safetensors     # Échantillons 1000-1999
├── ...
└── chunk_0059.safetensors     # Derniers échantillons
```

### Format des chunks

Chaque fichier `.safetensors` contient :
- **`x`** : Images, shape `[N, 1, 28, 28]` ou `[N, 784]`, dtype `float32`, valeurs dans [0, 1]
- **`y`** : Labels, shape `[N]`, dtype `int64`, valeurs dans [0, 9]
- **`t`** : Indices temporels, shape `[N]`, dtype `int64`, séquentiels
- **`phase`** : Phase, shape `[N]`, dtype `int64`, 0=clean, 1=drift
- **`intensity`** : Intensité de perturbation, shape `[N]`, dtype `float32`, dans [0, 1]

### Manifest JSON

Le fichier `manifest.json` contient toutes les informations nécessaires pour interpréter le flux :

```json
{
  "version": "1.0",
  "seed": 42,
  "dataset": {
    "name": "MNIST",
    "split": "train",
    "n_samples": 60000,
    "shuffled": false
  },
  "format": {
    "image_shape": [1, 28, 28],
    "flattened": false,
    "normalized": true,
    "dtype": "float32"
  },
  "stream": {
    "chunk_size": 1000,
    "n_chunks": 60,
    "chunk_files": ["chunk_0000.safetensors", ...]
  },
  "distribution_shift": {
    "t0": 30000,
    "perturbation_type": "gaussian_noise",
    "schedule": {
      "type": "linear_ramp",
      "t0": 30000,
      "max_intensity": 0.8,
      "ramp_duration": 5000
    }
  }
}
```

## Types de perturbations

### `gaussian_noise`
Ajoute un bruit gaussien d'intensité croissante.
- Meilleur pour simuler du bruit de capteur
- σ = intensity × 0.5

### `brightness`
Réduit progressivement la luminosité.
- Simule des conditions d'éclairage changeantes
- factor = 1 - intensity × 0.7

### `contrast`
Réduit le contraste vers du gris uniforme.
- Simule une perte de définition
- Interpolation vers valeur 0.5

### `combined` (recommandé)
Combinaison de bruit et luminosité.
- Plus réaliste pour des dégradations réelles
- Dégradation progressive et visible

## Consommation du flux

### En Python

```python
from safetensors.numpy import load_file
import json

# Charger le manifest
with open("mnist_stream/manifest.json") as f:
    manifest = json.load(f)

# Charger un chunk
chunk = load_file("mnist_stream/chunk_0000.safetensors")
images = chunk["x"]  # [1000, 1, 28, 28]
labels = chunk["y"]  # [1000]
times = chunk["t"]   # [1000]

# Itérer sur le flux
for chunk_file in manifest["stream"]["chunk_files"]:
    chunk = load_file(f"mnist_stream/{chunk_file}")
    for i in range(len(chunk["y"])):
        x, y, t = chunk["x"][i], chunk["y"][i], chunk["t"][i]
        # Traiter l'échantillon...
```

### En NOMA (à implémenter)

Le format safetensors est directement compatible avec NOMA via les primitives d'I/O existantes.

## Reproductibilité

Le système garantit la reproductibilité totale :

1. **Seed globale** : Fixe l'ordre de shuffle et les perturbations
2. **Ordering explicite** : L'ordre des échantillons est documenté dans le manifest
3. **Perturbations déterministes** : Même seed → mêmes transformations
4. **Format binaire stable** : Safetensors garantit la portabilité

Deux exécutions avec les mêmes paramètres produisent des flux **bit-à-bit identiques**.

## Exemples d'utilisation

### Flux rapide pour debug (test set)

```bash
python generate_stream.py \
  --output-dir ./debug_stream \
  --use-test \
  --t0 5000 \
  --chunk-size 500 \
  --ramp-duration 2000
```

### Flux complet pour benchmark

```bash
python generate_stream.py \
  --output-dir ./benchmark_stream \
  --seed 42 \
  --t0 30000 \
  --chunk-size 1000 \
  --perturbation combined \
  --max-intensity 0.85 \
  --ramp-duration 5000 \
  --shuffle
```

### Flux avec dégradation brutale

```bash
python generate_stream.py \
  --output-dir ./brutal_shift \
  --t0 25000 \
  --ramp-duration 100 \
  --max-intensity 1.0
```

## Métriques et validation

Le script de vérification calcule automatiquement :
- Statistiques par phase (mean, std, min, max)
- Distribution des labels
- Continuité temporelle
- Intensité de perturbation

Pour une analyse approfondie, utilisez le mode visualisation qui génère :
- Évolution temporelle de l'intensité
- Histogrammes de distribution
- Images moyennes et écarts-types par phase
- Séquences de transition

## Limitations et extensions futures

### Limitations actuelles
- Perturbations limitées aux transformations pixel-wise
- Pas de transformations géométriques (rotation, translation)
- Pas de simulation de blur/flou

### Extensions possibles
- Perturbations géométriques (rotation, shear, scale)
- Blur gaussien progressif
- Occlusions aléatoires
- Changements de background
- Perturbations adversariales
- Multiple distribution shifts
- Schedules non-linéaires (exponentiel, par paliers)

## Intégration avec NOMA

Le flux généré est prêt pour la consommation par NOMA. Les prochaines étapes :

1. Implémenter un loader NOMA pour safetensors chunks
2. Créer un modèle baseline NOMA (CNN simple)
3. Implémenter les métriques TTA (accuracy online, adaptation loss)
4. Comparer avec baselines Python (PyTorch eager, compiled)

## Support

Pour toute question ou problème :
1. Vérifier que toutes les dépendances sont installées
2. Exécuter le script de vérification
3. Consulter les visualisations pour diagnostic
4. Vérifier les logs de génération

---

**Version** : 1.0  
**Date** : Janvier 2026  
**License** : Voir LICENSE du projet parent
