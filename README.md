# Sudoku Solver — Heuristic Search (GBFS)

Résolution de Sudoku 9×9 par **Greedy Best-First Search** guidé par des **heuristiques** (cases vides, somme des domaines, combinaison pondérée), génération de successeurs **valides** et choix de variable **MRV**. Objectif : comparer le guidage heuristique (succès, nœuds explorés, temps).

## Installation
```bash
git clone https://github.com/salmalys/sudoku-solver.git
cd sudoku-solver
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
.\.venv\Scripts\activate
pip install -r requirements.txt

python main.py        # exécution par défaut
python main.py -h     # aide (choix d’heuristique, seed, chemin de grille, etc.)
```

## Notebook d’expériences
jupyter notebook sudoku_experiments.ipynb

## Structure (rôle de chaque fichier)

- `main.py` — Point d’entrée/CLI : charge la grille, sélectionne l’heuristique, lance GBFS, affiche métriques.
- `gbfs_solver.py` — Moteur GBFS : file de priorité, expansion d’états valides, critères d’arrêt.
- `heuristics.py` — Heuristiques : cases vides, somme des domaines, combinaison pondérée.
- `sudoku_state.py` — État + contraintes (lignes/colonnes/blocs), génération de successeurs, sélection **MRV**.
- `sudoku_examples.py` — Grilles d’exemple (facile → difficile) pour tests rapides.
- `dataset_analyzer.py` — Chargement/analyse de datasets, stats basiques.
- `visualization.py` — Fonctions d’affichage/graphes (optionnel).
- `sudoku_experiments.ipynb` — Expériences (comparaisons d’heuristiques, métriques).
- `data/` — Grilles d’entrée et éventuelles sorties.
- `requirements.txt` — Dépendances Python du projet.
