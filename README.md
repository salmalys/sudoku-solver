# Sudoku Solver — Heuristic Search (GBFS)

Résolution de grilles de Sudoku de taille 9×9 par **Greedy Best-First Search** en utilisant plusieurs fonctions heuristiques (cases vides, somme des domaines, combinaison pondérée).

L'objectif du projet est de comparer l'impact de chaque heuristique sur les performances de l'algorithme (temps d'exécution, nombre d'états explorés, taille maximale de la file ouverte, taux de réussite).

## Installation du projet 
```bash
git clone https://github.com/salmalys/sudoku-solver.git
cd sudoku-solver

# Création d'un environnement virtuel 
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
.\.venv\Scripts\activate

# Installation des librairies nécessaires 
pip install -r requirements.txt
```
Pour tester la résolution des grilles de Sudoku, un script interactif peut être lancé pour choisir l’heuristique, le niveau de difficulté de la grille, etc..
```bash
python main.py        
```
## Notebook d’expériences
```bash
jupyter notebook sudoku_experiments.ipynb
```
## Structure 

- `main.py` — Point d’entrée/CLI : charge la grille, sélectionne l’heuristique, lance GBFS, affiche métriques.
- `gbfs_solver.py` — Moteur GBFS : file de priorité, expansion d’états valides, critères d’arrêt.
- `heuristics.py` — Heuristiques : cases vides, somme des domaines, combinaison pondérée.
- `sudoku_state.py` — État + contraintes (lignes/colonnes/blocs), génération de successeurs, sélection **MRV**.
- `sudoku_examples.py` — Grilles d’exemple (facile → difficile) pour tests rapides.
- `dataset_analyzer.py` — Chargement/analyse de datasets, stats basiques.
- `visualization.py` — Fonctions d’affichage/graphes (optionnel).
- `sudoku_experiments.ipynb` — Expériences (comparaisons d’heuristiques, métriques).
- `data/` — Dossier contenant le dataset de grilles de Sudoku
- `requirements.txt` — Dépendances Python du projet.
