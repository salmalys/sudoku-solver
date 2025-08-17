"""
Exemples de grilles de Sudoku pour les tests.
"""
from typing import Dict
from sudoku_state import SudokuState


def create_easy_example() -> SudokuState:
    """
    Crée un exemple de Sudoku facile (presque résolu).
    """
    grid = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]
    return SudokuState(grid)


def create_medium_example() -> SudokuState:
    """
    Crée un exemple de Sudoku de difficulté moyenne.
    """
    grid = [
        [0, 0, 0, 6, 0, 0, 4, 0, 0],
        [7, 0, 0, 0, 0, 3, 6, 0, 0],
        [0, 0, 0, 0, 9, 1, 0, 8, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 5, 0, 1, 8, 0, 0, 0, 3],
        [0, 0, 0, 3, 0, 6, 0, 4, 5],
        [0, 4, 0, 2, 0, 0, 0, 6, 0],
        [9, 0, 3, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 1, 0, 0]
    ]
    return SudokuState(grid)


def create_hard_example() -> SudokuState:
    """
    Crée un exemple de Sudoku difficile.
    """
    grid = [
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [4, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 5, 0, 4, 0, 7],
        [0, 0, 8, 0, 0, 0, 3, 0, 0],
        [0, 0, 1, 0, 9, 0, 0, 0, 0],
        [3, 0, 0, 4, 0, 0, 2, 0, 0],
        [0, 5, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 8, 0, 6, 0, 0, 0]
    ]
    return SudokuState(grid)


def create_minimal_example() -> SudokuState:
    """
    Crée un exemple très simple avec peu de cases à remplir.
    """
    grid = [
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 0]  # Une seule case vide
    ]
    return SudokuState(grid)


def create_almost_solved_example() -> SudokuState:
    """
    Crée un exemple presque résolu (2-3 cases vides).
    """
    grid = [
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 0, 5],
        [3, 4, 5, 2, 8, 6, 1, 0, 0]
    ]
    return SudokuState(grid)


def create_empty_grid() -> SudokuState:
    """
    Crée une grille complètement vide (pour tests extrêmes).
    """
    grid = [[0 for _ in range(9)] for _ in range(9)]
    return SudokuState(grid)


# Dictionnaire des exemples disponibles
EXAMPLES = {
    'minimal': create_minimal_example,
    'almost_solved': create_almost_solved_example,
    'easy': create_easy_example,
    'medium': create_medium_example,
    'hard': create_hard_example,
    'empty': create_empty_grid,
}


def get_example(name: str) -> SudokuState:
    """
    Récupère un exemple de Sudoku par son nom.
    
    Args:
        name: Nom de l'exemple
        
    Returns:
        SudokuState correspondant
        
    Raises:
        ValueError: Si le nom n'est pas reconnu
    """
    if name not in EXAMPLES:
        raise ValueError(f"Exemple inconnu: {name}. Disponibles: {list(EXAMPLES.keys())}")
    
    return EXAMPLES[name]()


def list_examples() -> Dict[str, str]:
    """
    Liste tous les exemples disponibles avec leurs descriptions.
    
    Returns:
        Dict {nom: description}
    """
    descriptions = {
        'minimal': 'Une seule case vide (très facile)',
        'almost_solved': '2-3 cases vides (test rapide)',
        'easy': 'Sudoku facile avec plusieurs cases vides',
        'medium': 'Sudoku de difficulté moyenne',
        'hard': 'Sudoku difficile (beaucoup de cases vides)',
        'empty': 'Grille complètement vide (test extrême)'
    }
    return descriptions