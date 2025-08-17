"""
Fonctions heuristiques pour la recherche GBFS sur le Sudoku.
"""
from sudoku_state import SudokuState


def h1_empty_cells(state: SudokuState) -> int:
    """
    Heuristique h1: Nombre de cases vides.
    Plus il y a de cases vides, plus on est loin du but.
    
    Args:
        state: État du Sudoku
        
    Returns:
        Nombre de cases vides (int)
    """
    return len(state.get_empty_cells())


def h2_domain_sum(state: SudokuState) -> int:
    """
    Heuristique h2: Somme des tailles des domaines de toutes les cases vides.
    Plus les domaines sont grands, plus on a d'options (donc potentiellement plus loin du but).
    
    Args:
        state: État du Sudoku
        
    Returns:
        Somme des tailles des domaines (int)
    """
    # TODO: À implémenter
    # Calculer la somme des tailles des domaines pour toutes les cases vides
    # Utiliser state.get_all_domains() pour obtenir tous les domaines
    domains = state.get_all_domains()
    return sum(len(domain) for domain in domains.values())


def h3_weighted_combination(state: SudokuState, w1: float = 1.0, w2: float = 0.1) -> float:
    """
    Heuristique h3: Combinaison pondérée de h1 et h2.
    h3 = w1 * h1 + w2 * h2
    
    Args:
        state: État du Sudoku
        w1: Poids pour h1 (nombre de cases vides)
        w2: Poids pour h2 (somme des domaines)
        
    Returns:
        Valeur heuristique pondérée (float)
    """
    # TODO: À implémenter
    # Combiner h1 et h2 avec les poids donnés
    return w1 * h1_empty_cells(state) + w2 * h2_domain_sum(state)


# Dictionnaire des heuristiques disponibles
HEURISTICS = {
    'h1': h1_empty_cells,
    'h2': h2_domain_sum,
    'h3': h3_weighted_combination,
}


def get_heuristic_function(name: str):
    """
    Récupère une fonction heuristique par son nom.
    
    Args:
        name: Nom de l'heuristique ('h1', 'h2', 'h3')
        
    Returns:
        Fonction heuristique correspondante
        
    Raises:
        ValueError: Si le nom n'est pas reconnu
    """
    if name not in HEURISTICS:
        raise ValueError(f"Heuristique inconnue: {name}. Disponibles: {list(HEURISTICS.keys())}")
    
    return HEURISTICS[name]