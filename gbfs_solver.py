"""
Implémentation de l'algorithme Greedy Best First Search pour le Sudoku.
"""
import heapq
import time
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass

from sudoku_state import SudokuState


@dataclass
class SearchStats:
    """Statistiques de la recherche."""
    success: bool
    states_developed: int
    max_open_size: int
    execution_time: float
    final_state: Optional[SudokuState] = None
    
    def __str__(self) -> str:
        status = "SUCCÈS" if self.success else "ÉCHEC"
        return f"""=== Statistiques GBFS ===
Résultat: {status}
États développés: {self.states_developed}
Taille max open list: {self.max_open_size}
Temps d'exécution: {self.execution_time:.4f}s
"""


class Node:
    """
    Nœud pour la recherche GBFS.
    Contient un état et sa valeur heuristique.
    """
    
    def __init__(self, state: SudokuState, heuristic_value: float):
        self.state = state
        self.h = heuristic_value
    
    def __lt__(self, other):
        """Comparaison pour la file de priorité (plus petit h = priorité plus haute)."""
        return self.h < other.h
    
    def __eq__(self, other):
        """Égalité basée sur l'état."""
        if not isinstance(other, Node):
            return False
        return self.state == other.state
    
    def __hash__(self):
        """Hash basé sur l'état."""
        return hash(self.state)


class GBFSSolver:
    """
    Solveur Sudoku utilisant Greedy Best First Search.
    """
    
    def __init__(self, heuristic_function: Callable[[SudokuState], float]):
        """
        Initialise le solveur.
        
        Args:
            heuristic_function: Fonction heuristique à utiliser
        """
        self.heuristic_function = heuristic_function
        self.stats = None
    
    def solve(self, initial_state: SudokuState, max_iterations: int = 100000) -> SearchStats:
        """
        Résout le Sudoku avec GBFS.
        
        Args:
            initial_state: État initial du Sudoku
            max_iterations: Nombre maximum d'itérations pour éviter les boucles infinies
            
        Returns:
            SearchStats avec les résultats de la recherche
        """
        start_time = time.perf_counter()
        
        # Vérifier si l'état initial est déjà une solution
        if initial_state.is_goal():
            end_time = time.perf_counter()
            return SearchStats(
                success=True,
                states_developed=0,
                max_open_size=0,
                execution_time=end_time - start_time,
                final_state=initial_state
            )
        
        # Initialiser les structures de données
        open_list = []  # File de priorité (heap)
        closed_set = set()  # États déjà visités
        
        # Ajouter l'état initial à la open list
        initial_node = Node(initial_state, self.heuristic_function(initial_state))
        heapq.heappush(open_list, initial_node)
        
        # Statistiques
        states_developed = 0
        max_open_size = 1
        
        # Boucle principale GBFS
        iteration = 0
        while open_list and iteration < max_iterations:
            iteration += 1
            
            # Mettre à jour la taille max de la open list
            max_open_size = max(max_open_size, len(open_list))
            
            # Prendre le nœud avec la meilleure heuristique (plus petit h)
            current_node = heapq.heappop(open_list)
            current_state = current_node.state
            
            # Marquer l'état comme visité
            closed_set.add(current_state)
            states_developed += 1
            
            # Vérifier si on a atteint le but
            if current_state.is_goal():
                end_time = time.perf_counter()
                return SearchStats(
                    success=True,
                    states_developed=states_developed,
                    max_open_size=max_open_size,
                    execution_time=end_time - start_time,
                    final_state=current_state
                )
            
            # Générer les successeurs
            successors = current_state.generate_successors()
            
            for successor_state in successors:
                # Ignorer si déjà visité
                if successor_state in closed_set:
                    continue
                
                # Créer un nœud pour le successeur
                h_value = self.heuristic_function(successor_state)
                successor_node = Node(successor_state, h_value)
                
                # Vérifier si déjà dans la open list
                # (Simplification: on peut ajouter des doublons, heapq gérera)
                heapq.heappush(open_list, successor_node)
        
        # Échec: pas de solution trouvée
        end_time = time.perf_counter()
        return SearchStats(
            success=False,
            states_developed=states_developed,
            max_open_size=max_open_size,
            execution_time=end_time - start_time,
            final_state=None
        )
    
    def solve_with_params(self, initial_state: SudokuState, 
                         heuristic_params: Dict[str, Any] = None,
                         max_iterations: int = 100000) -> SearchStats:
        """
        Résout avec des paramètres pour l'heuristique (utile pour h3).
        
        Args:
            initial_state: État initial
            heuristic_params: Paramètres à passer à la fonction heuristique
            max_iterations: Limite d'itérations
            
        Returns:
            SearchStats
        """
        if heuristic_params is None:
            return self.solve(initial_state, max_iterations)
        
        # Sauvegarder la fonction originale AVANT de créer le wrapper
        original_function = self.heuristic_function
        
        # Créer une fonction wrapper qui utilise les paramètres
        def parameterized_heuristic(state: SudokuState) -> float:
            # Vérifier si la fonction accepte des paramètres supplémentaires
            import inspect
            sig = inspect.signature(original_function)  # Utiliser original_function !
            if len(sig.parameters) > 1:  # Plus que juste 'state'
                return original_function(state, **heuristic_params)
            else:
                # Fonction sans paramètres supplémentaires (h1, h2)
                return original_function(state)
        
        # Temporairement remplacer la fonction heuristique
        self.heuristic_function = parameterized_heuristic
        
        try:
            result = self.solve(initial_state, max_iterations)
        finally:
            # Restaurer la fonction originale
            self.heuristic_function = original_function
        
        return result