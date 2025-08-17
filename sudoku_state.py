"""
Représentation d'état pour le Sudoku et calcul des domaines.
"""
from typing import List, Set, Tuple, Optional

class SudokuState:
    """
    Représente un état du Sudoku (grille 9x9).
    Gère le calcul des domaines (valeurs possibles) pour chaque case vide.
    """
    
    def __init__(self, grid: List[List[int]]):
        """
        Initialise un état Sudoku.
        
        Args:
            grid: Grille 9x9 avec des 0 pour les cases vides
        """
        self.grid = [row[:] for row in grid]  # Copie profonde
        self.size = 9
        self._domains_cache = None
        
    def __hash__(self):
        """Hash pour utiliser SudokuState dans un set (closed list)."""
        return hash(tuple(tuple(row) for row in self.grid))
    
    def __eq__(self, other):
        """Égalité pour la closed list."""
        if not isinstance(other, SudokuState):
            return False
        return self.grid == other.grid
    
    def copy(self) -> 'SudokuState':
        """Crée une copie de l'état."""
        return SudokuState(self.grid)
    
    def get_empty_cells(self) -> List[Tuple[int, int]]:
        """Retourne la liste des coordonnées des cases vides."""
        empty = []
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j] == 0:
                    empty.append((i, j))
        return empty
    
    def is_goal(self) -> bool:
        """Vérifie si l'état est un état but (grille complète et valide)."""
        # Vérifier qu'il n'y a plus de cases vides
        if self.get_empty_cells():
            return False
        
        # Vérifier la validité de la grille complète
        return self.is_valid()
    
    def is_valid(self) -> bool:
        """Vérifie si la grille actuelle est valide (pas de violations)."""
        # Vérifier les lignes
        for i in range(self.size):
            row = [self.grid[i][j] for j in range(self.size) if self.grid[i][j] != 0]
            if len(row) != len(set(row)):
                return False
        
        # Vérifier les colonnes
        for j in range(self.size):
            col = [self.grid[i][j] for i in range(self.size) if self.grid[i][j] != 0]
            if len(col) != len(set(col)):
                return False
        
        # Vérifier les blocs 3x3
        for block_row in range(3):
            for block_col in range(3):
                block = []
                for i in range(3):
                    for j in range(3):
                        r = block_row * 3 + i
                        c = block_col * 3 + j
                        if self.grid[r][c] != 0:
                            block.append(self.grid[r][c])
                if len(block) != len(set(block)):
                    return False
        
        return True
    
    def get_domain(self, row: int, col: int) -> Set[int]:
        """
        Calcule le domaine (valeurs possibles) pour une case donnée.
        
        Args:
            row, col: Coordonnées de la case
            
        Returns:
            Set des valeurs possibles (1-9)
        """
        if self.grid[row][col] != 0:
            return set()  # Case déjà remplie
        
        # Commencer avec toutes les valeurs possibles
        domain = set(range(1, 10))
        
        # Exclure les valeurs de la ligne
        for j in range(self.size):
            if self.grid[row][j] != 0:
                domain.discard(self.grid[row][j])
        
        # Exclure les valeurs de la colonne
        for i in range(self.size):
            if self.grid[i][col] != 0:
                domain.discard(self.grid[i][col])
        
        # Exclure les valeurs du bloc 3x3
        block_row = row // 3
        block_col = col // 3
        for i in range(3):
            for j in range(3):
                r = block_row * 3 + i
                c = block_col * 3 + j
                if self.grid[r][c] != 0:
                    domain.discard(self.grid[r][c])
        
        return domain
    
    def get_all_domains(self) -> dict:
        """
        Calcule tous les domaines pour toutes les cases vides.
        
        Returns:
            Dict {(row, col): set(values)} pour chaque case vide
        """
        if self._domains_cache is None:
            self._domains_cache = {}
            for row, col in self.get_empty_cells():
                self._domains_cache[(row, col)] = self.get_domain(row, col)
        
        return self._domains_cache
    
    def get_mrv_cell(self) -> Optional[Tuple[int, int]]:
        """
        Trouve la case vide avec le plus petit domaine (MRV - Minimum Remaining Values).
        
        Returns:
            (row, col) de la case MRV, ou None si plus de cases vides
        """
        domains = self.get_all_domains()
        if not domains:
            return None
        
        # Trouver la case avec le plus petit domaine non-vide
        min_domain_size = float('inf')
        mrv_cell = None
        
        for (row, col), domain in domains.items():
            if len(domain) == 0:
                # Domaine vide = impasse
                continue
            if len(domain) < min_domain_size:
                min_domain_size = len(domain)
                mrv_cell = (row, col)
        
        return mrv_cell
    
    def set_cell(self, row: int, col: int, value: int) -> 'SudokuState':
        """
        Crée un nouvel état avec une case remplie.
        
        Args:
            row, col: Coordonnées de la case
            value: Valeur à placer
            
        Returns:
            Nouvel état SudokuState
        """
        new_state = self.copy()
        new_state.grid[row][col] = value
        new_state._domains_cache = None  # Invalider le cache
        return new_state
    
    def generate_successors(self) -> List['SudokuState']:
        """
        Génère tous les successeurs légaux en utilisant MRV.
        
        Returns:
            Liste des états successeurs valides
        """
        mrv_cell = self.get_mrv_cell()
        if mrv_cell is None:
            return []  # Plus de cases vides
        
        row, col = mrv_cell
        domain = self.get_domain(row, col)
        
        successors = []
        for value in domain:
            new_state = self.set_cell(row, col, value)
            # On ne génère que des états valides par construction
            if new_state.is_valid():
                successors.append(new_state)
        
        return successors