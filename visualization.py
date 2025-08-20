"""
Utilitaires pour visualiser les grilles de Sudoku.
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from sudoku_state import SudokuState


def print_grid(grid: List[List[int]], title: str = ""):
    """
    Affiche une grille de Sudoku dans la console avec des séparateurs pour les blocs 3x3.
    
    Args:
        grid: Grille 9x9 (0 pour les cases vides)
        title: Titre optionnel à afficher
    """
    if title:
        print(f"\n=== {title} ===")
    
    print("┌───────┬───────┬───────┐")
    
    for i in range(9):
        if i == 3 or i == 6:
            print("├───────┼───────┼───────┤")
        
        row_str = "│ "
        for j in range(9):
            if j == 3 or j == 6:
                row_str += "│ "
            
            cell_value = str(grid[i][j]) if grid[i][j] != 0 else "."
            row_str += cell_value + " "
        
        row_str += "│"
        print(row_str)
    
    print("└───────┴───────┴───────┘")


def print_state(state: SudokuState, title: str = ""):
    """
    Affiche un SudokuState.
    
    Args:
        state: État du Sudoku
        title: Titre optionnel
    """
    print_grid(state.grid, title)
    empty_count = len(state.get_empty_cells())
    print(f"Nombre de cases vides: {empty_count}")


def print_verbose_step(iteration: int, state: SudokuState, h_value: float, open_size: int, total_developed: int):
    """
    Affiche les détails d'une étape de GBFS pour le mode verbose.
    
    Args:
        iteration: Numéro d'itération
        state: État actuel
        h_value: Valeur heuristique
        open_size: Taille de la open list
        total_developed: Nombre total d'états développés
    """
    # Afficher info d'étape
    if iteration % 10 == 1 or iteration <= 5:  # Premières étapes + tous les 10
        print(f"\n--- Itération {iteration} ---")
        print(f"h = {h_value:.1f} | Open: {open_size} | Développés: {total_developed}")
        
        # Afficher l'état actuel (compact)
        print_grid(state.grid, f"État itération {iteration}")
        
        # Info sur MRV
        mrv_cell = state.get_mrv_cell()
        if mrv_cell:
            row, col = mrv_cell
            domain = state.get_domain(row, col)
            print(f"MRV: case ({row+1},{col+1}) avec domaine {sorted(domain)} (taille: {len(domain)})")
    
    elif iteration % 100 == 0:  # Résumé tous les 100
        print(f"... Itération {iteration} (h={h_value:.1f}, open={open_size}, dev={total_developed})")


def print_successors_analysis(current_state: SudokuState, successors: list, heuristic_function):
    """
    Affiche l'analyse des successeurs générés.
    
    Args:
        current_state: État actuel
        successors: Liste des états successeurs
        heuristic_function: Fonction heuristique pour calculer les valeurs
    """
    if len(successors) == 0:
        print("Aucun successeur.")
        return
    
    if len(successors) > 3:
        print(f"Trop de successeurs ({len(successors)}) pour affichage détaillé.")
        return
    
    print(f"\n{len(successors)} successeur(s) générés:")
    
    # Trouver la case MRV pour comprendre les choix
    mrv_cell = current_state.get_mrv_cell()
    if mrv_cell:
        row, col = mrv_cell
        domain = current_state.get_domain(row, col)
        print(f"Choix pour case ({row+1},{col+1}): {sorted(domain)}")
    
    # Afficher les successeurs avec leurs valeurs heuristiques
    for i, successor in enumerate(successors):
        h_val = heuristic_function(successor)
        print(f"\nSuccesseur {i+1} (h={h_val:.1f}):")
        
        # Afficher seulement la case modifiée si possible
        if mrv_cell:
            row, col = mrv_cell
            new_value = successor.grid[row][col]
            print(f"  -> Case ({row+1},{col+1}) = {new_value}")
            
        else:
            # Affichage complet si pas de MRV identifiable
            print_grid(successor.grid, f"Successeur {i+1}")


def print_states_side_by_side(states: list, titles: list = None):
    """
    Affiche plusieurs états côte à côte.
    
    Args:
        states: Liste des SudokuState à afficher
        titles: Titres optionnels pour chaque état
    """
    if not states:
        return
    
    if titles is None:
        titles = [f"État {i+1}" for i in range(len(states))]
    
    # Limiter à 3 états maximum pour la lisibilité
    states = states[:3]
    titles = titles[:3]
    
    # Préparer les lignes de chaque grille
    grids_lines = []
    for state in states:
        lines = []
        lines.append("┌───────┬───────┬───────┐")
        
        for i in range(9):
            if i == 3 or i == 6:
                lines.append("├───────┼───────┼───────┤")
            
            row_str = "│ "
            for j in range(9):
                if j == 3 or j == 6:
                    row_str += "│ "
                
                cell_value = str(state.grid[i][j]) if state.grid[i][j] != 0 else "."
                row_str += cell_value + " "
            
            row_str += "│"
            lines.append(row_str)
        
        lines.append("└───────┴───────┴───────┘")
        grids_lines.append(lines)
    
    # Afficher les titres
    title_line = ""
    for i, title in enumerate(titles):
        title_line += f"{title:^23}"
        if i < len(titles) - 1:
            title_line += "   "
    print(f"\n{title_line}")
    
    # Afficher les grilles ligne par ligne
    for line_idx in range(len(grids_lines[0])):
        combined_line = ""
        for grid_idx in range(len(grids_lines)):
            combined_line += grids_lines[grid_idx][line_idx]
            if grid_idx < len(grids_lines) - 1:
                combined_line += "   "  # Espacement entre grilles
        print(combined_line)


def print_grid_extract(grid: List[List[int]], center_row: int, center_col: int, 
                      highlight_value: int = None, title: str = ""):
    """
    Affiche un extrait de grille centré sur une position donnée.
    
    Args:
        grid: Grille 9x9
        center_row: Ligne centrale (0-8)
        center_col: Colonne centrale (0-8)
        highlight_value: Valeur à mettre en évidence à la position centrale
        title: Titre optionnel
    """
    if title:
        print(f"\n{title}:")
    
    start_row = max(0, center_row - 1)
    end_row = min(9, center_row + 2)
    start_col = max(0, center_col - 2)
    end_col = min(9, center_col + 3)
    
    print("Extrait de grille:")
    for r in range(start_row, end_row):
        line = "  "
        for c in range(start_col, end_col):
            val = grid[r][c]
            if r == center_row and c == center_col and highlight_value is not None:
                line += f"{highlight_value} "
            else:
                line += f" {val if val != 0 else '.'} "
            if c == 2 or c == 5:
                line += "│ "
        print(line)


def print_verbose_header(initial_state: SudokuState, heuristic_function):
    """
    Affiche l'en-tête du mode verbose avec informations initiales.
    
    Args:
        initial_state: État initial du Sudoku
        heuristic_function: Fonction heuristique utilisée
    """
    print("DÉMARRAGE GBFS VERBOSE")
    print("=" * 50)
    print_grid(initial_state.grid, "État initial")
    print(f"Cases vides: {len(initial_state.get_empty_cells())}")
    print(f"Valeur heuristique initiale: {heuristic_function(initial_state)}")
    print_domains_info(initial_state)


def print_verbose_footer(success: bool, iteration: int, states_developed: int, 
                        max_open_size: int, max_iterations: int, final_state: SudokuState = None):
    """
    Affiche le pied de page du mode verbose avec résultats finaux.
    
    Args:
        success: Succès ou échec de la résolution
        iteration: Nombre d'itérations effectuées
        states_developed: Nombre d'états développés
        max_open_size: Taille maximale de la open list
        max_iterations: Limite d'itérations
        final_state: État final si solution trouvée
    """
    if success:
        print(f"\nSOLUTION TROUVÉE à l'itération {iteration}!")
        if final_state:
            print_grid(final_state.grid, "SOLUTION FINALE")
    else:
        if iteration >= max_iterations:
            print(f"\nLIMITE D'ITÉRATIONS ATTEINTE ({max_iterations})")
        else:
            print(f"\nÉCHEC - Pas de solution trouvée")
        print(f"États développés: {states_developed}")
        print(f"Taille max open list: {max_open_size}")


def plot_grid_matplotlib(grid: List[List[int]], title: str = "Sudoku", 
                        figsize: tuple = (8, 8), save_path: str = None):
    """
    Affiche une grille de Sudoku avec matplotlib (optionnel, pour une belle visualisation).
    
    Args:
        grid: Grille 9x9
        title: Titre du graphique
        figsize: Taille de la figure
        save_path: Chemin pour sauvegarder l'image (optionnel)
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Créer une grille visuelle
    grid_array = np.array(grid)
    
    # Couleurs: blanc pour vides, gris clair pour remplies
    colors = np.where(grid_array == 0, 0.9, 0.7)
    
    # Afficher la grille avec des couleurs
    im = ax.imshow(colors, cmap='gray', vmin=0, vmax=1)
    
    # Ajouter les chiffres
    for i in range(9):
        for j in range(9):
            if grid[i][j] != 0:
                ax.text(j, i, str(grid[i][j]), ha='center', va='center', 
                       fontsize=14, fontweight='bold', color='black')
            else:
                ax.text(j, i, '', ha='center', va='center', fontsize=14)
    
    # Lignes de grille principales (blocs 3x3)
    for i in range(4):
        ax.axhline(y=i*3-0.5, color='black', linewidth=3)
        ax.axvline(x=i*3-0.5, color='black', linewidth=3)
    
    # Lignes de grille secondaires
    for i in range(10):
        ax.axhline(y=i-0.5, color='gray', linewidth=1, alpha=0.7)
        ax.axvline(x=i-0.5, color='gray', linewidth=1, alpha=0.7)
    
    # Configuration des axes
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-0.5, 8.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Inverser l'axe Y pour avoir (0,0) en haut à gauche
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_state(state: SudokuState, title: str = "Sudoku"):
    """
    Affiche un SudokuState avec matplotlib.
    
    Args:
        state: État du Sudoku
        title: Titre du graphique
    """
    empty_count = len(state.get_empty_cells())
    full_title = f"{title} ({empty_count} cases vides)"
    plot_grid_matplotlib(state.grid, full_title)


def print_domains_info(state: SudokuState):
    """
    Affiche des informations sur les domaines des cases vides.
    
    Args:
        state: État du Sudoku
    """
    domains = state.get_all_domains()
    if not domains:
        print("Aucune case vide.")
        return
    
    print(f"\n=== Informations sur les domaines ({len(domains)} cases vides) ===")
    
    # Trier par taille de domaine
    sorted_domains = sorted(domains.items(), key=lambda x: len(x[1]))
    
    for (row, col), domain in sorted_domains:
        domain_str = "{" + ", ".join(map(str, sorted(domain))) + "}"
        print(f"Case ({row+1},{col+1}): {domain_str} (taille: {len(domain)})")
    
    # Statistiques
    domain_sizes = [len(domain) for domain in domains.values()]
    avg_domain_size = sum(domain_sizes) / len(domain_sizes)
    print(f"\nTaille moyenne des domaines: {avg_domain_size:.2f}")
    print(f"Domaines les plus contraints: {min(domain_sizes)} valeurs")
    print(f"Domaines les moins contraints: {max(domain_sizes)} valeurs")