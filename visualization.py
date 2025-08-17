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
        grid: Grille 9x9 (. pour les cases vides)
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