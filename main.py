"""
Script principal pour tester le solveur GBFS sur différents exemples de Sudoku.
"""
from heuristics import get_heuristic_function
from gbfs_solver import GBFSSolver
from visualization import print_state, print_domains_info
from sudoku_examples import get_example, list_examples


def test_single_example(example_name: str, heuristic_name: str, 
                       show_details: bool = True, heuristic_params: dict = None):
    """
    Teste le solveur sur un exemple donné avec une heuristique donnée.
    
    Args:
        example_name: Nom de l'exemple à tester
        heuristic_name: Nom de l'heuristique à utiliser
        show_details: Afficher les détails (grilles, domaines)
        heuristic_params: Paramètres pour l'heuristique (pour h3)
    """
    print(f"\n{'='*60}")
    print(f"TEST: {example_name.upper()} avec heuristique {heuristic_name.upper()}")
    print(f"{'='*60}")
    
    try:
        # Charger l'exemple
        initial_state = get_example(example_name)
        
        if show_details:
            print_state(initial_state, "État initial")
            print_domains_info(initial_state)
        
        # Créer le solveur
        heuristic_func = get_heuristic_function(heuristic_name)
        solver = GBFSSolver(heuristic_func)
        
        # Résoudre
        print(f"\nLancement de GBFS avec {heuristic_name}...")
        
        if heuristic_params:
            stats = solver.solve_with_params(initial_state, heuristic_params)
            print(f"Paramètres heuristique: {heuristic_params}")
        else:
            stats = solver.solve(initial_state)
        
        # Afficher les résultats
        print(stats)
        
        if stats.success and show_details:
            print_state(stats.final_state, "Solution trouvée")
        
        return stats
        
    except Exception as e:
        print(f"ERREUR: {e}")
        return None


def compare_heuristics(example_name: str = 'easy'):
    """
    Compare les différentes heuristiques sur un même exemple.
    
    Args:
        example_name: Nom de l'exemple à utiliser
    """
    print(f"\n{'='*80}")
    print(f"COMPARAISON DES HEURISTIQUES SUR L'EXEMPLE '{example_name.upper()}'")
    print(f"{'='*80}")
    
    heuristics_to_test = ['h1', 'h2']
    results = {}
    
    # Test des heuristiques de base
    for h_name in heuristics_to_test:
        print(f"\n--- Test avec {h_name} ---")
        stats = test_single_example(example_name, h_name, show_details=False)
        if stats:
            results[h_name] = stats
    
    # Test de h3 avec différents poids
    h3_configs = [
        {'w1': 1.0, 'w2': 0.1},
        {'w1': 1.0, 'w2': 0.5},
        {'w1': 0.5, 'w2': 1.0},
    ]
    
    for i, params in enumerate(h3_configs):
        h_name = f"h3_config_{i+1}"
        print(f"\n--- Test avec h3 (w1={params['w1']}, w2={params['w2']}) ---")
        stats = test_single_example(example_name, 'h3', show_details=False, 
                                  heuristic_params=params)
        if stats:
            results[h_name] = stats
    
    # Résumé des résultats
    print(f"\n{'='*80}")
    print("RÉSUMÉ DE LA COMPARAISON")
    print(f"{'='*80}")
    print(f"{'Heuristique':<15} {'Succès':<8} {'États dév.':<12} {'Temps (s)':<10} {'Max open':<10}")
    print("-" * 80)
    
    for h_name, stats in results.items():
        success_str = "✓" if stats.success else "✗"
        print(f"{h_name:<15} {success_str:<8} {stats.states_developed:<12} "
              f"{stats.execution_time:<10.4f} {stats.max_open_size:<10}")


def interactive_test():
    """
    Mode interactif pour tester différentes combinaisons.
    """
    print("\n=== MODE INTERACTIF ===")
    print("Exemples disponibles:")
    for name, desc in list_examples().items():
        print(f"  {name}: {desc}")
    
    print("\nHeuristiques disponibles:")
    print("  h1: Nombre de cases vides")
    print("  h2: Somme des domaines")
    print("  h3: Combinaison pondérée")
    
    while True:
        print("\n" + "-" * 50)
        example = input("Choisir un exemple (ou 'quit' pour quitter): ").strip()
        
        if example.lower() == 'quit':
            break
        
        if example not in list_examples():
            print(f"Exemple inconnu. Disponibles: {list(list_examples().keys())}")
            continue
        
        heuristic = input("Choisir une heuristique (h1, h2, h3): ").strip()
        
        if heuristic not in ['h1', 'h2', 'h3']:
            print("Heuristique inconnue. Disponibles: h1, h2, h3")
            continue
        
        heuristic_params = None
        if heuristic == 'h3':
            try:
                w1 = float(input("Poids w1 (défaut 1.0): ") or "1.0")
                w2 = float(input("Poids w2 (défaut 0.1): ") or "0.1")
                heuristic_params = {'w1': w1, 'w2': w2}
            except ValueError:
                print("Poids invalides, utilisation des valeurs par défaut.")
                heuristic_params = {'w1': 1.0, 'w2': 0.1}
        
        test_single_example(example, heuristic, show_details=True, 
                          heuristic_params=heuristic_params)


def main():
    """
    Fonction principale avec différents modes de test.
    """
    print("===================== SOLVEUR SUDOKU AVEC GREEDY BEST FIRST SEARCH =====================")
    print("Implémentation avec heuristiques h1, h2 et h3")
    
    # Test rapide sur un exemple simple
    test_single_example('minimal', 'h1', show_details=True)
    
    # Comparaison des heuristiques
    compare_heuristics('easy')
    
    # Mode interactif
    try:
        interactive_test()
    except KeyboardInterrupt:
        print("\n\nAu revoir! 👋")


if __name__ == "__main__":
    main()