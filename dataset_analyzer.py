"""
Analyseur pour le dataset de Sudoku au format CSV.
Utilise les fonctions déjà implémentées pour analyser et valider les grilles.
"""
import pandas as pd
import ast
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from sudoku_state import SudokuState
from heuristics import h1_empty_cells, h2_domain_sum
from visualization import print_state, print_domains_info


class SudokuDatasetAnalyzer:
    """
    Analyseur pour le dataset de grilles de Sudoku.
    """
    
    def __init__(self, csv_path: str):
        """
        Initialise l'analyseur avec le fichier CSV.
        
        Args:
            csv_path: Chemin vers le fichier CSV
        """
        self.csv_path = csv_path
        self.df = None
        self.sudoku_states = []
        self.analysis_results = {}
        
    def load_dataset(self) -> pd.DataFrame:
        """
        Charge le dataset CSV et convertit les grilles.
        
        Returns:
            DataFrame pandas avec les données
        """
        print("📂 Chargement du dataset...")
        
        # Charger le CSV
        self.df = pd.read_csv(self.csv_path)
        print(f"   ✅ {len(self.df)} grilles chargées")
        
        # Convertir les grilles string en listes
        print("🔄 Conversion des grilles...")
        grids = []
        valid_indices = []
        
        for idx, row in self.df.iterrows():
            try:
                # Convertir la string représentant la grille en liste Python
                grid_str = row['grid']
                grid_list = ast.literal_eval(grid_str)
                
                # Vérifier que c'est bien une grille 9x9
                if len(grid_list) == 9 and all(len(row) == 9 for row in grid_list):
                    grids.append(grid_list)
                    valid_indices.append(idx)
                else:
                    print(f"   ⚠️ Grille {idx} invalide: taille incorrecte")
                    
            except Exception as e:
                print(f"   ❌ Erreur parsing grille {idx}: {e}")
        
        # Créer les SudokuState
        self.sudoku_states = []
        for i, grid in enumerate(grids):
            try:
                state = SudokuState(grid)
                self.sudoku_states.append(state)
            except Exception as e:
                print(f"   ❌ Erreur création SudokuState {valid_indices[i]}: {e}")
                valid_indices.remove(valid_indices[i])
        
        # Filtrer le DataFrame pour ne garder que les grilles valides
        self.df = self.df.loc[valid_indices].reset_index(drop=True)
        
        print(f"   ✅ {len(self.sudoku_states)} SudokuStates créés avec succès")
        return self.df
    
    def validate_grids(self) -> Dict[str, any]:
        """
        Valide toutes les grilles du dataset.
        
        Returns:
            Dictionnaire avec les résultats de validation
        """
        print("\n🔍 VALIDATION DES GRILLES")
        print("=" * 40)
        
        validation_results = {
            'total_grids': len(self.sudoku_states),
            'valid_grids': 0,
            'invalid_grids': 0,
            'invalid_indices': [],
            'validation_details': []
        }
        
        for i, state in enumerate(self.sudoku_states):
            row_data = self.df.iloc[i]
            
            # Valider la grille
            is_valid = state.is_valid()
            is_goal = state.is_goal()
            empty_count = len(state.get_empty_cells())
            
            detail = {
                'index': i,
                'id': row_data['id'],
                'level': row_data['level'],
                'declared_givens': row_data['givens'],
                'actual_givens': 81 - empty_count,
                'empty_cells': empty_count,
                'is_valid': is_valid,
                'is_solved': is_goal
            }
            
            validation_results['validation_details'].append(detail)
            
            if is_valid:
                validation_results['valid_grids'] += 1
                status = "✅"
            else:
                validation_results['invalid_grids'] += 1
                validation_results['invalid_indices'].append(i)
                status = "❌"
            
            print(f"{status} Grille {i:3d} ({row_data['id']}): "
                  f"valide={is_valid}, cases_vides={empty_count}, "
                  f"givens_déclarés={row_data['givens']}")
        
        self.analysis_results['validation'] = validation_results
        
        print(f"\n📊 RÉSUMÉ VALIDATION:")
        print(f"   Total: {validation_results['total_grids']}")
        print(f"   Valides: {validation_results['valid_grids']} "
              f"({validation_results['valid_grids']/validation_results['total_grids']*100:.1f}%)")
        print(f"   Invalides: {validation_results['invalid_grids']}")
        
        if validation_results['invalid_indices']:
            print(f"   Indices invalides: {validation_results['invalid_indices']}")
        
        return validation_results
    
    def analyze_difficulty_levels(self) -> Dict[str, any]:
        """
        Analyse les niveaux de difficulté du dataset.
        
        Returns:
            Statistiques par niveau de difficulté
        """
        print("\n📈 ANALYSE PAR NIVEAU DE DIFFICULTÉ")
        print("=" * 40)
        
        if 'validation' not in self.analysis_results:
            self.validate_grids()
        
        # Créer DataFrame des détails
        details_df = pd.DataFrame(self.analysis_results['validation']['validation_details'])
        
        # Statistiques par niveau
        level_stats = {}
        for level in details_df['level'].unique():
            level_data = details_df[details_df['level'] == level]
            
            stats = {
                'count': len(level_data),
                'valid_percentage': (level_data['is_valid'].sum() / len(level_data)) * 100,
                'avg_empty_cells': level_data['empty_cells'].mean(),
                'std_empty_cells': level_data['empty_cells'].std(),
                'min_empty_cells': level_data['empty_cells'].min(),
                'max_empty_cells': level_data['empty_cells'].max(),
                'givens_accuracy': (level_data['declared_givens'] == level_data['actual_givens']).mean() * 100
            }
            
            level_stats[level] = stats
            
            print(f"\n🎯 Niveau '{level}':")
            print(f"   Nombre de grilles: {stats['count']}")
            print(f"   Pourcentage valides: {stats['valid_percentage']:.1f}%")
            print(f"   Cases vides: {stats['avg_empty_cells']:.1f} ± {stats['std_empty_cells']:.1f}")
            print(f"   Range cases vides: [{stats['min_empty_cells']}, {stats['max_empty_cells']}]")
            print(f"   Précision 'givens': {stats['givens_accuracy']:.1f}%")
        
        self.analysis_results['difficulty_levels'] = level_stats
        return level_stats
    
    def analyze_heuristic_values(self) -> Dict[str, any]:
        """
        Calcule les valeurs heuristiques pour toutes les grilles valides.
        
        Returns:
            Statistiques des valeurs heuristiques
        """
        print("\n🧮 ANALYSE DES VALEURS HEURISTIQUES")
        print("=" * 40)
        
        heuristic_data = []
        
        for i, state in enumerate(self.sudoku_states):
            if state.is_valid():  # Seulement les grilles valides
                row_data = self.df.iloc[i]
                
                # Calculer h1 et h2
                h1_val = h1_empty_cells(state)
                h2_val = h2_domain_sum(state)
                
                # Calculer quelques statistiques sur les domaines
                domains = state.get_all_domains()
                if domains:
                    domain_sizes = [len(d) for d in domains.values()]
                    min_domain = min(domain_sizes)
                    max_domain = max(domain_sizes)
                    avg_domain = np.mean(domain_sizes)
                else:
                    min_domain = max_domain = avg_domain = 0
                
                heuristic_data.append({
                    'index': i,
                    'id': row_data['id'],
                    'level': row_data['level'],
                    'h1': h1_val,
                    'h2': h2_val,
                    'min_domain_size': min_domain,
                    'max_domain_size': max_domain,
                    'avg_domain_size': avg_domain,
                    'total_domains': len(domains)
                })
        
        heuristic_df = pd.DataFrame(heuristic_data)
        
        # Statistiques globales
        print(f"📊 Statistiques heuristiques (sur {len(heuristic_df)} grilles valides):")
        print(f"   h1 (cases vides): {heuristic_df['h1'].mean():.1f} ± {heuristic_df['h1'].std():.1f}")
        print(f"   h2 (somme domaines): {heuristic_df['h2'].mean():.1f} ± {heuristic_df['h2'].std():.1f}")
        print(f"   Taille moyenne domaines: {heuristic_df['avg_domain_size'].mean():.2f}")
        
        # Par niveau de difficulté
        print(f"\n📈 Par niveau de difficulté:")
        for level in heuristic_df['level'].unique():
            level_data = heuristic_df[heuristic_df['level'] == level]
            print(f"   {level}: h1={level_data['h1'].mean():.1f}, "
                  f"h2={level_data['h2'].mean():.1f}, "
                  f"avg_domain={level_data['avg_domain_size'].mean():.2f}")
        
        self.analysis_results['heuristics'] = heuristic_df
        return heuristic_df
    
    def create_visualizations(self) -> None:
        """
        Crée des visualisations des analyses.
        """
        print("\n📊 CRÉATION DES VISUALISATIONS")
        print("=" * 30)
        
        if 'heuristics' not in self.analysis_results:
            self.analyze_heuristic_values()
        
        heuristic_df = self.analysis_results['heuristics']
        
        # Configuration des subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Analyse du Dataset Sudoku', fontsize=16, fontweight='bold')
        
        # 1. Distribution h1 par niveau
        for level in heuristic_df['level'].unique():
            level_data = heuristic_df[heuristic_df['level'] == level]
            axes[0,0].hist(level_data['h1'], alpha=0.7, label=level, bins=15)
        axes[0,0].set_xlabel('h1 (cases vides)')
        axes[0,0].set_ylabel('Fréquence')
        axes[0,0].set_title('Distribution h1 par niveau')
        axes[0,0].legend()
        
        # 2. Distribution h2 par niveau
        for level in heuristic_df['level'].unique():
            level_data = heuristic_df[heuristic_df['level'] == level]
            axes[0,1].hist(level_data['h2'], alpha=0.7, label=level, bins=15)
        axes[0,1].set_xlabel('h2 (somme domaines)')
        axes[0,1].set_ylabel('Fréquence')
        axes[0,1].set_title('Distribution h2 par niveau')
        axes[0,1].legend()
        
        # 3. Corrélation h1 vs h2
        for level in heuristic_df['level'].unique():
            level_data = heuristic_df[heuristic_df['level'] == level]
            axes[0,2].scatter(level_data['h1'], level_data['h2'], alpha=0.7, label=level)
        axes[0,2].set_xlabel('h1 (cases vides)')
        axes[0,2].set_ylabel('h2 (somme domaines)')
        axes[0,2].set_title('Corrélation h1 vs h2')
        axes[0,2].legend()
        
        # 4. Boxplot h1 par niveau
        levels = heuristic_df['level'].unique()
        h1_by_level = [heuristic_df[heuristic_df['level'] == level]['h1'] for level in levels]
        axes[1,0].boxplot(h1_by_level, labels=levels)
        axes[1,0].set_ylabel('h1 (cases vides)')
        axes[1,0].set_title('h1 par niveau (boxplot)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 5. Taille moyenne des domaines
        for level in heuristic_df['level'].unique():
            level_data = heuristic_df[heuristic_df['level'] == level]
            axes[1,1].hist(level_data['avg_domain_size'], alpha=0.7, label=level, bins=15)
        axes[1,1].set_xlabel('Taille moyenne des domaines')
        axes[1,1].set_ylabel('Fréquence')
        axes[1,1].set_title('Distribution taille moyenne domaines')
        axes[1,1].legend()
        
        # 6. Statistiques par niveau (barplot)
        level_summary = heuristic_df.groupby('level').agg({
            'h1': 'mean',
            'h2': 'mean',
            'avg_domain_size': 'mean'
        })
        
        x = np.arange(len(level_summary))
        width = 0.25
        
        axes[1,2].bar(x - width, level_summary['h1'], width, label='h1 (norm)', alpha=0.8)
        axes[1,2].bar(x, level_summary['h2']/10, width, label='h2/10', alpha=0.8)  # Normalisé pour visibilité
        axes[1,2].bar(x + width, level_summary['avg_domain_size'], width, label='avg_domain', alpha=0.8)
        
        axes[1,2].set_xlabel('Niveau de difficulté')
        axes[1,2].set_ylabel('Valeur moyenne')
        axes[1,2].set_title('Moyennes par niveau')
        axes[1,2].set_xticks(x)
        axes[1,2].set_xticklabels(level_summary.index, rotation=45)
        axes[1,2].legend()
        
        plt.tight_layout()
        plt.show()
    
    def show_sample_grids(self, n_samples: int = 3) -> None:
        """
        Affiche quelques grilles d'exemple pour chaque niveau.
        
        Args:
            n_samples: Nombre d'échantillons par niveau
        """
        print(f"\n🔍 ÉCHANTILLONS DE GRILLES ({n_samples} par niveau)")
        print("=" * 50)
        
        if 'heuristics' not in self.analysis_results:
            self.analyze_heuristic_values()
        
        heuristic_df = self.analysis_results['heuristics']
        
        for level in heuristic_df['level'].unique():
            print(f"\n📋 Niveau '{level}':")
            level_data = heuristic_df[heuristic_df['level'] == level]
            
            # Prendre quelques échantillons
            samples = level_data.head(n_samples)
            
            for _, sample in samples.iterrows():
                idx = sample['index']
                state = self.sudoku_states[idx]
                
                print(f"\n   Grille {sample['id']} (h1={sample['h1']}, h2={sample['h2']}):")
                print_state(state, f"")
    
    def generate_report(self) -> str:
        """
        Génère un rapport complet de l'analyse.
        
        Returns:
            Rapport sous forme de string
        """
        print("\n📄 GÉNÉRATION DU RAPPORT COMPLET")
        print("=" * 40)
        
        # S'assurer que toutes les analyses sont faites
        if 'validation' not in self.analysis_results:
            self.validate_grids()
        if 'difficulty_levels' not in self.analysis_results:
            self.analyze_difficulty_levels()
        if 'heuristics' not in self.analysis_results:
            self.analyze_heuristic_values()
        
        validation = self.analysis_results['validation']
        levels = self.analysis_results['difficulty_levels']
        heuristics = self.analysis_results['heuristics']
        
        report = f"""
🧩 RAPPORT D'ANALYSE DU DATASET SUDOKU
=====================================

📊 RÉSUMÉ GÉNÉRAL
-----------------
• Total de grilles: {validation['total_grids']}
• Grilles valides: {validation['valid_grids']} ({validation['valid_grids']/validation['total_grids']*100:.1f}%)
• Grilles invalides: {validation['invalid_grids']}

📈 ANALYSE PAR NIVEAU
--------------------"""
        
        for level, stats in levels.items():
            report += f"""
🎯 Niveau '{level}':
   • Nombre: {stats['count']} grilles
   • Validité: {stats['valid_percentage']:.1f}%
   • Cases vides: {stats['avg_empty_cells']:.1f} ± {stats['std_empty_cells']:.1f}
   • Range: [{stats['min_empty_cells']}, {stats['max_empty_cells']}]
   • Précision givens: {stats['givens_accuracy']:.1f}%"""
        
        report += f"""

🧮 VALEURS HEURISTIQUES
-----------------------
• h1 moyen: {heuristics['h1'].mean():.1f} ± {heuristics['h1'].std():.1f}
• h2 moyen: {heuristics['h2'].mean():.1f} ± {heuristics['h2'].std():.1f}
• Taille domaine moyenne: {heuristics['avg_domain_size'].mean():.2f}

📋 RECOMMANDATIONS
------------------
"""
        
        if validation['invalid_grids'] > 0:
            report += f"⚠️ {validation['invalid_grids']} grilles invalides détectées - à corriger\n"
        
        # Analyser la cohérence des niveaux
        level_h1_means = heuristics.groupby('level')['h1'].mean().sort_values()
        report += f"✅ Ordre de difficulté h1: {' < '.join(level_h1_means.index)}\n"
        
        report += f"""
🎯 UTILISATION RECOMMANDÉE
-------------------------
• Pour tests GBFS faciles: niveau '{level_h1_means.index[0]}'
• Pour tests GBFS moyens: niveau '{level_h1_means.index[len(level_h1_means)//2] if len(level_h1_means) > 1 else level_h1_means.index[0]}'
• Pour tests GBFS difficiles: niveau '{level_h1_means.index[-1]}'

Rapport généré le {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return report


def analyze_sudoku_dataset(csv_path: str, show_samples: bool = True, 
                          create_plots: bool = True) -> SudokuDatasetAnalyzer:
    """
    Fonction principale pour analyser un dataset de Sudoku.
    
    Args:
        csv_path: Chemin vers le fichier CSV
        show_samples: Afficher des échantillons de grilles
        create_plots: Créer les visualisations
        
    Returns:
        Analyseur configuré avec tous les résultats
    """
    print("🚀 ANALYSE COMPLÈTE DU DATASET SUDOKU")
    print("=" * 50)
    
    # Créer l'analyseur
    analyzer = SudokuDatasetAnalyzer(csv_path)
    
    # Charger les données
    analyzer.load_dataset()
    
    # Lancer toutes les analyses
    analyzer.validate_grids()
    analyzer.analyze_difficulty_levels()
    analyzer.analyze_heuristic_values()
    
    # Afficher des échantillons
    if show_samples:
        analyzer.show_sample_grids(n_samples=2)
    
    # Créer les visualisations
    if create_plots:
        analyzer.create_visualizations()
    
    # Générer et afficher le rapport
    report = analyzer.generate_report()
    print(report)
    
    return analyzer