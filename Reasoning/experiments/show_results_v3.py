import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

# --- Styles et Configurations pour les Graphiques ---
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)
# NOUVELLE PALETTE (Bleu séquentiel)
DIFFICULTY_PALETTE = {
    "easy":      "#6acc64",  # Vert clair
    "medium":    "#41823c",  # Vert forêt
    "hard":      "#4d4d4d",  # Gris foncé
    "very_hard": "#000000"   # Noir
}
MODEL_PALETTE = {
    "small": "#a2cffe",   # Bleu pastel clair
    "medium": "#4682b4",  # Bleu acier
    "large": "#000080",   # Bleu marine
    "extra large": "#000033" # Bleu encore plus foncé
}

DIFFICULTY_ORDER = ["easy", "medium", "hard","very_hard"]

def get_task_name_from_config_string(config_str):
    if "EntailConfig" in config_str: return "Entailment"
    if "SelectionConfig" in config_str: return "Selection"
    if "ReconstructionConfig" in config_str: return "Reconstruction"
    return "Unknown"

def get_difficulty_from_depth(task_name, depth):
    if task_name == "Reconstruction":
        if depth <= 1: return "easy"
        if depth == 2: return "medium"
        if depth == 3: return "hard"
        return "very_hard"
    else: # Entailment & Selection
        if depth <= 1: return "easy"
        if depth <= 2: return "medium"
        if depth == 3: return "hard"
        return "very_hard"

def load_and_preprocess_data(jsonl_path):
    if not os.path.exists(jsonl_path):
        print(f"Error: File not found at {jsonl_path}")
        return None

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    records = []
    for line in tqdm(lines, desc="Parsing JSONL file"):
        try:
            data = json.loads(line)
            metadata = data.get("metadata", {})
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            
            task_name = get_task_name_from_config_string(data.get("config", ""))
            proof_depth = metadata.get("proof_depth", 0)

            for model_name, response_data in data.get("llm_responses", {}).items():
                if "nano" in model_name: model_alias = "small : gpt5:nano"
                elif "mini" in model_name: model_alias = "medium : gpt5:mini"
                elif "openai/gpt-5" == model_name: model_alias = "large : gpt5"
                else: model_alias = "unknown"
                
                records.append({
                    "task": task_name,
                    "difficulty": get_difficulty_from_depth(task_name, proof_depth),
                    "domain": metadata.get("axiom_set", "N/A")[:3],
                    "model_alias": model_alias,
                    "score": float(response_data.get("score", 0.0))
                })
        except Exception as e:
            print(f"Skipping malformed line: {e}")
            continue
            
    df = pd.DataFrame(records)
    if not df.empty:
        df['difficulty'] = pd.Categorical(df['difficulty'], categories=DIFFICULTY_ORDER, ordered=True)
    return df

### --- NOUVELLES FONCTIONS DE PLOTTING --- ###

def plot_perf_by_task_and_difficulty(df, output_dir="plots"):
    """
    GRAPHIQUE 1 : Performance par tâche et par difficulté (tous domaines confondus).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculer le score moyen par modèle, tâche et difficulté
    summary = df.groupby(['model_alias', 'task', 'difficulty'])['score'].mean().reset_index()
    
    # Créer un graphique en barres groupées
    g = sns.catplot(
        data=summary,
        kind="bar",
        x="task",
        y="score",
        hue="difficulty",
        col="model_alias", # Un sous-graphique par modèle
        col_order=["small : gpt5:nano", "medium : gpt5:mini", "large : gpt5" ],
        palette=DIFFICULTY_PALETTE,
        height=6,
        aspect=0.8,
        legend_out=False
    )
    
    g.fig.suptitle('Performance by Task and Difficulty (All Domains)', fontsize=20, y=1.03)
    g.set_axis_labels("Task Type", "Average Score")
    g.set_titles("Model: {col_name}")
    g.set(ylim=(0, 1))
    g.add_legend(title='Difficulty')
    
    output_path = os.path.join(output_dir, 'perf_by_task_difficulty.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved plot to {output_path}")
    plt.close()


def plot_perf_by_domain_stacked(df, output_dir="plots"):
    """
    GRAPHIQUE 2 : Performance par domaine, empilée par difficulté (toutes tâches confondues).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Calculer le score moyen par modèle, domaine et difficulté
    summary = df.groupby(['model_alias', 'domain', 'difficulty'])['score'].mean().reset_index()
    
    # Normaliser les scores pour que la somme par barre (modèle, domaine) fasse 100%
    # Cela montre la *contribution* de chaque difficulté.
    summary['total_score'] = summary.groupby(['model_alias', 'domain'])['score'].transform('sum')
    summary['score_contribution'] = summary['score'] / summary['total_score']
    
    # Pivoter les données pour le plotting
    pivot_df = summary.pivot_table(index=['model_alias', 'domain'], columns='difficulty', values='score').fillna(0)
    
    # Créer un graphique en barres empilées
    fig, ax = plt.subplots(figsize=(12, 7))
    pivot_df.plot(
        kind='bar',
        stacked=True,
        color=[DIFFICULTY_PALETTE[d] for d in DIFFICULTY_ORDER],
        ax=ax,
        width=0.8
    )
    
    ax.set_title('Performance by Domain (All Tasks)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Model and Domain', fontsize=14)
    ax.set_ylabel('Average Score', fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Difficulty Level')
    ax.set_ylim(0, pivot_df.sum(axis=1).max() * 1.1) # Ajuster la limite Y
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'perf_by_domain_stacked.png')
    plt.savefig(output_path, dpi=300)
    print(f"✅ Saved plot to {output_path}")
    plt.close()


# --- POINT D'ENTRÉE PRINCIPAL ---
if __name__ == "__main__":
    
    results_file_path = "dataset_v4_gpt5.jsonl"
    plots_output_dir = "plotes_gpt"
    
    # 1. Charger et préparer les données
    results_df = load_and_preprocess_data(results_file_path)
    
    if results_df is not None and not results_df.empty:
        print(f"\\nLoaded {len(results_df)} total records successfully.")
        
        # 2. Générer Graphique 1
        print("\n--- Generating plot 1: Performance by Task and Difficulty ---")
        plot_perf_by_task_and_difficulty(results_df, plots_output_dir)
        
        # 3. Générer Graphique 2
        print("\n--- Generating plot 2: Performance by Domain (Stacked by Difficulty) ---")
        plot_perf_by_domain_stacked(results_df, plots_output_dir)
        
        print("\nAnalysis and plotting complete.")
    else:
        print("No valid data found to plot. Please check the input file and its format.")
    
 
 