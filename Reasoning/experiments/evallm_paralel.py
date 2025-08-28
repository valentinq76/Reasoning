
import json
import os
import asyncio
import httpx
from tqdm.auto import tqdm
import re
import random
import tempfile

# Import de vos modules spécifiques
from reasoning_core.tasks.formal_maths import (
    TheoremEntailmentTask,
    PremiseSelectionTask,
    ProofReconstructionTask,
)
from reasoning_core.template import Problem

# --- CONFIGURATION ---
api_key = os.environ["OPENROUTER_API_KEY"]

# 1. Fichiers d'entrée et de sortie
INPUT_DATASET_PATH = "dataset_v5gpt5.jsonl"
OUTPUT_DATASET_PATH = "dataset_v5.jsonl"

# 2. Modèles à évaluer
MODELS_TO_EVALUATE = [
    'openai/gpt-5-nano',
    'openai/gpt-5-mini',
    'openai/gpt-5'
]

# 3. Clé d'API et paramètres
API_KEY = os.getenv("OPENROUTER_API_KEY")
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "100"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "3000"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "120"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def model_supports_reasoning(model: str) -> bool:
    """
    Retourne True uniquement pour certains modèles OpenAI 'o3/o4' (exemples),
    afin d'éviter les 400 si le champ 'reasoning' n'est pas supporté.
    Ajustez selon vos besoins/catalogue.
    """
    m = model.lower()
    return m.startswith("openai/")


async def async_complete_reasoning(client, semaphore, prompt, model, effort='low', max_retries=MAX_RETRIES):
    """Version asynchrone qui interroge le LLM avec gestion de la concurrence et des erreurs."""
    async with semaphore:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            # "HTTP-Referer": "https://votre-app.example",  # Optionnel mais recommandé par OpenRouter
            # "X-Title": "reasoning-eval-script"            # Optionnel
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Use reasoning but with a maximun of 2048 token."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": MAX_TOKENS
        }
        if model_supports_reasoning(model):
            payload["reasoning"] = {"effort": effort}
        for attempt in range(max_retries):
            try:
                response = await client.post(
                    OPENROUTER_URL,
                    headers=headers,
                    json=payload,
                    timeout=REQUEST_TIMEOUT
                )
                response.raise_for_status()

                # Parsing robuste
                try:
                    data = response.json()
                    choices = data.get("choices", [])
                    if not choices:
                        print(f"Réponse malformée (choices vide) pour {model}.")
                        return {"reasoning": None, "answer": None}

                    message = choices[0].get("message", {})
                    return {
                        "reasoning": message.get("reasoning"),
                        "answer": message.get("content")
                    }
                except (KeyError, ValueError, IndexError) as e:
                    print(f"Réponse malformée pour {model} (pas de nouvel essai): {e}")
                    return {"reasoning": None, "answer": None}

            except httpx.HTTPStatusError as e:
                status = e.response.status_code if e.response else "N/A"
                body_snippet = e.response.text[:500] if e.response else ""
                if status in (429, 503):
                    # Rate limit / Service indisponible -> respect Retry-After si présent
                    retry_after = e.response.headers.get("Retry-After") if e.response else None
                    if retry_after:
                        try:
                            wait = float(retry_after)
                        except ValueError:
                            wait = min(30.0, 2.0 ** attempt) + random.uniform(0, 0.25)
                    else:
                        wait = min(30.0, 2.0 ** attempt) + random.uniform(0, 0.25)
                    print(f"HTTP {status} pour {model} (essai {attempt+1}/{max_retries}), attente {wait:.1f}s. Détail: {body_snippet}")
                    await asyncio.sleep(wait)
                    continue
                else:
                    print(f"Erreur HTTP {status} pour {model} (essai {attempt+1}/{max_retries}). Détail: {body_snippet}")

            except httpx.RequestError as e:
                print(f"Erreur de requête pour {model} (essai {attempt+1}/{max_retries}): {e}")

            # Backoff générique si on va réessayer
            if attempt < max_retries - 1:
                wait = 1.5 * (attempt + 1)
                await asyncio.sleep(wait)

        print(f"Échec de la requête pour {model} après {max_retries} essais.")
        return {"reasoning": None, "answer": None}


def extract_final_answer(response_text):
    """
    Extrait la réponse finale en cherchant des balises <result></result>.
    En cas d'échec, se rabat sur la recherche de True/False.
    """
    if not isinstance(response_text, str):
        return ""

    match = re.search(r"<result>(.*?)</result>", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()

    matches_true_false = re.findall(r'\b(True|False)\b', response_text, re.IGNORECASE)
    if matches_true_false:
        return matches_true_false[-1].capitalize()

    return response_text.strip()


def save_results_atomic(data_list, path=OUTPUT_DATASET_PATH):
    """Sauvegarde atomique de la liste de dictionnaires dans un fichier JSONL."""
    print(f"\nSauvegarde des résultats enrichis dans '{path}'...")
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)

    with tempfile.NamedTemporaryFile('w', delete=False, dir=directory, encoding='utf-8') as tmp:
        for item in data_list:
            tmp.write(json.dumps(item, ensure_ascii=False) + '\n')
        temp_name = tmp.name

    os.replace(temp_name, path)
    print("✅ Opération terminée avec succès !")


async def main():
    """Orchestre le chargement, l'évaluation parallèle et la sauvegarde des données."""
    if not API_KEY:
        print("ERREUR : La variable d'environnement OPENROUTER_API_KEY n'est pas configurée.")
        return

    # 1. Chargement des données
    print(f"Chargement des problèmes depuis '{INPUT_DATASET_PATH}'...")
    try:
        with open(INPUT_DATASET_PATH, 'r', encoding='utf-8') as f:
            problems_list = [json.loads(line) for line in f]
        print(f"-> {len(problems_list)} problèmes chargés.")
    except FileNotFoundError:
        print(f"ERREUR: Fichier d'entrée non trouvé : '{INPUT_DATASET_PATH}'.")
        return
    except json.JSONDecodeError as e:
        print(f"ERREUR: Le fichier semble ne pas être un JSONL valide. Erreur : {e}")
        return

    # 2. Préparer les tâches à exécuter
    tasks_to_run = []
    for i, problem in enumerate(problems_list):
        if 'prompt' not in problem:
            continue

        problem.setdefault('llm_responses', {})

        for model_name in MODELS_TO_EVALUATE:
            if model_name not in problem['llm_responses']:
                tasks_to_run.append({
                    "problem_index": i,
                    "model_name": model_name,
                    "prompt": problem['prompt'],
                    "metadata": problem.get('metadata', {}),
                    "ground_truth_answer": problem.get('answer')
                })

    if not tasks_to_run:
        print("✅ Aucune nouvelle évaluation à effectuer. Toutes les réponses sont déjà présentes.")
        save_results_atomic(problems_list, OUTPUT_DATASET_PATH)
        return

    print(f"\nDébut de l'évaluation : {len(tasks_to_run)} nouvelles réponses à générer...")

    # 3. Lancement asynchrone avec appariement correct (as_completed + index)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:

        async def run_one(idx, task):
            r = await async_complete_reasoning(
                client, semaphore, task['prompt'], task['model_name']
            )
            return idx, r

        tasks = [asyncio.create_task(run_one(i, t)) for i, t in enumerate(tasks_to_run)]
        results = [None] * len(tasks)

        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Évaluation des modèles"):
            idx, r = await fut
            results[idx] = r

    # 4. Traiter et fusionner les résultats (avec garde sur answer=None)
    print("Fusion des nouveaux résultats...")
    for task_info, llm_response in zip(tasks_to_run, results):
        if llm_response is None:
            llm_response = {"reasoning": None, "answer": None}

        problem_index = task_info['problem_index']
        model_name = task_info['model_name']
        metadata = task_info['metadata']

        ans_raw = llm_response.get("answer")
        final_ans = ans_raw
        #final_ans = extract_final_answer(ans_raw) if ans_raw else None

        score = None
        if final_ans is not None:
            problem_obj_for_scoring = Problem(metadata, task_info['ground_truth_answer'])
            if "perturbation" in metadata:
                T = TheoremEntailmentTask()
                score = T.score_answer(answer=final_ans, entry=problem_obj_for_scoring)
            elif "num_distractors" in metadata:
                T = PremiseSelectionTask()
                score = T.score_answer(answer=final_ans, entry=problem_obj_for_scoring)
            elif "correct_proof_structure_indices" in metadata:
                T = ProofReconstructionTask()
                score = T.score_answer(answer=final_ans, entry=problem_obj_for_scoring)

        llm_response['score'] = score
        problems_list[problem_index]['llm_responses'][model_name] = llm_response

    # 5. Enregistrer le fichier final (atomique)
    save_results_atomic(problems_list, OUTPUT_DATASET_PATH)


if __name__ == "__main__":
    asyncio.run(main())
