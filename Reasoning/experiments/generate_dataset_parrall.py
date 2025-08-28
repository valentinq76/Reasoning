import os


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")


import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import argparse
import random
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from easydict import EasyDict as edict

# Importation des classes de tâches et de configurations
from reasoning_core.tasks.formal_maths import (
    TheoremEntailmentTask, EntailConfig,
    PremiseSelectionTask, SelectionConfig,
    ProofReconstructionTask, ReconstructionConfig
)

# ---------------------------
# Utils
# ---------------------------

def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, edict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    return obj

def set_global_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

# ---------------------------
# Worker
# ---------------------------
def _worker_generate_one(task_class, config_instance, job_idx: int, base_seed: int | None,
                         max_retries: int = 100, per_attempt_timeout_sec: int | None = 20):
    import random
    import os
    import signal
    from time import monotonic

    # Seed déterministe
    if base_seed is not None:
        seed = base_seed + int(job_idx)
        random.seed(seed)
        try:
            import numpy as np
            np.random.seed(seed)
        except Exception:
            pass
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

    # Eviter sur-parallélisation intra-process
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

    task = task_class(config=config_instance)

    def _handle_timeout(signum, frame):
        raise TimeoutError("generate() attempt timed out")

    use_sigalrm = (per_attempt_timeout_sec and os.name == "posix")
    old_handler = None
    if use_sigalrm:
        old_handler = signal.signal(signal.SIGALRM, _handle_timeout)

    attempts = 0
    last_error = None
    try:
        while attempts < max_retries:
            attempts += 1
            try:
                if use_sigalrm:
                    signal.alarm(int(per_attempt_timeout_sec))
                problem = task.generate()
                if use_sigalrm:
                    signal.alarm(0)

                if problem:
                    problem.prompt = task.prompt(problem.metadata)
                    problem_dict = {
                        "prompt": problem.prompt,
                        "answer": make_json_serializable(problem.answer),
                        "metadata": make_json_serializable(problem.metadata),
                        "config": str(config_instance),
                        "domain": str(getattr(config_instance, "domains", None)),
                    }
                    return job_idx, problem_dict
            except TimeoutError as e:
                last_error = e
                # On réessaie avec un nouvel essai
                continue
            except Exception as e:
                last_error = e
                continue
    finally:
        if use_sigalrm and old_handler is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    raise RuntimeError(f"Worker failed after {max_retries} attempts for index {job_idx}. Last error: {last_error}")
# ---------------------------
# Orchestration parallèle
# ---------------------------
def generate_and_save_problems_parallel(task_class, config_instance, num_problems, output_file,
                                        workers: int, base_seed: int | None,
                                        per_attempt_timeout_sec: int = 20,
                                        max_retries: int = 100):
    import copy
    config_copy = copy.deepcopy(config_instance)
    if workers is None or workers <= 0:
        workers = os.cpu_count() or 1

    results = {}
    futures = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as executor:
        for i in range(num_problems):
            cfg_i = copy.deepcopy(config_copy)
            fut = executor.submit(
                _worker_generate_one, task_class, cfg_i, i, base_seed, max_retries, per_attempt_timeout_sec
            )
            futures.append(fut)

        from tqdm import tqdm
        with tqdm(total=num_problems, desc=f"Generating {task_class.__name__}") as pbar:
            for fut in as_completed(futures):
                idx, problem_dict = fut.result()  # si un worker échoue, on le voit ici
                results[idx] = problem_dict
                pbar.update(1)

    with open(output_file, "a", encoding="utf-8") as f:
        for i in range(num_problems):
            f.write(json.dumps(results[i], ensure_ascii=False) + "\n")

# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SynTask datasets for LLM evaluation (parallel).")
    parser.add_argument("--num_per_config", type=int, default=50, help="Nombre d'exemples à générer par configuration.")
    parser.add_argument("--output_file", type=str, default="dataset_hardt1t2.jsonl", help="Fichier JSONL de sortie.")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 1),
                        help="Nombre de workers en parallèle (processus).")
    parser.add_argument("--seed", type=int, default=12345,
                        help="Seed global pour une génération reproductible. Mettre -1 pour désactiver.")
    args = parser.parse_args()
    parser.add_argument("--per_attempt_timeout", type=int, default=50,
                    help="Timeout (s) par tentative de generate(); 0 = désactivé.")

    base_seed = None if (args.seed is None or args.seed < 0) else int(args.seed)

    # Option: supprimer l'ancien fichier
    if os.path.exists(args.output_file):
        os.remove(args.output_file)
        print(f"Removed existing dataset file: {args.output_file}")

    # Boucle identique à la version séquentielle (même ordre de sections)
    work_domains = ['FLD', 'GEO', 'TOP']
    all_domains = ['ALG','FLD', 'GEO','SET', 'TOP']
    

    for domains in all_domains:
        print("\n--- Task 1: Entailment Verification ---")
        config_e_easy = EntailConfig()
        config_e_easy.proof_depth = 1
        config_e_easy.perturbation = 2
        config_e_easy.domains = [domains]
        generate_and_save_problems_parallel(TheoremEntailmentTask, config_e_easy, args.num_per_config, args.output_file, args.workers, base_seed)

        config_e_medium = EntailConfig()
        config_e_medium.proof_depth = 2
        config_e_medium.perturbation = 3
        config_e_medium.domains = [domains]
        generate_and_save_problems_parallel(TheoremEntailmentTask, config_e_medium, args.num_per_config, args.output_file, args.workers, base_seed)

        config_e_hard = EntailConfig()
        config_e_hard.proof_depth = 3
        config_e_hard.perturbation = 4
        config_e_hard.domains = [domains]
        generate_and_save_problems_parallel(TheoremEntailmentTask, config_e_hard, args.num_per_config, args.output_file, args.workers, base_seed)

        print("\n--- Task 2: Premise Selection ---")
        config_s_easy = SelectionConfig()
        config_s_easy.proof_depth = 1
        config_s_easy.num_distractors = 2
        config_s_easy.domains = [domains]
        generate_and_save_problems_parallel(PremiseSelectionTask, config_s_easy, args.num_per_config, args.output_file, args.workers, base_seed)

        config_s_medium = SelectionConfig()
        config_s_medium.proof_depth = 2
        config_s_medium.num_distractors = 4
        config_s_medium.domains = [domains]
        generate_and_save_problems_parallel(PremiseSelectionTask, config_s_medium, args.num_per_config, args.output_file, args.workers, base_seed)

        config_s_hard = SelectionConfig()
        config_s_hard.proof_depth = 3
        config_s_hard.num_distractors = 6
        config_s_hard.domains = [domains]
        generate_and_save_problems_parallel(PremiseSelectionTask, config_s_hard, args.num_per_config, args.output_file, args.workers, base_seed)

        print("\n--- Task 3: Proof Reconstruction ---")
        config_r_easy = ReconstructionConfig()
        config_r_easy.proof_depth = 1
        config_r_easy.domains = [domains]
        generate_and_save_problems_parallel(ProofReconstructionTask, config_r_easy, args.num_per_config, args.output_file, args.workers, base_seed, per_attempt_timeout_sec=50)

        config_r_medium = ReconstructionConfig()
        config_r_medium.proof_depth = 2
        config_r_medium.domains = [domains]
        generate_and_save_problems_parallel(ProofReconstructionTask, config_r_medium, args.num_per_config, args.output_file, args.workers, base_seed)

        config_r_hard = ReconstructionConfig()
        config_r_hard.proof_depth = 3
        config_r_hard.domains = [domains]
        generate_and_save_problems_parallel(ProofReconstructionTask, config_r_hard, args.num_per_config, args.output_file, args.workers, base_seed)

    print(f"\n✅ Dataset generation complete. All problems saved to {args.output_file}")