# __init__.py

import importlib
import pkgutil
import ast
import os
from lazy_object_proxy import Proxy
from .tasks import _reasoning_gym

# ✍️ Import the internal registry from template.py.
from .template import _REGISTRY
from . import tasks

def _discover_tasks():
    """
    Parses task files to find all Task subclasses and their names without importing them.
    Returns a mapping of {task_name: module_name}.
    """
    task_map = {}
    tasks_path = tasks.__path__[0]
    for filename in os.listdir(tasks_path):
        if filename.endswith('.py') and not filename.startswith('_'):
            module_name = filename[:-3]
            with open(os.path.join(tasks_path, filename), 'r') as f:
                try:
                    tree = ast.parse(f.read(), filename=filename)
                except SyntaxError:
                    continue  # Skip files with syntax errors

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and any(b.id == 'Task' for b in node.bases if isinstance(b, ast.Name)):
                    # Default task_name is the class name in lowercase
                    task_name = node.name.lower()
                    # Look for an explicit `task_name = "..."` assignment
                    for body_item in node.body:
                        if (isinstance(body_item, ast.Assign) and
                            len(body_item.targets) == 1 and
                            isinstance(body_item.targets[0], ast.Name) and
                            body_item.targets[0].id == 'task_name'):
                            # For Python 3.8+ value is Constant, for older it's Str
                            if isinstance(body_item.value, (ast.Constant, ast.Str)):
                                task_name = body_item.value.s
                            break
                    task_map[task_name] = module_name
    return task_map

def _lazy_loader(task_name, module_name):
    """Triggers the module import and returns the specific task class from the registry."""
    # This import will trigger the __init_subclass__ for all tasks in the file,
    # populating _REGISTRY.
    importlib.import_module(f".tasks.{module_name}", __package__)
    return _REGISTRY[task_name]

# ✍️ This is the single, public-facing dictionary. It is populated once and never changes size.
# First, discover all tasks and their corresponding modules without importing.
_task_to_module_map = _discover_tasks()

# Then, build the DATASETS proxy dictionary with the correct task names.
DATASETS = {
    task_name: Proxy(lambda task=task_name, module=module_name: _lazy_loader(task, module))
    for task_name, module_name in _task_to_module_map.items()
}

scorers = {
    k: Proxy(lambda k=k: lambda answer, entry: DATASETS[k]().score_answer(None, answer, entry))
    for k in DATASETS.keys()
}
scorers['RG'] = _reasoning_gym.RG().score_answer

def get_score_answer_fn(task_name, *args, **kwargs):
    if task_name in scorers:
        return scorers[task_name]
    raise ValueError(f"Task {task_name} not found. Available: {list(DATASETS.keys())}")

def register_to_reasoning_gym():
    import reasoning_gym
    for task_name, task_cls_proxy in DATASETS.items():
        # Accessing the proxy triggers the lazy load
        task = task_cls_proxy()
        if task_name not in reasoning_gym.factory.DATASETS:
            reasoning_gym.register_dataset(task_name, task.__class__, task.config.__class__)

__all__ = ["DATASETS", "get_score_answer_fn", "register_to_reasoning_gym"]