from lm_eval.tasks import TaskManager

tm = TaskManager()
# all_tasks = tm.all_tasks

# mmlu_group = tm.all_tasks
# default_subjects = [t for t in tm.all_tasks
#                     if t.startswith("mmlu_")
#                     and not t.endswith("_generative")
#                     and not t.endswith("_continuation")]
# print("default mmlu subjects:", len(default_subjects))
# print(sorted(default_subjects)[:5])


# Ask the harness to resolve the "mmlu" group directly
# task_dict = tm.load_task_or_group("mmlu")
# resolved = sorted(task_dict.keys())
# print("mmlu expands to:", len(resolved), "subjects")
# print(resolved[:5])
# print(resolved[-3:])


task_dict = tm.load_task_or_group("mmlu")
resolved = sorted(task_dict.keys())

print("total:", len(resolved))
for name in resolved:
    print(name)