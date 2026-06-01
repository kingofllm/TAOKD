import os
import subprocess
import itertools
import csv
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

param_grid = {
    "lr": [0.01],
    "temperature": [1],
    "momentum": [0.9],
    "weight_decay": [5e-4],
    "alpha": [0.1],
    "beta": [0.1],
    "epochs": [300],
    "distill": [1],
    "gamma": [0.1],
    "n_heads": [2],
    "n_groups": [2],
    "stride": [1],
    "scheduler_milestones": ["140 200 250"],
    "use_aligned_loss": [True],
    "use_evo_loss": [True],
}

MAX_PARALLEL = 1
output_root = "checkpoint"
os.makedirs(output_root, exist_ok=True)

result_file = os.path.join(output_root, "results.csv")
with open(result_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Run_ID", "Params", "Accuracy", "Loss", "Log_Path"])

param_combinations = list(itertools.product(*param_grid.values()))
param_keys = list(param_grid.keys())
print(f"Total experiments to run: {len(param_combinations)}")

def run_experiment(idx, params):
    params_dict = dict(zip(param_keys, params))
    run_id = f"run_{idx+1:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_path = os.path.join(output_root, f"{run_id}.log")
    output_dir = os.path.join(output_root, run_id)
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "python", "/root/TAOKD/model/train.py",
        "--distill", str(params_dict["distill"]),
        "--epochs", str(params_dict["epochs"]),
        "--lr", str(params_dict["lr"]),
        "--temperature", str(params_dict["temperature"]),
        "--momentum", str(params_dict["momentum"]),
        "--weight_decay", str(params_dict["weight_decay"]),
        "--scheduler_milestones"
    ] + params_dict["scheduler_milestones"].split() + [
        "--gamma", str(params_dict["gamma"]),
        "--n_heads", str(params_dict["n_heads"]),
        "--n_groups", str(params_dict["n_groups"]),
        "--stride", str(params_dict["stride"]),
        "--alpha", str(params_dict["alpha"]),
        "--beta", str(params_dict["beta"]),
        "--use_aligned_loss", str(params_dict["use_aligned_loss"]),
        "--use_evo_loss", str(params_dict["use_evo_loss"]),
        "--output_dir", output_dir
    ]

    print(f"[START] {run_id}")
    start_time = time.time()
    with open(log_path, "w") as log_file:
        subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    end_time = time.time()

    acc = None
    loss = None
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if "Student Model" in line and "Accuracy" in line:
                try:
                    acc = float(line.split("Accuracy:")[1].split("%")[0])
                except:
                    pass
            if "Student Model" in line and "Loss" in line:
                try:
                    loss = float(line.split("Loss:")[1])
                except:
                    pass

    with open(result_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([run_id, params_dict, acc, loss, log_path])

    duration = (end_time - start_time) / 60
    print(f"[END] {run_id} | Acc={acc} | Loss={loss} | Time={duration:.1f} min")
    return run_id, acc, loss

with ProcessPoolExecutor(max_workers=MAX_PARALLEL) as executor:
    futures = []
    for idx, combo in enumerate(param_combinations):
        futures.append(executor.submit(run_experiment, idx, combo))

    for future in as_completed(futures):
        try:
            run_id, acc, loss = future.result()
        except Exception as e:
            print(f"[ERROR] {e}")

print("\n✅ All experiments finished.")
print(f"Results saved to {result_file}")
