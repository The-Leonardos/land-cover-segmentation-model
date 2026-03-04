import torch
from landcover.training import HyperparameterTuning
from landcover import DATA_PATH
import datetime
import wandb
from dotenv import load_dotenv
import os

if __name__ == "__main__":
    load_dotenv("../config.env")
    KEY = os.getenv("WANDB_API_KEY")

    # auth login to weights and biases
    wandb.login(key=KEY)

    # check device availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device]: {device.upper()}")

    # hyperparameter tuning instance
    tuning = HyperparameterTuning(3, 5, device)

    # trial runs
    df, best_params = tuning.run()

    # save results as csv
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{date_time}_deeplabv3plus_tuning_results.csv"
    csv_path = DATA_PATH / "hyperparameter_tuning_logs" / filename
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    # print best parameters
    print("\n\n\nHyperparameter tuning completed.\n")
    print("Best Parameters:")
    for key, value in best_params.items():
        print(f"{key}: {value}")