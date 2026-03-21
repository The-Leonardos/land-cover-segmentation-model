import torch
from landcover.training import HyperparameterTuning
from landcover import DATA_PATH
import datetime
import wandb
from dotenv import load_dotenv
import os

# ENCODERS TO TEST
        # 1. resnet50 (DONE)
        # 3. resnet34 (DONE)
        # 4. resnet101 (DONE)
        # 5. efficientnet_b0 (DONE)
        # 6. efficientnet_b3 (TO BE TESTED)
        # 7. densenet-169
        # 8. densenet-201

if __name__ == "__main__":
    load_dotenv("../config.env")
    KEY = os.getenv("WANDB_API_KEY")

    # auth login to weights and biases
    wandb.login(key=KEY)

    # check device availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device]: {device.upper()}")
    torch.backends.cudnn.benchmark = True

    # hyperparameter tuning instance
    tuning = HyperparameterTuning(
        n_trials=60,
        epochs=25, 
        encoder="efficientnet_b0",
        version="v0",
        device=device
    )

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