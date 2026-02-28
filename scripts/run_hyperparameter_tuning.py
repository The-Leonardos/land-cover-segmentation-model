import torch
from landcover.training import HyperparameterTuning
from landcover import DATA_PATH

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('device:', device)

    tuning = HyperparameterTuning(10, 30, device)

    df, _ = tuning.run()

    csv_path = DATA_PATH / 'hyperparameter_tuning' / 'deeplabv3plus_tuning_results.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(csv_path, index=False)