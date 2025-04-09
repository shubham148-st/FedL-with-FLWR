from typing import List, Tuple
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Metrics
import toml
import pathlib
import csv
from graph import Forserver

config = toml.load("pyproject.toml")
rounds = config["server"]["global_epochs"]
clients = config["server"]["clients"]
round_accuracies = []
round_losses = []

LOG_FILE = "metrics_log.csv"

def log_metrics_to_csv(round_num, metrics: List[Tuple[int, Metrics]], avg_acc, avg_loss, filename=LOG_FILE):
    write_header = not pathlib.Path(filename).exists()
    with open(filename, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Round", "Client", "Accuracy", "Loss", "Round_Avg_Acc", "Round_Avg_Loss"])
        for idx, (num_examples, m) in enumerate(metrics):
            writer.writerow([round_num, f"Client_{idx+1}", m.get("accuracy", 0.0), m.get("loss", 0.0), avg_acc, avg_loss])

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    total_examples = sum(examples)
    if total_examples == 0:
        return {"accuracy": 0.0, "loss": 0.0}

    weighted_accuracy = sum(accuracies) / total_examples
    weighted_loss = sum(losses) / total_examples

    round_accuracies.append(weighted_accuracy)
    round_losses.append(weighted_loss)

    # Log metrics
    round_num = len(round_accuracies)
    log_metrics_to_csv(round_num, metrics, weighted_accuracy, weighted_loss)

    return {"accuracy": weighted_accuracy, "loss": weighted_loss}

# Define strategy
strategy = FedAvg(evaluate_metrics_aggregation_fn=weighted_average, min_available_clients=clients)

# Define config
config = ServerConfig(rounds)

# Flower ServerApp
app = ServerApp(config=config, strategy=strategy)

# Legacy mode
if __name__ == "__main__":
    from flwr.server import start_server
    start_server(server_address="0.0.0.0:5006", config=config, strategy=strategy)
    Forserver.plot_metrics(round_accuracies, round_losses)
