import matplotlib.pyplot as plt

class Forserver():
    def plot_metrics(round_accuracies, round_losses):
        rounds = range(1, len(round_accuracies) + 1)

        plt.figure(figsize=(10, 4))

        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(rounds, round_accuracies, marker='o')
        plt.title("Accuracy over Rounds")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")

        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(rounds, round_losses, marker='o', color='red')
        plt.title("Loss over Rounds")
        plt.xlabel("Round")
        plt.ylabel("Loss")

        plt.suptitle("Federated Learning - Accuracy & Loss over Rounds", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # adjust for suptitle
        plt.savefig("flwr_training_metrics.png")
        plt.show()
