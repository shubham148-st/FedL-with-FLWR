from flwr.client import NumPyClient, ClientApp
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
import toml
config = toml.load("pyproject.toml")
epochs = config["training"]["local_epochs"]


#data loading
def load_data(train_path='data/Client1/train', test_path='data/Client1/test'):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=transform)
    test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader

from modelCNN import CNN, traintest




# Flower client
class FlowerClient(NumPyClient):
    def __init__(self, net, train_loader, test_loader):
        self.net = net
        self.train_loader = train_loader
        self.test_loader = test_loader

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        traintest.train(self.net, self.train_loader, epochs)
        # Save model
        torch.save(self.net.state_dict(), "trained_model_client_1.pt") 
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = traintest.test(self.net, self.test_loader)
        print(f"Client Evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
        return loss, len(self.test_loader.dataset), {"accuracy": accuracy / 100, "loss":loss}  # Convert to [0,1] range



net = CNN() # model instance
def client_fn(cid: str):
    """Create and return an instance of Flower `Client`."""
    train_loader, test_loader = load_data()
    return FlowerClient(net, train_loader, test_loader).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)


if __name__ == "__main__":
    from flwr.client import start_client

    train_loader, test_loader = load_data()
    start_client(
        server_address="127.0.0.1:5006",
        client=FlowerClient(net, train_loader, test_loader).to_client(),
    )
