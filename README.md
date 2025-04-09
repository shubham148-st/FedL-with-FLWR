🧠 FedL-with-FLWR
🚀 Overview
This project implements a Federated Learning (FL) system for image-based plant disease detection using PyTorch, CNN models, and the Flower (FLWR) framework.
The model is trained across multiple clients without sharing raw data, ensuring data privacy.

⚠️ Note: This implementation uses Flower v0.1.0, an early version of the Flower framework. It supports only 4–5 clients in a broadcasted network setup. Due to its limitations, it's best suited for small-scale university-level AIML projects.

🚧 If you're familiar with the latest Flower versions (e.g., 1.x), consider upgrading the code. Contributions are welcome and would benefit the community!

📦 Dependencies
The following Python libraries are required:

flwr==1.17.0
flwr-datasets==0.5.0
torch==2.5.1
torchvision==0.20.1
scikit-learn==1.6.1
pandas==2.2.3
numpy==2.2.4
matplotlib==3.10.1
seaborn==0.13.2
ray==2.31.0
protobuf==4.25.6
tqdm==4.67.1
pyarrow==19.0.1
grpcio==1.71.0
requests==2.32.3
📅 To install all dependencies:

pip install -r requirements.txt
📁 Data Setup
Organize your image dataset as follows:

Client_1/
  ├── train/
  └── test/
Client_2/
  ├── train/
  └── test/
The dataset should consist of images labeled into two classes:

0 → Healthy
1 → Diseased
(You can modify this in modelCNN.py if needed.)
💪 Using Pre-trained Models
Pre-trained models (trained_model_client1.pt, trained_model_client2.pt) are provided for convenience.
Dataset used: ~20K images of tomato, potato, and bell pepper leaves. (Download link coming soon)
To use without retraining:
Move all files from the test/ folder to the root directory of this project.
Place images for validation inside the valid/ folder.
Run:
python predictCL1.py
# or
python predictCL2.py
⚙️ Running from Scratch
Install dependencies (recommended in a virtual environment).
In three separate terminals, run the following:
# Terminal 1
python server.py

# Terminal 2
python client_1.py

# Terminal 3
python client_2.py
The system will:
Start training using FedAvg
Display accuracy/loss graphs
Generate logs in CSV format
Save trained models for each client
📝 Notes
Designed for Federated Learning experiments using Flower.
Ensure paths in scripts are updated if directory structures are changed.
Recommended to use virtual environments for clean package management.
