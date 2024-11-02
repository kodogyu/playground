import wandb
from cnn_mnist_classifier import train

# wandb config
project_name = "playground-CNN_MNIST_Classifier"

wandb.login()
sweep_configuration = {
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "test_accuracy"},
    "parameters": {
        "num_epochs": {
            "value": 30
        },
        "batch_size": {
            "values": [16, 32, 64, 128, 256]
        },
        "learning_rate": {
            "distribution": "uniform",
            "min": 0.0001,
            "max": 0.001
        },
        "random_seed": {
            "value": 0
        }
    }
}
sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)


def main():
    wandb.init(project=project_name)
    
    test_acc = train(wandb.config)
    wandb.log({"test_accuracy": test_acc})

if __name__ == "__main__":
    wandb.agent(sweep_id, main, count=64)
