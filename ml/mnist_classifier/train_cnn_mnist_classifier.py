import wandb
from cnn_mnist_classifier import train

class Config:
    def __init__(self):
        self.batch_size = 16
        self.learning_rate = 0.001
        self.random_seed = 0
        self.num_epochs = 30

# wandb config
project_name = "playground-CNN_MNIST_Classifier"

wandb.login()
hyperparameter_configuration = Config()



def main():
    wandb.init(project=project_name)
    
    test_acc = train(hyperparameter_configuration)
    wandb.log({"test_accuracy": test_acc})

if __name__ == "__main__":
    main()
