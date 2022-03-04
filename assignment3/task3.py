import pathlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import torch.nn

import utils
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer
from trainer import compute_loss_and_accuracy


import numpy as np


class ExampleModel(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        # TODO: Implement this function (Task  2a)
        num_filters = 64  # Set number of filters in first conv layer
        self.num_classes = num_classes
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            # nn.Sigmoid(),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(64),

            nn.Conv2d(
                in_channels=num_filters,
                out_channels=128,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            # nn.Sigmoid(),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(128),

            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            # nn.Sigmoid(),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(256)
        )
        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
        self.num_output_features = 8*4*128
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_output_features, 64),
            nn.ReLU(),
            # nn.Softmax(),
            nn.BatchNorm1d(64),
            nn.Linear(64, num_classes)
        )

        # self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        # TODO: Implement this function (Task  2a)
        batch_size = x.shape[0]
        feat = self.feature_extractor(x)
        # feat = self.dropout(feat)
        out = self.classifier(feat)
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()


def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(0)
    epochs = 10
    batch_size = 64
    learning_rate = 1e-3
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    model = ExampleModel(image_channels=3, num_classes=10)
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    trainer.train()
    # create_plots(trainer, "task3")

    model_name = "big_filter"
    np.save(f"history/model_{model_name}_train.npy", trainer.train_history)
    np.save(f"history/model_{model_name}_val.npy", trainer.validation_history)
    np.save(f"history/model_{model_name}_test.npy", trainer.test_history)

    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    utils.plot_loss(trainer.test_history["loss"], label="Test loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    utils.plot_loss(trainer.test_history["accuracy"], label="Test Accuracy")
    plt.legend()
    # # these are matplotlib.patch.Patch properties
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # # place a text box in upper left in axes coords
    # plt.text(0.05, 0.95, f"Final accuracy:\nTrain: {trainer.train_history['accuracy']}%\nValidation: {trainer.validation_history['accuracy']}%\nTest: {trainer.test_history['accuracy']}%", fontsize=14,
    #         verticalalignment='top', bbox=props)

    # text_box = AnchoredText(f"Final accuracy:\nTrain: {next(reversed(trainer.train_history['accuracy'].values()))}%\nValidation: {next(reversed(trainer.validation_history['accuracy'].values()))}%\nTest: {next(reversed(trainer.test_history['accuracy'].values()))}%", frameon=True, loc=4, pad=0.5)
    # plt.setp(text_box.patch, facecolor='white', alpha=0.5)
    # plt.gca().add_artist(text_box)

    plt.savefig(plot_path.joinpath(f"task3_test_plot.png"))


    # train_history4d = np.load("model_task4d_train.npy", allow_pickle=True)[()]
    # val_history4d = np.load("model_task4d_val.npy", allow_pickle=True)[()]
    plt.show()

def calc_best_values():
    utils.set_seed(0)
    epochs = 10
    batch_size = 64
    learning_rate = 5e-2
    early_stop_count = 4
    dataloader_train, dataloader_val, dataloader_test = load_cifar10(batch_size)
    model = ExampleModel(image_channels=3, num_classes=10)
    model.load_state_dict(utils.load_best_checkpoint(pathlib.Path('checkpoints')))
    model.eval()

    train_loss, train_acc = compute_loss_and_accuracy(
        dataloader_train, model, torch.nn.CrossEntropyLoss()
    )

    validation_loss, validation_acc = compute_loss_and_accuracy(
        dataloader_val, model, torch.nn.CrossEntropyLoss()
    )

    test_loss, test_acc = compute_loss_and_accuracy(
        dataloader_test, model, torch.nn.CrossEntropyLoss()
    )

    print("Train_acc: " + str(train_acc))
    print("Val_acc: " + str(validation_acc))
    print("Test_acc: " + str(test_acc))

if __name__ == "__main__":
    main()
    #calc_best_values()


