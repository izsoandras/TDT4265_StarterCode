import utils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

def run():
    model_name = "pre_batchnorm"
    pre_train_hist = np.load(f"history/model_{model_name}_train.npy", allow_pickle=True)[()]
    pre_val_hist = np.load(f"history/model_{model_name}_val.npy", allow_pickle=True)[()]
    pre_test_hist = np.load(f"history/model_{model_name}_test.npy", allow_pickle=True)[()]
    model_name = "post_batchnorm"
    post_train_hist = np.load(f"history/model_{model_name}_train.npy", allow_pickle=True)[()]
    post_val_hist = np.load(f"history/model_{model_name}_val.npy", allow_pickle=True)[()]
    post_test_hist = np.load(f"history/model_{model_name}_test.npy", allow_pickle=True)[()]

    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(pre_train_hist["loss"], label="Training loss before BN", npoints_to_average=10)
    utils.plot_loss(pre_val_hist["loss"], label="Validation loss before BN")
    utils.plot_loss(post_train_hist["loss"], label="Training loss after BN", npoints_to_average=10)
    utils.plot_loss(post_val_hist["loss"], label="Validation loss after BN")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(pre_val_hist["accuracy"], label="Validation Accuracy before BN")
    utils.plot_loss(pre_test_hist["accuracy"], label="Test Accuracy before BN")
    utils.plot_loss(post_val_hist["accuracy"], label="Validation Accuracy after BN")
    utils.plot_loss(post_test_hist["accuracy"], label="Test Accuracy after BN")
    plt.legend()
    # # these are matplotlib.patch.Patch properties
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # # place a text box in upper left in axes coords
    # plt.text(0.05, 0.95, f"Final accuracy:\nTrain: {trainer.train_history['accuracy']}%\nValidation: {trainer.validation_history['accuracy']}%\nTest: {trainer.test_history['accuracy']}%", fontsize=14,
    #         verticalalignment='top', bbox=props)

    # text_box = AnchoredText(f"Final accuracy:\nTrain: {next(reversed(train_hist['accuracy'].values()))}%\nValidation: {next(reversed(val_hist['accuracy'].values()))}%", frameon=True, loc=4, pad=0.5)
    # plt.setp(text_box.patch, facecolor='white', alpha=0.5)
    # plt.gca().add_artist(text_box)
    plt.show()
    # plt.savefig(plot_path.joinpath(f"task3_test_plot.png"))

if __name__ == "__main__":
    run()
