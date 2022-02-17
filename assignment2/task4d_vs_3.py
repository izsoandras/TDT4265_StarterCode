import numpy as np
from matplotlib import pyplot as plt
import utils


if __name__ == "__main__":
    train_history3c = np.load("model_task3c_momentum_train.npy", allow_pickle=True)[()]
    val_history3c = np.load("model_task3c_momentum_val.npy", allow_pickle=True)[()]

    train_history4d = np.load("model_task4d_train.npy", allow_pickle=True)[()]
    val_history4d = np.load("model_task4d_val.npy", allow_pickle=True)[()]

    plt.figure()
    plt.subplot(1, 2, 1)
    utils.plot_loss(train_history3c["loss"],
                    "Task 3c)", npoints_to_average=10)
    utils.plot_loss(
        train_history4d["loss"], "Task 4d)", npoints_to_average=10)
    # plt.ylim([0, .4])
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    # plt.ylim([0.87, 1])
    utils.plot_loss(val_history3c["accuracy"], "Task 3c)")
    utils.plot_loss(val_history4d["accuracy"], "Task 4d)")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.savefig('task4d_vs_3c.png')
    plt.show()
