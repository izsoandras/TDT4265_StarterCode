import numpy as np

import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # Task 4a   -   Too few hidden units
    neurons_per_layer = [32, 10]

    model4a = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer4a = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model4a, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history4a, val_history4a = trainer4a.train(num_epochs)

    np.save("model_task4a_train.npy", train_history4a)
    np.save("model_task4a_val.npy", val_history4a)
    # train_history4a = np.load("model_task4a_train.npy", allow_pickle=True)[()]
    # val_history4a = np.load("model_task4a_val.npy", allow_pickle=True)[()]

    # Example created for comparing with and without shuffling.
    # For comparison, show all loss/accuracy curves in the same plot
    # YOU CAN DELETE EVERYTHING BELOW!

    neurons_per_layer = [128, 10]

    model4b = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer4b = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model4b, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history4b, val_history4b = trainer4b.train(num_epochs)

    np.save("model_task4b_train.npy", train_history4b)
    np.save("model_task4b_val.npy", val_history4b)
    # train_history4b = np.load("model_task4b_train.npy", allow_pickle=True)[()]
    # val_history4b = np.load("model_task4b_val.npy", allow_pickle=True)[()]

    plt.figure()
    plt.subplot(1, 2, 1)
    utils.plot_loss(train_history4a["loss"],
                    "hidden num: 32", npoints_to_average=10)
    utils.plot_loss(
        train_history4b["loss"], "hidden num: 128", npoints_to_average=10)
    plt.ylim([0, .4])
    plt.subplot(1, 2, 2)
    plt.ylim([0.87, 1])
    utils.plot_loss(val_history4a["accuracy"], "hidden num: 32")
    utils.plot_loss(val_history4b["accuracy"], "hidden num: 128")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.savefig('task4ab.png')
    plt.show()
