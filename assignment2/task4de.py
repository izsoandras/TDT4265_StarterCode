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

    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_momentum = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # Task 4a   -   Too few hidden units
    neurons_per_layer = [60, 60, 10]

    model4d = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer4d = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model4d, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history4d, val_history4d = trainer4d.train(num_epochs)

    np.save("model_task4d_train.npy", train_history4d)
    np.save("model_task4d_val.npy", val_history4d)
    # train_history2 = np.load("model_task4a_train.npy", allow_pickle=True)
    # val_history2 = np.load("model_task4a_val.npy", allow_pickle=True)

    # Example created for comparing with and without shuffling.
    # For comparison, show all loss/accuracy curves in the same plot
    # YOU CAN DELETE EVERYTHING BELOW!

    # Task 4b   -   Too many hidden units
    neurons_per_layer = [64,]*10+[10,]

    model4e = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer4e = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model4e, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history4e, val_history4e = trainer4e.train(num_epochs)

    np.save("model_task4e_train.npy", train_history4e)
    np.save("model_task4e_val.npy", val_history4e)
    # train_history4e = np.load("model_task4e_train.npy", allow_pickle=True)
    # val_history4e = np.load("model_task4e_val.npy", allow_pickle=True)

    plt.figure()
    plt.subplot(1, 2, 1)
    utils.plot_loss(train_history4d["loss"],
                    "2 hidden layer", npoints_to_average=10)
    utils.plot_loss(
        train_history4e["loss"], "hidden num: 128", npoints_to_average=10)
    plt.ylim([0, .4])
    plt.subplot(1, 2, 2)
    plt.ylim([0.87, 1])
    utils.plot_loss(val_history4d["accuracy"], "2 hidden layer")
    utils.plot_loss(val_history4e["accuracy"], "10 hidden layer")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.show()
    plt.savefig('task4de.png')
