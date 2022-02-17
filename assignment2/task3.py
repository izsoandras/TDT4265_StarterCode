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

    # Example created for comparing with and without shuffling.
    # For comparison, show all loss/accuracy curves in the same plot
    # YOU CAN DELETE EVERYTHING BELOW!

    # TASK 2 MODEL

    case = "Task 2 Model"

    model2 = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer2 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model2, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    #train_history2, val_history2 = trainer2.train(num_epochs)
    #np.save("model_task2_train.npy", train_history2)
    #np.save("model_task2_val.npy", val_history2)

    # np.save("model_task2_train.npy", train_history2)
    # np.save("model_task2_val.npy", val_history2)
    train_history2 = np.load("model_task2_train.npy", allow_pickle=True)[()]
    val_history2 = np.load("model_task2_val.npy", allow_pickle=True)[()]

    #MODEL FOR TASK 3A - IMPORVED INIT
    case = "Task 3a Model - Use improved initialisation"
    use_improved_weight_init = True

    model3a = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer3a = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model3a, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    #train_history3a, val_history3a = trainer3a.train(num_epochs)
    #np.save("model_task3a_train.npy", train_history3a)
    #np.save("model_task3a_val.npy", val_history3a)

    train_history3a = np.load("model_task3a_train.npy", allow_pickle=True)
    val_history3a = np.load("model_task3a_val.npy", allow_pickle=True)


    #MODEL FOR TASK 3B - Improved Sigmoid
    case = "Task 3b Model - Use improved sigmoid"
    use_improved_sigmoid = True

    model3b = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer3b = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model3b, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history3, val_history3 = trainer3.train(num_epochs)

    np.save("model_task3_impr_sig_train", train_history3)
    np.save("model_task3_impr_sig_val", val_history3)
    # train_history3 = np.load("model_task3_impr_init_train.npy", allow_pickle=True)[()]
    # val_history3 = np.load("model_task3_impr_init_val.npy", allow_pickle=True)[()]

    # MDOEL FOR TASK 3C - MOMENTUM
    case = "Task 3 - Use momentum"
    use_momentum = True
    learning_rate = 0.02 #according to task

    model3c = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer3c = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model3c, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    #train_history3c, val_history3c = trainer3c.train(num_epochs)
    #np.save("model_task3c_momentum_train", train_history3c)
    #np.save("model_task3c_momentum_val", val_history3c)
    
    train_history3c = np.load("model_task3c_momentum_train.npy", allow_pickle=True)
    val_history3c = np.load("model_task3c_momentum_val.npy", allow_pickle=True)

    np.save("model_task3c_momentum_train", train_history3c)
    np.save("model_task3c_momentum_val", val_history3c)
    # train_history3c = np.load("model_task3c_momentum_train.npy", allow_pickle=True)[()]
    # val_history3c = np.load("model_task3c_momentum_val.npy", allow_pickle=True)[()]


    # plt.subplot(1, 2, 1)
    # utils.plot_loss(train_history2[()]["loss"],
    #                 "Task 2 Model", npoints_to_average=10)
    # utils.plot_loss(
    #     train_history3["loss"], case, npoints_to_average=10)
    # plt.ylim([0, .4])
    # plt.subplot(1, 2, 2)
    # plt.ylim([0.87, 1])
    # utils.plot_loss(val_history2[()]["accuracy"], "Task 2 Model")
    # utils.plot_loss(val_history3["accuracy"], case)
    # plt.ylabel("Validation Accuracy")
    # plt.legend()
    # plt.show()

    plt.figure()
    plt.subplot(1, 2, 1)
    utils.plot_loss(train_history3["loss"],
                    "Improved sigmoid", npoints_to_average=10)
    utils.plot_loss(
        train_history3c[()]["loss"], "Momentum", npoints_to_average=10)
    plt.ylim([0, .4])
    plt.subplot(1, 2, 2)
    plt.ylim([0.87, 1])
    utils.plot_loss(val_history3["accuracy"], "Improved sigmoid")
    utils.plot_loss(val_history3c["accuracy"], "Momentum")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.savefig('task3c_momentum.png')
    plt.show()

    plt.figure()
    plt.subplot(1, 2, 1)
    utils.plot_loss(train_history2[()]["loss"],
                    "Task 2", npoints_to_average=10)
    utils.plot_loss(
        train_history3c[()]["loss"], "Momentum", npoints_to_average=10)
    plt.ylim([0, .4])
    plt.subplot(1, 2, 2)
    plt.ylim([0.87, 1])
    utils.plot_loss(val_history2[()]["accuracy"], "Task 2")
    utils.plot_loss(val_history3c[()]["accuracy"], "Momentum")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.show()
