import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images
from trainer import BaseTrainer
from task3a import cross_entropy_loss, SoftmaxModel, one_hot_encode
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    # TODO: Implement this function (task 3c)
    inference = model.forward(X)
    corr_predictions = 0
    for i in range(X.shape[0]):
        inference[i, :] = (inference[i, :] == np.max(inference[i, :]))*1
        corr_predictions += sum(inference[i, :] == targets[i, :]) == len(inference[i, :])
    #print(corr_predictions)
    accuracy = corr_predictions / targets.shape[0]
    return accuracy


class SoftmaxTrainer(BaseTrainer):

    def train_step(self, X_batch: np.ndarray, Y_batch: np.ndarray):
        """
        Perform forward, backward and gradient descent step here.
        The function is called once for every batch (see trainer.py) to perform the train step.
        The function returns the mean loss value which is then automatically logged in our variable self.train_history.

        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """
        # TODO: Implement this function (task 3b)
        outputs = self.model.forward(X_batch)
        self.model.backward(X_batch, outputs, Y_batch)
        self.model.w -= learning_rate * self.model.grad
        loss = cross_entropy_loss(Y_batch, outputs)

        return loss

    def validation_step(self):
        """
        Perform a validation step to evaluate the model at the current step for the validation set.
        Also calculates the current accuracy of the model on the train set.
        Returns:
            loss (float): cross entropy loss over the whole dataset
            accuracy_ (float): accuracy over the whole dataset
        Returns:
            loss value (float) on batch
            accuracy_train (float): Accuracy on train dataset
            accuracy_val (float): Accuracy on the validation dataset
        """
        # NO NEED TO CHANGE THIS FUNCTION
        logits = self.model.forward(self.X_val)
        loss = cross_entropy_loss(Y_val, logits)

        accuracy_train = calculate_accuracy(
            X_train, Y_train, self.model)
        accuracy_val = calculate_accuracy(
            X_val, Y_val, self.model)
        return loss, accuracy_train, accuracy_val


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = 0.01
    batch_size = 128
    l2_reg_lambda = 0
    shuffle_dataset = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # ANY PARTS OF THE CODE BELOW THIS CAN BE CHANGED.

    # Intialize model
    model = SoftmaxModel(l2_reg_lambda)
    # Train model
    trainer = SoftmaxTrainer(
        model, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    print("Final Validation accuracy:", calculate_accuracy(X_val, Y_val, model))

    plt.ylim([0.2, 0.6])
    utils.plot_loss(train_history["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.savefig("task3b_softmax_train_loss.png")
    plt.show()

    # Plot accuracy
    plt.ylim([0.89, .95]) #formerly 0.93 upper boundry
    utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task3b_softmax_train_accuracy.png")
    plt.show()

    # Train a model with L2 regularization (task 4b)

    model_reg2 = SoftmaxModel(l2_reg_lambda=2.0)
    trainer = SoftmaxTrainer(
        model_reg2, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_reg2, val_history_reg2 = trainer.train(num_epochs)

    model_reg0 = SoftmaxModel(l2_reg_lambda=0.0)
    trainer = SoftmaxTrainer(
        model_reg0, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_reg0, val_history_reg0 = trainer.train(num_epochs)

    plt.figure()
    ws = [model_reg0.w, model_reg2.w]
    w_im = np.zeros((28*2,28*10))
    for i in range(0,2):
        for k in range(0,10):
            plt.subplot(2,10, i*10+k+1)
            w_im[i*28:(i+1)*28,k*28:(k+1)*28] = np.reshape(ws[i][0:-1, k], (28, 28))
            # plt.imshow(im)

    plt.imsave("task4b_softmax_weight.png", w_im, cmap="gray")

    # Task 4c-e
    l2_lambdas = [.002, .02, .2, 2]
    models = []
    trainers = []
    train_histories = []
    val_histories = []
    for lbd in l2_lambdas:
        model = SoftmaxModel(l2_reg_lambda=lbd)
        trainer = SoftmaxTrainer(
            model, learning_rate, batch_size, shuffle_dataset,
            X_train, Y_train, X_val, Y_val,
        )
        train_history, val_history = trainer.train(num_epochs)

        models.append(model)
        trainers.append(trainer)
        train_histories.append(train_history)
        val_histories.append(val_history)

    # Plotting of accuracy for difference values of lambdas (task 4c)

    # Plot accuracy
    plt.figure()
    plt.ylim([0.75, .93]) #formerly 0.93 upper boundry
    for lbd, vh in zip(l2_lambdas, val_histories):
        utils.plot_loss(vh["accuracy"], r"$\lambda=$"+str(lbd))

    plt.xlabel("Number of Training Steps")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.savefig("task4c_l2_reg_accuracy.png")
    plt.show()


    # Task 4d - Plotting of the l2 norm for each weight
    plt.figure()
    l2_norms = [np.linalg.norm(m.w)**2 for m in models]
    plt.bar(range(0,len(l2_lambdas)), l2_norms,tick_label=l2_lambdas)
    plt.xlabel("Lambda values")
    plt.ylabel("$L_2 norm of the weights$")
    plt.savefig("task4e_l2_reg_norms.png")
    plt.show()
