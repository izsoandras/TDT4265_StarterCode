import numpy as np
import matplotlib.pyplot as plt


def run():
    np.random.seed(19680801)

    fig, ax = plt.subplots()
    x = 30*np.random.randn(10000)

    ax.hist(x, 50)
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(0.05, 0.95, f"Final accuracy:\nTrain: 87.8%\nValidation: 78.5%\nTest: 75.6%", transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    plt.show()

if __name__ == "__main__":
    run()