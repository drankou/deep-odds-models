import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(history, model_name, loss=True, accuracy=True):
    if loss:
        # loss during training
        plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.xticks(np.arange(min(history.epoch), max(history.epoch) + 1, 5.0))

        plt.legend()
        plt.savefig('images/' + model_name + '_loss.png')
    if accuracy:
        # accuracy during training
        plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(history.history["categorical_accuracy"], label="Training categorical accuracy")
        plt.plot(history.history["val_categorical_accuracy"], label="Validation categorical accuracy")
        plt.xticks(np.arange(min(history.epoch), max(history.epoch) + 1, 5.0))

        plt.legend()
        plt.savefig('images/' + model_name + '_acc.png')
