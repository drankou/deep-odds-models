import matplotlib.pyplot as plt
import numpy as np
from utils import get_features_labels
from lstm import LSTMDataGenerator


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


# minute-by-minute model evaluation
def acc_minute_by_minute(df, model, model_name, is_lstm=False):
    result = {}

    scaler = np.loadtxt("models/scaler.csv").astype(np.float32)
    for i in range(0, 91):
        df_minute = df[df["minute"] == i]

        if is_lstm:
            df.iloc[:, 2:-1] = np.multiply(df.iloc[:, 2:-1], scaler[1:])
            test_generator = LSTMDataGenerator(df, batch_size=64, minute_odds=True, start_odds=True)
            _, acc = model.evaluate(x=test_generator, verbose=0)
        else:
            X, y = get_features_labels(df_minute)
            X = np.multiply(X, scaler)
            _, acc = model.evaluate(X, y, verbose=0)

        result[i] = acc

    # plot minute-by-minute model accuracy
    x, y = zip(*sorted(result.items()))
    plt.figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(x, y)
    plt.xticks(np.arange(min(x), max(x) + 5, 5.0))
    plt.show()
    plt.savefig("/images/minute-by-minute" + model_name + ".png")

    return result
