import tensorflow as tf
import pandas as pd
import numpy as np
from utils import get_features_labels
from betting_simulator import BettingSimulator

MODEL_NAME = "feedforward_stats_minute_odds_start_odds"


def classification_accuracy(model, minute_odds=False, start_odds=False):
    df_test = pd.read_csv("datasets/test.csv")
    df_test = df_test.drop(columns=['event.start_time', 'team.home', 'team.away', 'league.name', 'league.id',
                                    'country.code', 'final_score'])

    X_test, y_test = get_features_labels(df_test, minute_odds, start_odds)

    scaler = np.loadtxt("models/scaler.csv").astype(np.float32)[:X_test.shape[1]]
    X_test = np.multiply(X_test, scaler)

    loss, acc = model.evaluate(X_test, y_test, verbose=1)

    model.summary()
    print("Evaluation loss: %f, accuracy: %f " % (loss, acc))


def betting_evaluation(model):
    df_test = pd.read_csv("datasets/test_betting.csv")
    df_test = df_test.drop(columns=['event.start_time', 'team.home', 'team.away', 'league.name', 'league.id',
                                    'country.code', 'final_score'])

    print("Number of matches for betting evaluation: ", len(df_test) / 91)

    simulator = BettingSimulator(model, df_test, minute_odds=True, start_odds=True, odds_min=1.7, odds_max=2.1,
                                 kelly_criterion=True, draw_bets=True)
    simulator.simulate()
    simulator.summary()


if __name__ == '__main__':
    model = tf.keras.models.load_model('models/' + MODEL_NAME)

    classification_accuracy(model, minute_odds=True, start_odds=True)
    betting_evaluation(model)
