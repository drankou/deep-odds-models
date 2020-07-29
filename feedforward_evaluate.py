import tensorflow as tf
import pandas as pd
import numpy as np
from utils import get_features_labels
from betting_simulator import BettingSimulator
from feedforward import create_base_model

MODEL_NAME = "feedforward_stats_minute_odds_start_odds"


def classification_accuracy(model, case_study):
    df_test = pd.read_csv("datasets/test.csv")
    df_test = df_test.drop(columns=['event.start_time', 'team.home', 'team.away', 'league.name', 'league.id',
                                    'country.code', 'final_score'])

    if case_study == 1:
        X_test, y_test = get_features_labels(df_test)
    elif case_study == 2:
        X_test, y_test = get_features_labels(df_test, minute_odds=True)
    elif case_study == 3:
        X_test, y_test = get_features_labels(df_test, start_odds=True)
    else:
        X_test, y_test = get_features_labels(df_test, minute_odds=True, start_odds=True)

    scaler = np.loadtxt("models/scaler.csv").astype(np.float32)[:X_test.shape[1]]
    X_test = np.multiply(X_test, scaler)

    loss, acc = model.evaluate(X_test, y_test, verbose=1)

    model.summary()
    print("Evaluation loss: %f, accuracy: %f " % (loss, acc))


def betting_evaluation(model, case_study=4):
    df_test = pd.read_csv("datasets/test.csv")
    df_test = df_test.drop(columns=['event.start_time', 'team.home', 'team.away', 'league.name', 'league.id',
                                    'country.code', 'final_score'])

    if case_study == 1:
        X_test, y_test = get_features_labels(df_test)
    elif case_study == 2:
        X_test, y_test = get_features_labels(df_test, minute_odds=True)
    elif case_study == 3:
        X_test, y_test = get_features_labels(df_test, start_odds=True)
    else:
        X_test, y_test = get_features_labels(df_test, minute_odds=True, start_odds=True)

    simulator = BettingSimulator(model, df_test, kelly_criterion=True, sure_bet_threshold=0.25, odds_max=2.2)
    # simulator.simulate()
    # simulator.summary()


if __name__ == '__main__':
    model = tf.keras.models.load_model('models/' + MODEL_NAME)

    classification_accuracy(model, case_study=0)
    betting_evaluation(model)
