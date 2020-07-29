import tensorflow as tf
import pandas as pd
import numpy as np
from utils import get_features_labels
from lstm import LSTMDataGenerator
from betting_simulator import BettingSimulator

MODEL_NAME = "lstm_stats_minute_odds_start_odds"
BATCH_SIZE = 64


def classification_accuracy(model, case_study):
    df_test = pd.read_csv("datasets/test.csv")
    df_test = df_test.drop(columns=['event.start_time', 'team.home', 'team.away', 'league.name', 'league.id',
                                    'country.code', 'final_score'])

    # scale except event.id, minute and result
    scaler = np.loadtxt("models/scaler.csv").astype(np.float32)[1:]
    df_test.iloc[:, 2:-1] = np.multiply(df_test.iloc[:, 2:-1], scaler)

    if case_study == 1:
        test_generator = LSTMDataGenerator(df_test, batch_size=BATCH_SIZE)
    elif case_study == 2:
        test_generator = LSTMDataGenerator(df_test, batch_size=BATCH_SIZE, minute_odds=True)
    elif case_study == 3:
        test_generator = LSTMDataGenerator(df_test, batch_size=BATCH_SIZE, start_odds=True)
    else:
        test_generator = LSTMDataGenerator(df_test, batch_size=BATCH_SIZE, minute_odds=True, start_odds=True)

    loss, acc = model.evaluate(x=test_generator, verbose=1)
    model.summary()
    print("Evaluation loss: %f, accuracy: %f " % (loss, acc))


def betting_evaluation(model, case_study=4):
    df_test = pd.read_csv("datasets/test.csv")
    df_test = df_test.drop(columns=['event.start_time', 'team.home', 'team.away', 'league.name', 'league.id',
                                    'country.code', 'final_score'])

    simulator = BettingSimulator(model, df_test, is_lstm=True, kelly_criterion=True, sure_bet_threshold=0.25,
                                 odds_max=2.2)
    # simulator.simulate()
    # simulator.summary()


if __name__ == '__main__':
    model = tf.keras.models.load_model('models/' + MODEL_NAME)

    # classification_accuracy(model, case_study=0)
    betting_evaluation(model)
