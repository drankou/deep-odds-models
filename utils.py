import tensorflow as tf
import pandas as pd
import numpy as np


def get_train_val_test(train_split=0.2):
    df = pd.read_csv("datasets/train.csv")
    df_test = pd.read_csv("datasets/test.csv")

    df = df.drop(columns=['event.start_time', 'team.home', 'team.away', 'league.name', 'league.id', 'country.code',
                          'final_score'])
    df_test = df_test.drop(columns=['event.start_time', 'team.home', 'team.away', 'league.name', 'league.id',
                                    'country.code', 'final_score'])

    total = len(df) / 91
    split = int(total * (1 - train_split)) * 91

    df_train = df.iloc[:split, :]
    df_val = df.iloc[split:, :]

    return df_train, df_val, df_test


def get_features_labels(df, minute_odds=False, start_odds=False):
    # shuffle dataset
    df = df.sample(frac=1)

    # match-related features
    X = df.iloc[:, 1:-7]
    y = tf.keras.utils.to_categorical(df.iloc[:, -1])

    if minute_odds:
        X = pd.concat([X, df.iloc[:, -7:-4]], axis=1)
    if start_odds:
        X = pd.concat([X, df.iloc[:, -4:-1]], axis=1)

    return X, y


def get_features_for_minute(data, minute_odds=False, start_odds=False):
    X = data[1:-7]

    if minute_odds:
        X = X.append(data[-7:-4])
    if start_odds:
        X = X.append(data[-4:-1])

    return X


def construct_full_match_sequence(event, minute, minute_odds=False, start_odds=False):
    to_minute_sequence = event.iloc[:int(minute) + 1, 2:-7]
    if minute_odds:
        to_minute_sequence = pd.concat([to_minute_sequence, event.iloc[:int(minute) + 1, -7:-4]], axis=1)
    if start_odds:
        to_minute_sequence = pd.concat([to_minute_sequence, event.iloc[:int(minute) + 1, -4:-1]], axis=1)

    full_match_sequence = pad_minutes(to_minute_sequence, shape=(91, to_minute_sequence.shape[1]))

    return full_match_sequence


def pad_minutes(minutes, shape=(91, 24)):
    result = np.zeros(shape)
    result[:minutes.shape[0], :minutes.shape[1]] = minutes

    return result
