import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from utils import construct_full_match_sequence


# LSTM neural network model
def create_base_lstm_model(input_shape=(91, 24)):
    model = Sequential()
    model.add(LSTM(64, activation='tanh', recurrent_activation='sigmoid', input_shape=input_shape))
    model.add(Dropout(rate=0.5))
    model.add(Dense(3, activation='softmax'))

    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=[tf.keras.metrics.categorical_accuracy])

    return model


class LSTMDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size=64, shuffle=True, minute_odds=False, start_odds=False):
        'Initialization'
        self.df = df
        if shuffle:
            self.df = df.sample(frac=1)
        self.events = self.__prepare_events()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.minute_odds = minute_odds
        self.start_odds = start_odds

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate data

        # select batch of minutes from dataframe
        minutes_batch = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__reshape_data(minutes_batch)

        return X, y

    def __prepare_events(self):
        events = {}

        for _, event in self.df.groupby('event.id'):
            event_id = event['event.id'].iloc[0]
            events[event_id] = event.sort_values(by='minute')

        return events

    # reshape training data into a 3D array: X = (n_events*n_minutes_per_event, n_minutes, n_features)
    def __reshape_data(self, minutes_batch):
        X = []
        y = []

        # iterate over minutes
        for _, row in minutes_batch.iterrows():
            event_id = row['event.id']
            minute = row['minute']

            # find event in dictionary and construct full (91, n_features) sequence up to given minute
            event = self.events[event_id]
            full_match_sequence = construct_full_match_sequence(event, minute,
                                                                minute_odds=self.minute_odds,
                                                                start_odds=self.start_odds)

            X.append(full_match_sequence)
            y.append(tf.keras.utils.to_categorical(row['result'], num_classes=3))

        return np.array(X), np.array(y)
