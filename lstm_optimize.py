import tensorflow as tf
import numpy as np

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from tensorflow.keras.layers import Dense, Activation, LSTM, Dropout
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from utils import get_train_val_test, get_features_labels
from lstm import LSTMDataGenerator


def data():
    df_train, df_val, df_test = get_train_val_test()
    scaler = MinMaxScaler()

    df_train = df_train.iloc[:int((len(df_train) / 91) * 0.5) * 91, :]
    df_val = df_val.iloc[:int((len(df_val) / 91) * 0.5) * 91, :]
    df_test = df_val.iloc[:int((len(df_test) / 91) * 0.5) * 91, :]

    df_train.iloc[:, 2:-1] = scaler.fit_transform(df_train.iloc[:, 2:-1])
    df_val.iloc[:, 2:-1] = scaler.transform(df_val.iloc[:, 2:-1])
    df_test.iloc[:, 2:-1] = scaler.transform(df_test.iloc[:, 2:-1])

    training_generator = LSTMDataGenerator(df_train, batch_size=64, minute_odds=True, start_odds=True)
    validation_generator = LSTMDataGenerator(df_val, batch_size=64, minute_odds=True, start_odds=True)
    # testing_generator = LSTMDataGenerator(df_test, batch_size=64, minute_odds=True, start_odds=True)

    # np.savetxt("models/lstm_scaler.csv", scaler.scale_, delimiter=",")

    return training_generator, validation_generator


def model(train, val):
    model = Sequential()
    model.add(
        LSTM({{choice([16, 32, 64, 128])}}, activation='tanh', recurrent_activation='sigmoid', input_shape=(91, 24),
             return_sequences=True))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(
        LSTM({{choice([16, 32, 64, 128])}}, activation='tanh', recurrent_activation='sigmoid', input_shape=(91, 24)))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    adam = tf.keras.optimizers.Adam(lr={{choice([10 ** -3, 10 ** -2, 10 ** -1])}})
    rmsprop = tf.keras.optimizers.RMSprop(lr={{choice([10 ** -3, 10 ** -2, 10 ** -1])}})
    sgd = tf.keras.optimizers.SGD(lr={{choice([10 ** -3, 10 ** -2, 10 ** -1])}})

    choiceval = {{choice(['adam', 'sgd', 'rmsprop'])}}
    if choiceval == 'adam':
        optim = adam
    elif choiceval == 'rmsprop':
        optim = rmsprop
    else:
        optim = sgd

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optim)

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)
    model.fit(x=train,
              batch_size={{choice([32, 64, 128])}},
              epochs=2,
              verbose=2,
              callbacks=[callback]
              )
    score, acc = model.evaluate(x=test, verbose=0)
    print("Model architecture:", model.summary())
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=30,
                                          trials=Trials())

    print(best_run)
    print(best_model)
