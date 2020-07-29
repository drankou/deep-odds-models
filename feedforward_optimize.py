import tensorflow as tf
import numpy as np

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from utils import get_train_val_test, get_features_labels

EPOCHS = 10


def data():
    df_train, df_val, _ = get_train_val_test()
    X_train, y_train = get_features_labels(df_train, minute_odds=True, start_odds=True)
    X_val, y_val = get_features_labels(df_val, minute_odds=True, start_odds=True)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    np.savetxt("./models/scaler.csv", scaler.scale_, delimiter=",")

    return X_train, y_train, X_val, y_val


def model(X_train, y_train, X_val, y_val):
    model = Sequential()
    model.add(Dense({{choice([8, 16, 32, 64])}}, input_shape=(25,)))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.add(Dense({{choice([8, 16, 32, 64])}}))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
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

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)
    model.fit(X_train, y_train,
              batch_size={{choice([32, 64, 128])}},
              epochs=5,
              verbose=2,
              validation_data=(X_val, y_val),
              callbacks=[callback]
              )
    score, acc = model.evaluate(X_val, y_val, verbose=0)
    print("Model architecture:", model.summary())
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=30,
                                          trials=Trials())
