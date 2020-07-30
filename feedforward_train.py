import numpy as np
import tensorflow as tf

from feedforward import create_base_model
from utils import get_features_labels
from sklearn.preprocessing import MinMaxScaler
from plots import plot_training_history
from utils import get_train_val_test

EPOCHS = 20
BATCH_SIZE = 64
CASE_STUDY = 0

df_train, df_val, df_test = get_train_val_test()

if CASE_STUDY == 1:
    MODEL_NAME = "feedforward_stats"
    X_train, y_train = get_features_labels(df_train)
    X_val, y_val = get_features_labels(df_val)
elif CASE_STUDY == 2:
    MODEL_NAME = "feedforward_stats_minute_odds"
    X_train, y_train = get_features_labels(df_train, minute_odds=True)
    X_val, y_val = get_features_labels(df_val, minute_odds=True)
elif CASE_STUDY == 3:
    MODEL_NAME = "feedforward_stats_start_odds"
    X_train, y_train = get_features_labels(df_train, start_odds=True)
    X_val, y_val = get_features_labels(df_val, start_odds=True)
else:
    MODEL_NAME = "feedforward_stats_minute_odds_start_odds"
    X_train, y_train = get_features_labels(df_train, minute_odds=True, start_odds=True)
    X_val, y_val = get_features_labels(df_val, minute_odds=True, start_odds=True)

print("CASE STUDY: %d, MODEL_NAME: %s" % (CASE_STUDY, MODEL_NAME))

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

np.savetxt("models/" + MODEL_NAME + ".csv", scaler.scale_, delimiter=",")

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="models/weights/" + MODEL_NAME + ".h5",
    monitor='val_categorical_accuracy',
    mode='max',
    save_weights_only=True,
    save_best_only=True)

model = create_base_model(X_train.shape[1])
history = model.fit(x=X_train,
                    y=y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(X_val, y_val),
                    callbacks=[model_checkpoint_callback]
                    )

model.load_weights("models/weights/" + MODEL_NAME + ".h5")
model.save("models/" + MODEL_NAME)

plot_training_history(history, MODEL_NAME)
