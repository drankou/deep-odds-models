import numpy as np
import tensorflow as tf
from lstm import LSTMDataGenerator, create_base_lstm_model
from plots import plot_training_history
from sklearn.preprocessing import MinMaxScaler
from utils import get_train_val_test

BATCH_SIZE = 128
EPOCHS = 5
CASE_STUDY = 0

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
df_train, df_val, df_test = get_train_val_test()

# scale features except minute
scaler = MinMaxScaler()

df_train.iloc[:, 2:-1] = scaler.fit_transform(df_train.iloc[:, 2:-1])
df_val.iloc[:, 2:-1] = scaler.transform(df_val.iloc[:, 2:-1])

np.savetxt("models/lstm_scaler.csv", scaler.scale_, delimiter=",")

if CASE_STUDY == 1:
    MODEL_NAME = "lstm_stats"
    INPUT_DIM = (91, 18)
    training_generator = LSTMDataGenerator(df_train, batch_size=BATCH_SIZE)
    validation_generator = LSTMDataGenerator(df_val, batch_size=BATCH_SIZE)
elif CASE_STUDY == 2:
    MODEL_NAME = "lstm_stats_minute_odds"
    INPUT_DIM = (91, 21)
    training_generator = LSTMDataGenerator(df_train, batch_size=BATCH_SIZE, minute_odds=True)
    validation_generator = LSTMDataGenerator(df_val, batch_size=BATCH_SIZE, minute_odds=True)
elif CASE_STUDY == 3:
    MODEL_NAME = "lstm_stats_start_odds"
    INPUT_DIM = (91, 21)
    training_generator = LSTMDataGenerator(df_train, batch_size=BATCH_SIZE, start_odds=True)
    validation_generator = LSTMDataGenerator(df_val, batch_size=BATCH_SIZE, start_odds=True)
else:
    MODEL_NAME = "lstm_stats_minute_odds_start_odds"
    INPUT_DIM = (91, 24)
    training_generator = LSTMDataGenerator(df_train, batch_size=BATCH_SIZE, minute_odds=True, start_odds=True)
    validation_generator = LSTMDataGenerator(df_val, batch_size=BATCH_SIZE, minute_odds=True, start_odds=True)

print("CASE STUDY: %d, MODEL_NAME: %s" % (CASE_STUDY, MODEL_NAME))

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="models/weights/" + MODEL_NAME + ".h5",
    monitor='val_categorical_accuracy',
    mode='max',
    save_weights_only=True,
    save_best_only=True
)

model = create_base_lstm_model(INPUT_DIM)
history = model.fit(
    x=training_generator,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[model_checkpoint_callback]
)

model.load_weights("models/weights/" + MODEL_NAME + ".h5")
model.save("models/" + MODEL_NAME)

plot_training_history(history, MODEL_NAME)
