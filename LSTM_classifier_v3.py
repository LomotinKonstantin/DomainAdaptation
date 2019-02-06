import tensorflow as tf
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.backend import clear_session
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.callbacks import CSVLogger


def process_vector(vector: list, padding_size: int) -> np.ndarray:
    array = np.array([np.array(sublist) for sublist in vector])
    if len(array) == 0:
        return np.zeros([1, 1, 128])
    try:
        return np.pad(array, ((0, padding_size-len(array)), (0, 0)),
                      mode='constant', constant_values=0.0)
    except Exception as e:
        return np.zeros([1, 1, 128])


def process_batch(batch: pd.DataFrame):
    max_len = max(map(len, batch["vectors"]))
    batch["vectors"] = batch["vectors"].apply(process_vector, args=[max_len])
#     batch["target_bin"] = batch["target_bin"].values.reshape([-1, 1, 1])


def batch_generator(fname: str,
                    batch_size: int,
                    from_line=None,
                    to_line=None) -> pd.DataFrame:
    skiprows = None
    if from_line is not None:
        skiprows = range(1, from_line)
    nrows = to_line
    if from_line is not None and to_line is not None:
        nrows = to_line - from_line
    for batch in pd.read_csv(open(fname), sep="\t",
                             chunksize=batch_size,
                             skiprows=skiprows, nrows=nrows):
        batch["vectors"] = batch["vectors"].apply(eval)
        yield batch


def train_test_generator(fname: str,
                         batch_size: int,
                         test_percent: float) -> pd.DataFrame:
    generator = batch_generator(fname=fname,
                                batch_size=batch_size)
    for num, batch in enumerate(generator):
        print("Train/test batch", num + 1)
        process_batch(batch)
        X_train, X_test, y_train, y_test = train_test_split(batch["vectors"].values,
                                                            batch["target_bin"].values,
                                                            test_size=test_percent)
        y_train = y_train.reshape([-1, 1, 1])
        y_test = y_test.reshape([-1, 1, 1])
        X_train = np.array(list(X_train))
        X_test = np.array(list(X_test))
        yield (X_train, X_test, y_train, y_test)


def train_test_model(model):
    for X_train, X_test, y_train, y_test in train_test_generator(movies_vectors_file,
                                                                 batch_size=5000,
                                                                 test_percent=0.2):
        csv_logger = CSVLogger('../reports/training_log.csv',
                               append=True, separator='\t')
        model.fit(X_train, y_train,
                  validation_data=(X_test, y_test),
                  steps_per_epoch=50, epochs=30, verbose=1,
                  validation_steps=30, callbacks=[csv_logger])


def data_generator(path: str, batch_size: int) -> tuple:
    generator = batch_generator(fname=path, from_line=9000, to_line=9500,
                                batch_size=batch_size)
    for num, batch in enumerate(generator):
        process_batch(batch)
        X = batch["vectors"].values
        y = batch["target_bin"].values
        # Postprocessing
        X = np.array(list(X))
        y = y.reshape([-1, 1, 1])
        yield X, y


def train_model(model, train_path: str, batch_size: int):
    csv_logger = CSVLogger('../reports/training_log.csv',
                           append=True, separator='\t')
    cntr = 1
    for X_train, y_train in data_generator(train_path, batch_size):
        try:
            print("Training batch ", cntr, "of shape", X_train.shape)
            cntr += 1
            model.fit(X_train, y_train,
                      # steps_per_epoch=50,
                      epochs=3,
                      verbose=1,
                      callbacks=[csv_logger])
        except ValueError as ve:
            print("X:\n", X_train)
            print("y:\n", y_train)
            raise ve


def test_model(model, test_path: str, report_path: str, batch_size: int):
    y_true = []
    y_pred = []
    cntr = 1
    for X_test, y_test in data_generator(test_path, batch_size):
        print("Testing batch ", cntr)
        cntr += 1
        predict = np.round(model.predict(X_test).mean(axis=1).reshape(-1))
        predict = list(map(int, predict))
        y_true.extend(y_test.reshape(y_test.shape[0]))
        y_pred.extend(predict)
    report = classification_report(y_true, y_pred)
    with open(report_path, "w") as report_file:
        report_file.write(report)


if __name__ == '__main__':
    print("Started")
    train_path = {"electr": "../data/train_electr_vectors_balanced.csv",
                  "movies": "../data/train_movies_vectors_balanced.csv"}
    test_path = {"electr": "../data/test_electr_vectors_balanced.csv",
                 "movies": "../data/test_movies_vectors_balanced.csv"}
    report_path = "../reports/report_LSTM_v3.csv"
    print("Creating model")
    clear_session()
    hidden_size1 = 32
    hidden_size2 = 256
    model = Sequential()
    model.add(LSTM(hidden_size1, return_sequences=True, input_shape=(None, 128)))
    # model.add(LSTM(hidden_size2, return_sequences=True))
    # model.add(Dense(hidden_size2, activation="hard_sigmoid"))
    model.add(Dense(1, activation='hard_sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adagrad')
    print("Starting model training")
    batch_size = 3000
    train_model(model, train_path["movies"], batch_size)
    print("Testing model")
    test_model(model, test_path["movies"], report_path, batch_size)
    model.save("../models/LSTM_v3.hdf5")



