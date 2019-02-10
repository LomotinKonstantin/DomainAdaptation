import sys
import os
from datetime import datetime
from keras.models import load_model
from keras.callbacks import CSVLogger
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def count_lines(filename: str) -> int:
    count = 0
    for line in open(filename):
        count += 1
    return count


def get_timestamp() -> str:
    timestamp = datetime.today().strftime("%d-%b-%Y__%X")
    timestamp = timestamp.replace(":", "-")
    return timestamp


def append_timestamp(path: str) -> str:
    parts = os.path.split(path)
    filename = parts[-1]
    filename_parts = filename.split(".")
    new_filename = ".".join([".".join(filename_parts[:-1]) + "_" + get_timestamp(),
                             filename_parts[-1]])
    return os.path.join(*parts[:-1], new_filename)


def process_vector(vector: list, padding_size: int) -> np.ndarray:
    if len(vector) == 0:
        array = np.zeros([1, 128])
    else:
        array = np.array([np.array(sublist) for sublist in vector])
    res = np.pad(array, ((0, padding_size - len(array)), (0, 0)),
                 mode='constant', constant_values=0.0)
    return res


def process_batch(batch: pd.DataFrame):
    max_len = max(map(len, batch["vectors"]))
    batch["vectors"] = batch["vectors"].apply(process_vector, args=[max_len])


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


def data_generator(path: str, batch_size: int) -> tuple:
    generator = batch_generator(fname=path,  # from_line=9200, to_line=9500,
                                batch_size=batch_size)
    for num, batch in enumerate(generator):
        process_batch(batch)
        X = batch["vectors"].values
        y = batch["target_bin"].values
        # Postprocessing
        X = np.array(list(X))
        y = y.reshape([-1, 1, 1])
        yield X, y


def indefinite_data_generator(path: str, batch_size: int):
    while True:
        generator = data_generator(path, batch_size)
        for data_tuple in generator:
            yield data_tuple


def train_model(model,
                train_path: str,
                batch_size: int,
                steps_per_epoch: int,
                log_fname: str):
    csv_logger = CSVLogger(log_fname,
                           append=True, separator='\t')
    generator = indefinite_data_generator(train_path, batch_size)
    model.fit_generator(generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=3,
                        verbose=1,
                        callbacks=[csv_logger], use_multiprocessing=True)


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
