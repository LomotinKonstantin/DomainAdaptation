from utils import *

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.backend import clear_session

if __name__ == '__main__':
    print("Started")
    timestamp = get_timestamp()
    train_path = {"electr": "../data/train_electr_vectors_balanced.csv",
                  "movies": "../data/train_movies_vectors_balanced.csv"}
    # debug_train_path = {"electr": "../data/train_electr_vectors_balanced.csv",
    #                     "movies": "../data/movies_vectors_balanced.csv"}
    test_path = {"electr": "../data/test_electr_vectors_balanced.csv",
                 "movies": "../data/test_movies_vectors_balanced.csv"}
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
    steps_per_epoch = int(count_lines(train_path["movies"]) / batch_size)
    log_fname = '../reports/training_log_{}.csv'.format(timestamp)
    report_path = "../reports/report_LSTM_v3_{}.csv".format(timestamp)
    train_model(model, train_path["movies"],
                batch_size,
                steps_per_epoch,
                log_fname)
    print("Testing model")
    test_model(model, test_path["movies"], report_path, batch_size)
    model.save("../models/LSTM_v3.hdf5")
    print("Done!")
