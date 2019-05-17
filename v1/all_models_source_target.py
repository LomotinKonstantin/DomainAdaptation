line_counts = {
    "Apps_for_Android_5.json.gz": 752937,
    "CDs_and_Vinyl_5.json.gz": 1097592,
    "Electronics_5.json.gz": 1689188,
    "Kindle_Store_5.json.gz": 982619,
    "Movies_and_TV_5.json.gz": 1697533
}


def create_lstm_classifier():
    model = Sequential()
    model.add(LSTM(hidden_size1, return_sequences=True, input_shape=(None, 128)))
    # model.add(LSTM(hidden_size2, return_sequences=True))
    # model.add(Dense(hidden_size2, activation="hard_sigmoid"))
    model.add(Dense(1, activation='hard_sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adagrad')
    print("Starting model training")
    batch_size = 1000
    epochs = 2
    steps_per_epoch = int(count_lines(train_path["movies"]) / batch_size)
    log_fname = '../reports/training_log_{}.csv'.format(timestamp)
    report_path = "../reports/report_LSTM_v3_{}.csv".format(timestamp)
    train_model(model, train_path["movies"],
                batch_size,
                steps_per_epoch,
                log_fname, epochs=epochs)
    print("Testing model")
    test_model(model, test_path["movies"], report_path, batch_size)
    model.save("../models/LSTM_v3_{}.hdf5".format(hidden_size1))
    print("Done!")


if __name__ == '__main__':
    pass
