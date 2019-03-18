from os.path import exists
from os import mkdir

from keras.models import Sequential
from keras.layers import LSTM, Dense

from utils import get_timestamp, train_model, count_lines, test_model, data_generator


def validate_data(paths: list):
    for file in paths:
        g = data_generator(file, 1000)
        for n, (x, y) in enumerate(g):
            if x.shape[2] != 128:
                print("Invalid batch #{} with size {} in file {}:"
                      "\n X.shape = {}, y.shape = {}".format(n,
                                                             1000,
                                                             file,
                                                             x.shape,
                                                             y.shape))
                exit()


if __name__ == '__main__':
    print("Started")
    timestamp = get_timestamp()
    train_path = {"electr": "../data/train_electr_vectors_balanced.csv",
                  "movies": "../data/train_movies_vectors_balanced.csv"}
    test_path = {"electr": "../data/test_electr_vectors_balanced.csv",
                 "movies": "../data/test_movies_vectors_balanced.csv"}
    # print("Validating data shape")
    # validate_data([train_path["electr"],
    #                train_path["movies"],
    #                test_path["electr"],
    #                test_path["movies"]])
    data_dim = 128
    num_classes = 2
    latent_space_dim = 32
    print("Creating model")
    print("Training AE layers")
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    model.add(LSTM(latent_space_dim, return_sequences=True,
                   input_shape=(None, data_dim)))
    model.add(LSTM(data_dim, return_sequences=True))

    model.compile(loss='mean_squared_error',
                  optimizer='adagrad')
    batch_size = 50
    epochs = 10
    movies_lines = count_lines(train_path["movies"])
    electr_lines = count_lines(train_path["electr"])
    total_lines = movies_lines + electr_lines
    steps_per_epoch = int(total_lines / batch_size)
    report_folder = "../reports/{}".format(timestamp)
    if not exists(report_folder):
        mkdir(report_folder)
    log_fname = '{}/training_AE.csv'.format(report_folder)
    train_model(model,
                [train_path["electr"], train_path["movies"]],
                batch_size=batch_size,
                steps_per_epoch=steps_per_epoch,
                log_fname=log_fname,
                epochs=epochs, ae=True)
    model.layers.pop()
    model.save("../models/AE_layers_popped_{}.hdf5".format(timestamp))
    hidden_size1 = 64
    hidden_size2 = 128
    model.add(LSTM(hidden_size1, return_sequences=True, input_shape=(None, 128)))
    # model.add(LSTM(hidden_size2, return_sequences=True))
    # model.add(Dense(hidden_size2, activation="hard_sigmoid"))
    model.add(Dense(1, activation='hard_sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adagrad')
    steps_per_epoch = int(movies_lines / batch_size)
    epochs = 10
    log_fname = '{}/training_AE_classifier.csv'.format(report_folder)
    print("Training classifier")
    train_model(model,
                train_path["movies"],
                batch_size=batch_size,
                steps_per_epoch=steps_per_epoch,
                log_fname=log_fname,
                epochs=epochs)
    print("Testing model")
    report_path = "{}/report_LSTM_AE_source.csv".format(report_folder)
    test_model(model, test_path["movies"], report_path, batch_size)
    report_path = "{}/report_LSTM_AE_target.csv".format(report_folder)
    test_model(model, test_path["electr"], report_path, batch_size)
    model.save("../models/LSTM_AE_{}_{}.hdf5".format(hidden_size1, timestamp))
    print("Done!")
