from os.path import exists, getctime, join
from os import mkdir, listdir

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense

from utils import get_timestamp, train_model, count_lines, test_model, data_generator


def create_AE_model(latent_space_dim: int, data_dim: int):
    model = Sequential()
    model.add(LSTM(latent_space_dim, return_sequences=True,
                   input_shape=(None, data_dim)))
    model.add(LSTM(data_dim, return_sequences=True))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam')
    return model


def most_recent_AE_model(folder: str):
    files = listdir(folder)
    ae_models = list(filter(
        lambda a: a.startswith("AE_layers_popped"),
        files
    ))
    if len(ae_models) == 0:
        return None
    most_recent = max(ae_models,
                      key=lambda a: getctime(join("..", "models", a)))
    return load_model(most_recent)


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

    batch_size = 2000
    epochs = 20
    movies_lines = count_lines(train_path["movies"])
    electr_lines = count_lines(train_path["electr"])
    total_lines = movies_lines + electr_lines
    steps_per_epoch = int(total_lines / batch_size)
    report_folder = "../reports/{}".format(timestamp)
    if not exists(report_folder):
        mkdir(report_folder)
    log_fname = '{}/training_AE.csv'.format(report_folder)
    mem_logfile_AE = "{}/memlog_AE.log".format(report_folder)
    mem_logfile_classifier = "{}/memlog_classifier.log".format(report_folder)
    model = create_AE_model()
    train_model(model,
                [train_path["electr"], train_path["movies"]],
                batch_size=batch_size,
                steps_per_epoch=steps_per_epoch,
                log_fname=log_fname,
                memlog_fname=mem_logfile_AE,
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
                  optimizer='adam')
    steps_per_epoch = int(movies_lines / batch_size)
    epochs = 20
    log_fname = '{}/training_AE_classifier.csv'.format(report_folder)
    print("Training classifier")
    train_model(model,
                train_path["movies"],
                batch_size=batch_size,
                steps_per_epoch=steps_per_epoch,
                log_fname=log_fname,
                memlog_fname=mem_logfile_classifier,
                epochs=epochs)
    print("Testing model")
    report_path = "{}/report_LSTM_AE_source.csv".format(report_folder)
    test_model(model, test_path["movies"], report_path, batch_size)
    report_path = "{}/report_LSTM_AE_target.csv".format(report_folder)
    test_model(model, test_path["electr"], report_path, batch_size)
    model.save("../models/LSTM_AE_{}_{}.hdf5".format(hidden_size1, timestamp))
    print("Done!")
