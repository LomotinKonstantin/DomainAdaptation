import sys
from subprocess import Popen
from configparser import ConfigParser

from keras.models import load_model
from keras.layers import LSTM, Dense

from utils import get_timestamp, train_model, count_lines


def create_classifier(model, hidden_size1: int):
    model.add(LSTM(hidden_size1, return_sequences=True, input_shape=(None, 128), name="class_lstm1"))
    model.add(Dense(1, activation='hard_sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam')
    return model


if __name__ == '__main__':
    argc = len(sys.argv)
    if argc not in [1, 2]:
        raise ValueError("Wrong arguments {}", sys.argv)
    lstm_config = ConfigParser()
    lstm_config.read("LSTM_config.ini")
    data_config = ConfigParser()
    data_config.read("data_config.ini")
    swap_config = ConfigParser()
    swap_section = "Settings"
    swap_config_file = "swap_config.ini"
    swap_config.read(swap_config_file)

    batch_size = lstm_config.getint("Settings", "batch_size")
    train_path = dict(data_config["Training"])
    test_path = dict(data_config["Testing"])
    if len(sys.argv) == 2:
        step = 2
        epoch = int(sys.argv[1])
        print("Process for epochs {}-{}".format(epoch, epoch + step))
        total_epochs = lstm_config.getint("Settings", "epochs")
        model_path = swap_config.get(swap_config, "model_path")
        model = load_model(model_path)
        steps_per_epoch = swap_config.getint(swap_section, "steps_per_epoch")
        train_model(model,
                    [train_path["movies"]],
                    batch_size=batch_size,
                    steps_per_epoch=steps_per_epoch,
                    log_fname=None,
                    memlog_fname=None,
                    epochs=min(step, total_epochs - epoch), ae=False)
        epoch += step
        model.save(model_path)
        if epoch >= total_epochs:
            print("Done", epoch, "/", total_epochs, "epochs")
            exit()

    else:
        timestamp = get_timestamp()
        model_path = "../models/AE_classifier_{}.hdf5".format(timestamp)
        model = load_model(swap_config.get(swap_section, "model_path"))
        hidden_size = lstm_config.getint("Settings", "hidden_size1")
        create_classifier(model, hidden_size)
        model.save(model_path)
        movies_lines = count_lines(train_path["movies"])
        steps_per_epoch = int(movies_lines / batch_size)
        swap_config.set(swap_section, "model_path", model_path)
        swap_config.set(swap_section, "steps_per_epoch", str(steps_per_epoch))
        swap_config.write(open(swap_config_file, "w"))
        epoch = 0

    Popen("python3 LSTM_classifier_layers.py {}".format(str(epoch)), shell=True)

