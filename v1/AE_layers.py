import sys
from subprocess import Popen
from configparser import ConfigParser

from keras.models import Sequential, load_model
from keras.layers import LSTM

from utils import get_timestamp, train_model, count_lines


def create_AE_model(latent_space_dim: int, data_dim: int):
    model = Sequential()
    model.add(LSTM(latent_space_dim, return_sequences=True,
                   input_shape=(None, data_dim)))
    model.add(LSTM(data_dim, return_sequences=True))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam')
    return model


if __name__ == '__main__':
    argc = len(sys.argv)
    if argc not in [1, 2]:
        raise ValueError("Wrong arguments {}", sys.argv)
    ae_config = ConfigParser()
    ae_config.read("AE_config.ini")
    data_config = ConfigParser()
    data_config.read("data_config.ini")
    swap_config = ConfigParser()
    swap_section = "Settings"
    swap_config_file = "swap_config.ini"

    batch_size = ae_config.getint("Training", "batch_size")
    train_path = dict(data_config["Training"])
    test_path = dict(data_config["Testing"])
    if len(sys.argv) == 2:
        step = 2
        epoch = int(sys.argv[1])
        print("Process for epochs {}-{}".format(epoch, epoch + step))
        swap_config.read(swap_config_file)
        total_epochs = ae_config.getint("Training", "epochs")
        steps_per_epoch = swap_config.getint(swap_section, "steps_per_epoch")
        model_path = swap_config.get(swap_section, "model_path")
        model = load_model(model_path)
        train_model(model,
                    [train_path["electr"], train_path["movies"]],
                    batch_size=batch_size,
                    steps_per_epoch=steps_per_epoch,
                    log_fname=None,
                    memlog_fname=None,
                    epochs=min(step, total_epochs-epoch), ae=True)
        epoch += step
        if epoch >= total_epochs:
            model.layers.pop()
            model.save(model_path)
            print("Done", epoch, "/", total_epochs, "epochs")
            exit()
        model.save(model_path)
    else:
        timestamp = get_timestamp()
        model = create_AE_model(ae_config.getint("ModelStructure", "latent_space_dim"),
                                ae_config.getint("ModelStructure", "data_dim"))
        model_path = "../models/AE_layers_popped_{}.hdf5".format(timestamp)
        model.save(model_path)

        movies_lines = count_lines(train_path["movies"])
        electr_lines = count_lines(train_path["electr"])
        total_lines = movies_lines + electr_lines
        steps_per_epoch = int(total_lines / batch_size)

        swap_config.add_section(swap_section)
        swap_config.set(swap_section, "model_path", model_path)
        swap_config.set(swap_section, "steps_per_epoch", str(steps_per_epoch))
        swap_config.write(open("swap_config.ini", "w"))
        epoch = 0
    Popen("python3 AE_layers.py {}".format(str(epoch)), shell=True)

