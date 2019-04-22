import sys
from subprocess import Popen
from configparser import ConfigParser

from keras.models import load_model
from keras.layers import LSTM, Dense, Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint

from gradient_reversal_keras_tf.flipGradientTF import GradientReversal

from utils import *


def unison_shuffled_copies(*arrays):
    arrays = list(arrays)
    p = np.random.permutation(arrays[0])
    for i in range(len(arrays)):
        arrays[i] = arrays[i][p]
    return arrays


def train_DANN(classifier_model,
               da_model,
               comb_model,
               source_path: str,
               target_path: str,
               # source_lines: int,
               # target_lines: int,
               batch_size: int,
               # steps_per_epoch: int,
               epochs: int,
               comb_model_path: str,
               class_model_path: str):
    source_gen = indefinite_data_generator(source_path, batch_size,)
    target_gen = indefinite_data_generator(target_path, batch_size,)
    for i in range(epochs):
        print("Epoch {} / {}".format(i + 1, epochs))
        X_s, y_s = next(source_gen)
        # Source domain кодируем 0
        domain_s = np.zeros(y_s.shape)
        X_t, y_t = next(target_gen)
        # Target domain кодируем 1
        domain_t = np.full(y_t.shape, fill_value=1)
        classifier_model.fit(X_s, y_s, epochs=1, verbose=0)
        da_model.fit(X_s, domain_s, epochs=1, verbose=0)
        da_model.fit(X_t, domain_t, epochs=1, verbose=0)

        if i % 10 == 0:
            classifier_model.save(class_model_path)
            comb_model.save(comb_model_path)


def create_DANN():
    inputs = Input((None, 128))
    # Feature extractor
    feature_extractor = LSTM(128, return_sequences=True)(inputs)
    feature_extractor = LSTM(64, return_sequences=True)(feature_extractor)
    feature_extractor = LSTM(32, return_sequences=True)(feature_extractor)
    # Domain adversarial layers
    d_a = GradientReversal(0.5)(feature_extractor)
    d_a = LSTM(64, return_sequences=True)(d_a)
    d_a = LSTM(64, return_sequences=True)(d_a)
    d_a = Dense(1, activation="softmax", name="domain_adv")(d_a)
    # Classification layers
    classifier = LSTM(128, return_sequences=True)(feature_extractor)
    classifier = Dense(1, activation="softmax", name="classifier")(classifier)

    comb_model = Model(inputs=inputs, outputs=[classifier, d_a])
    comb_model.compile(loss='binary_crossentropy', optimizer='adagrad')

    classifier_model = Model(inputs=inputs, outputs=[classifier])
    classifier_model.compile(loss='binary_crossentropy', optimizer='adagrad')

    domain_adapt_model = Model(inputs=inputs, outputs=d_a)
    domain_adapt_model.compile(loss='binary_crossentropy', optimizer='adagrad')

    return classifier_model, domain_adapt_model, comb_model


if __name__ == '__main__':
    section = "Settings"
    dann_config = ConfigParser()
    dann_config.read("DANN_config.ini")

    data_config = ConfigParser()
    data_config.read("data_config.ini")
    train_path = dict(data_config["Training"])

    batch_size = dann_config.getint(section, "batch_size")
    epochs = dann_config.getint(section, "epochs")
    # movies_lines = count_lines(train_path["movies"])
    # electr_lines = count_lines(train_path["electr"])
    # steps_per_epoch = max([movies_lines, electr_lines]) / epochs
    cl_model, da_model, comb_model = create_DANN()
    timestamp = get_timestamp()
    comb_model_path = "../models/comb_DANN_checkpoint_{}.hdf5".format(timestamp)
    class_model_path = "../models/classifier_DANN_checkpoint_{}.hdf5".format(timestamp)
    train_DANN(classifier_model=cl_model,
               da_model=da_model,
               source_path=train_path["movies"],
               target_path=train_path["electr"],
               batch_size=batch_size,
               epochs=epochs,
               comb_model=comb_model,
               comb_model_path=comb_model_path,
               class_model_path=class_model_path)
    comb_model.save("../models/comb_DANN_final_{}.hdf5".format(timestamp))
    cl_model.save("../models/classifier_DANN_final_{}.hdf5".format(timestamp))
