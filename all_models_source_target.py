import os

from gensim.models import Word2Vec
import keras.backend as K
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, LSTM, Input
from numpy.random import normal, randint
import numpy as np

from utils2 import get_timestamp, train_model, test_model
from gradient_reversal_keras_tf.flipGradientTF import GradientReversal

# line_counts = {
#     "../data/Apps_for_Android_5.json.gz": 752_937,
#     "../data/CDs_and_Vinyl_5.json.gz": 1_097_592,
#     "../data/Electronics_5.json.gz": 1_689_188,
#     "../data/Kindle_Store_5.json.gz": 982_619,
#     "../data/Movies_and_TV_5.json.gz": 1_697_533
# }

line_counts = {
    "../data/Apps_for_Android_5.json.gz": 100_000,
    "../data/CDs_and_Vinyl_5.json.gz": 100_000,
    "../data/Electronics_5.json.gz": 100_000,
    "../data/Kindle_Store_5.json.gz": 100_000,
    "../data/Movies_and_TV_5.json.gz": 100_000,
}

BIN_DROP = 0.1
epochs = 1
batch_size = 100
test_percent = 0.3
w2v_model = Word2Vec.load("../models/w2v_5dom.model")
train_files = list(line_counts.keys())
source_domain = "../data/Movies_and_TV_5.json.gz"
target_domains = [i for i in train_files if i != source_domain]
data_folder = "../data/"


def domain_name_from_file(fname: str) -> str:
    part = fname.split("_5")[0]
    return part.split("/")[-1]


def gaussian_noise(generator, ae: bool):
    def noise_gen():
        for X, y in generator:
            if ae:
                yield X + normal(0, 0.02, X.shape), X
            else:
                yield X + normal(0, 0.02, X.shape), y

    return noise_gen()


def binary_noise(generator, ae: bool):
    def noise_gen():
        for X, y in generator:
            for n, matr in enumerate(X):
                x_indices = randint(0, matr.shape[0], size=BIN_DROP * matr.shape[0])
                y_indices = randint(0, matr.shape[1], size=BIN_DROP * matr.shape[1])
                X[n, x_indices, y_indices] = 0
            if ae:
                yield X, X
            else:
                yield X, y
    return noise_gen()


def create_lstm_classifier(model=None):
    if model is None:
        model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(None, 128)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='softmax'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adagrad')
    return model


def create_AE_model(latent_space_dim: int, data_dim: int, model=None):
    if model is None:
        model = Sequential()
    model.add(LSTM(latent_space_dim, return_sequences=True,
                   input_shape=(None, data_dim)))
    model.add(LSTM(data_dim, return_sequences=True))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam')
    return model


def create_SDAE_model(layers: list, data_dim: int):
    model = Sequential()
    model.add(LSTM(layers[0], return_sequences=True,
                   input_shape=(None, data_dim)))
    for latent_space_dim in layers[1:]:
        model.add(LSTM(latent_space_dim, return_sequences=True))
    model.add(LSTM(data_dim, return_sequences=True))
    model.compile(loss='binary_crossentropy',
                  optimizer='adagrad')
    return model


def create_DANN():
    inputs = Input((None, 128))
    # Feature extractor
    feature_extractor = LSTM(128, return_sequences=True)(inputs)
    feature_extractor = LSTM(64, return_sequences=True)(feature_extractor)
    feature_extractor = LSTM(32, return_sequences=True)(feature_extractor)
    # Domain adversarial layers
    d_a = GradientReversal(0.5)(feature_extractor)
    d_a = LSTM(64, return_sequences=True)(d_a)
    d_a = Dense(64, activation='relu')(d_a)
    d_a = Dense(32, activation='relu')(d_a)
    d_a = Dense(1, activation="softmax", name="domain_adv")(d_a)
    # Classification layers
    classifier = LSTM(128, return_sequences=True)(feature_extractor)
    classifier = Dense(128, activation='relu')(classifier)
    classifier = Dense(32, activation='relu')(classifier)
    classifier = Dense(1, activation="softmax", name="classifier")(classifier)

    comb_model = Model(inputs=inputs, outputs=[classifier, d_a])
    comb_model.compile(loss='binary_crossentropy', optimizer='adadelta')

    classifier_model = Model(inputs=inputs, outputs=[classifier])
    classifier_model.compile(loss='binary_crossentropy', optimizer='adadelta')

    domain_adapt_model = Model(inputs=inputs, outputs=d_a)
    domain_adapt_model.compile(loss='binary_crossentropy', optimizer='adadelta')

    return classifier_model, domain_adapt_model, comb_model


def train_on_source(model, ae: bool):
    train_model(model, train_files=[source_domain], batch_size=batch_size,
                epochs=epochs, ae=ae, line_count_hint=line_counts,
                test_percent=test_percent, w2v_model=w2v_model)


def test_on_source(model, report_path: str):
    test_model(model, batch_size=batch_size, line_count_hint=line_counts,
               test_paths=[source_domain], test_percent=test_percent,
               w2v_model=w2v_model, report_path=report_path)


def test_on_target(model, report_path_with_format: str):
    for i in target_domains:
        path = i
        print(f"Testing on {i}")
        report_path = report_path_with_format.format(domain_name_from_file(i))
        test_model(model, batch_size=batch_size, line_count_hint=line_counts,
                   test_paths=[path], test_percent=test_percent, w2v_model=w2v_model,
                   report_path=report_path)


def train_and_test_on_target(ae: bool,
                             model_name: str,
                             report_folder: str,
                             model_folder: str,
                             clear_model: str):
    for i in target_domains:
        model = load_model(clear_model)
        domain = domain_name_from_file(i)
        report_path = report_folder + f"{model_name}_target-target_{domain}.csv"
        print(f"Training on {i}")
        train_model(model, train_files=[i],
                    batch_size=batch_size, epochs=epochs, ae=ae,
                    line_count_hint=line_counts, test_percent=test_percent,
                    w2v_model=w2v_model)
        print(f"Testing on {i}")
        test_model(model, batch_size=batch_size, line_count_hint=line_counts,
                   test_paths=[i], test_percent=test_percent, w2v_model=w2v_model,
                   report_path=report_path)
        model.save(model_folder + f"{model_name}_{domain_name_from_file(i)}.hdf5")


# 1. Обучить на source без адаптации
# 2. Протестировать на source и target
# 3. Обучить на source с адаптацией
# 4. Протестировать на source и target
# 5. Обучить на таргете
# 6. Протестировать на таргете
if __name__ == '__main__':
    timestamp = get_timestamp()
    data_folder = "../data/"
    report_folder = f"../reports/{timestamp}/"
    model_folder = f"../models/{timestamp}/"
    os.mkdir(report_folder)
    os.mkdir(model_folder)

    # LSTM
    model = create_lstm_classifier()
    cp_fp = model_folder + "LSTM_source_{epoch}.hdf5"
    report_path = report_folder + "LSTM_source-source.csv"
    print("Training LSTM")
    train_on_source(model, ae=False, cp_fp=cp_fp)
    print("Testing LSTM on source")
    test_on_source(model, report_path)
    report_path = report_folder + "LSTM_source-target_{}.csv"
    print("Testing LSTM on target")
    test_on_target(model, report_path)
    print("Training and testing LSTM on target")
    train_and_test_on_target(model, ae=False, model_name="LSTM",
                             report_folder=report_folder, model_folder=model_folder)
    K.clear_session()
    exit(0)
    # AE + LSTM
    model = create_AE_model(64, 128)
    cp_fp = model_folder + "AE_layers_{epoch}.hdf5"
    report_path = report_folder + "AE_LSTM_test_report.csv"
    print("Training AE layers")
    train_model(model, train_files=train_files, batch_size=batch_size,
                epochs=epochs, ae=True, line_count_hint=line_counts,
                test_percent=test_percent, w2v_model=w2v_model, checkpoint_fpath=cp_fp)
    model.layers.pop()
    model = create_lstm_classifier(model)
    cp_fp = model_folder + "AE_LSTM_{epoch}.hdf5"
    print("Training LSTM AE")
    train_model(model, train_files=train_files, batch_size=batch_size,
                epochs=epochs, ae=False, line_count_hint=line_counts,
                test_percent=test_percent, w2v_model=w2v_model, checkpoint_fpath=cp_fp)
    print("Testing LSTM AE")
    test_model(model, batch_size=batch_size, line_count_hint=line_counts,
               test_paths=train_files, test_percent=test_percent, w2v_model=w2v_model,
               report_path=report_path)

    K.clear_session()

    report_path = report_folder + "SDAE_LSTM_test_report.csv"
    model = None
    for latent_space_dim in [128, 72, 64]:
        model = create_AE_model(latent_space_dim, 128, model)
        cp_fp = model_folder + "SDAE_layers_" + str(latent_space_dim) + "_{epoch}.hdf5"
        print(f"Training AE layers {latent_space_dim}")
        train_model(model, train_files=train_files, batch_size=batch_size,
                    epochs=epochs, ae=True, line_count_hint=line_counts,
                    test_percent=test_percent, w2v_model=w2v_model,
                    checkpoint_fpath=cp_fp, noise_decorator=gaussian_noise)
        model.layers.pop()
    model = create_lstm_classifier(model)
    cp_fp = model_folder + "SDAE_LSTM_{epoch}.hdf5"
    print("Training LSTM SDAE")
    train_model(model, train_files=train_files, batch_size=batch_size,
                epochs=epochs, ae=False, line_count_hint=line_counts,
                test_percent=test_percent, w2v_model=w2v_model, checkpoint_fpath=cp_fp)
    print("Testing LSTM SDAE")
    test_model(model, batch_size=batch_size, line_count_hint=line_counts,
               test_paths=train_files, test_percent=test_percent, w2v_model=w2v_model,
               report_path=report_path)

    K.clear_session()
