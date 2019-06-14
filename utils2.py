from collections import OrderedDict
from datetime import datetime
import gzip
import json
import os
import pickle

import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from gensim.models import Word2Vec
from sklearn.metrics import classification_report

from preprocessor import Preprocessor


def pad_vector(vector: list, padding_size: int) -> np.ndarray:
    if len(vector) == 0:
        array = np.zeros([1, 128])
    else:
        array = np.array([np.array(sublist) for sublist in vector])
    res = np.pad(array, ((0, padding_size - len(array)), (0, 0)),
                 mode='constant', constant_values=0.0)
    return res


def process_batch(batch: pd.DataFrame):
    if len(batch) == 0:
        return
    lens = list(map(len, batch["vectors"]))
    if not lens:
        return
    max_len = max(lens)
    batch["vectors"] = batch["vectors"].apply(pad_vector, args=[max_len])


def process_vec_batch(vectors: list) -> np.array:
    max_len = max(map(len, vectors))
    res = np.array(list(map(lambda a: pad_vector(a, max_len), vectors)))
    assert res.shape[0] == len(vectors)
    assert res.shape[1] == max_len, f"{res.shape[2]} != {max_len}"
    return res


def raw_chunk_generator(path: str,
                        chunk_size: int,
                        from_line: int = 0,
                        to_line: int = 0) -> pd.DataFrame:
    # assert to_line >= from_line
    chunk = []
    columns = ("reviewText", "overall")
    with gzip.open(path) as fp:
        for line_num, json_line in enumerate(fp):
            if line_num < from_line:
                continue
            if to_line > 0 and line_num == to_line:
                yield pd.DataFrame(chunk) if chunk else pd.DataFrame(columns=columns)
                return
            parsed_line = json.loads(json_line, object_pairs_hook=OrderedDict)
            parsed_line["reviewText"] += " " + parsed_line["summary"]
            parsed_line = {key: value for key, value in parsed_line.items()
                           if key in columns}
            chunk.append(parsed_line)
            if len(chunk) == chunk_size:
                yield pd.DataFrame(chunk)
                chunk = []
    if chunk:
        yield pd.DataFrame(chunk)


def vector_chunk_generator(path: str,
                           chunk_size: int,
                           w2v_model: Word2Vec,
                           from_line: int = 0,
                           to_line: int = 0,
                           pad=True
                           ) -> pd.DataFrame:
    preprocessor = Preprocessor()
    for chunk in raw_chunk_generator(path,
                                     chunk_size,
                                     from_line=from_line,
                                     to_line=to_line):
        vec_lst = []
        for i in chunk.index:
            vec_lst.append(raw_to_vec(chunk.loc[i, "reviewText"],
                                      preprocessor,
                                      w2v_model))
        chunk["vectors"] = vec_lst
        if pad:
            process_batch(chunk)
        yield chunk


def raw_to_vec(text: str, prepr: Preprocessor, w2v_model: Word2Vec):
    return text_to_matr(prepr.preprocess(text), w2v_model)


def text_to_matr(text: str, w2v_model: Word2Vec) -> np.array:
    tokens = text.split()
    vec_lst = []
    for word in tokens:
        try:
            vec_lst.append(np.array(w2v_model.wv.get_vector(word)))
        except KeyError:
            pass
    return np.array(vec_lst)


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


def train_data_generator(files: list,
                         test_percent: float,
                         chunk_size: int,
                         w2v_model: Word2Vec,
                         line_counts: dict,
                         autoencoder: bool) -> np.array:
    for i in files:
        assert os.path.exists(i)
        assert i in line_counts
    assert 0 < test_percent < 1
    assert chunk_size > 0
    for fname in files:
        to_line = int(line_counts[fname] * (1 - test_percent))
        gen = vector_chunk_generator(fname,
                                     chunk_size=chunk_size,
                                     w2v_model=w2v_model, to_line=to_line)
        for chunk in gen:
            x = chunk["vectors"].values
            x = np.array(list(x))
            y = chunk["overall"].values
            y[y <= 3] = 0
            y[y > 3] = 1
            y = y.reshape([-1, 1, 1])
            if autoencoder:
                yield x, x
            else:
                yield x, y


def test_data_generator(files: list,
                        test_percent: float,
                        chunk_size: int,
                        w2v_model: Word2Vec,
                        line_counts: dict) -> np.array:
    for i in files:
        assert os.path.exists(i)
        assert i in line_counts
    assert 0 <= test_percent <= 1
    assert chunk_size > 0
    for fname in files:
        from_line = int(line_counts[fname] * (1 - test_percent)) + 1
        gen = vector_chunk_generator(fname,
                                     chunk_size=chunk_size,
                                     w2v_model=w2v_model, from_line=from_line)
        for chunk in gen:
            x = chunk["vectors"].values
            x = np.array(list(x))
            y = chunk["overall"].values
            y[y <= 3] = 0
            y[y > 3] = 1
            y = y.reshape([-1, 1, 1])
            yield x, y


def infinite_tr_vect_gen(files: list,
                         test_percent: float,
                         chunk_size: int,
                         w2v_model: Word2Vec,
                         line_counts: dict,
                         autoencoder: bool) -> np.array:
    while True:
        g = train_data_generator(files=files,
                                 chunk_size=chunk_size,
                                 w2v_model=w2v_model,
                                 line_counts=line_counts,
                                 autoencoder=autoencoder,
                                 test_percent=test_percent)
        for chunk in g:
            if len(chunk) == 0:
                continue
            yield chunk


def train_model(model,
                train_files: list,
                batch_size: int,
                epochs: int,
                ae: bool,
                line_count_hint: dict,
                test_percent: float,
                w2v_model: Word2Vec,
                # checkpoint_fpath: str,
                verbose=1,
                noise_decorator=None):
    # callbacks = [ModelCheckpoint(checkpoint_fpath, save_best_only=True, period=10)]
    generator = infinite_tr_vect_gen(files=train_files,
                                     test_percent=test_percent,
                                     chunk_size=batch_size,
                                     line_counts=line_count_hint,
                                     autoencoder=ae,
                                     w2v_model=w2v_model)
    if noise_decorator:
        generator = noise_decorator(generator, ae)
    steps_per_epoch = sum(line_count_hint.values()) * (1 - test_percent) / batch_size
    steps_per_epoch = int(steps_per_epoch)
    model.fit_generator(generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        verbose=verbose,
                        # callbacks=callbacks
                        )


def test_model(model,
               test_paths: list,
               test_percent: float,
               line_count_hint: dict,
               w2v_model: Word2Vec,
               report_path: str,
               batch_size: int,
               comment="",
               norm=False):
    # assert os.path.exists(report_path)
    y_true = []
    y_pred = []
    cntr = 1
    for X_test, y_test in test_data_generator(files=test_paths,
                                              chunk_size=batch_size,
                                              line_counts=line_count_hint,
                                              test_percent=test_percent,
                                              w2v_model=w2v_model):
        print(f"Testing batch {cntr}")
        cntr += 1
        predict = model.predict(X_test)
        if norm:
            predict = np.array(list(map(lambda x: (x > x.mean()).astype(int), predict)))
        predict = np.round(predict.mean(axis=1).reshape(-1)).astype(int)
        y_true.extend(y_test.reshape(y_test.shape[0]))
        y_pred.extend(predict)
    report = classification_report(y_true, y_pred)
    with open(report_path, "w") as report_file:
        report_file.write(report)
        report_file.write(f"\n{comment}\n")


def _pickle_gen(pickle_file: str):
    with gzip.open(pickle_file, "rb") as fp:
        while 1:
            try:
                yield pickle.load(fp)
            except EOFError:
                return


def pickle_vectors_generator(pickle_file: str, batch_size: int, from_line=0, to_line=-1):
    x = []
    y = []
    for n, line in enumerate(_pickle_gen(pickle_file)):
        if n < from_line:
            continue
        x.append(line[0])
        y.append(line[1])
        if len(x) == batch_size or n == to_line:
            y = np.array(list(y))
            y[y <= 3] = 0
            y[y > 3] = 1
            y = y.reshape([-1, 1, 1])
            yield process_vec_batch(x), y
            if n == to_line:
                return
            x = []
            y = []
    if x and y:
        y = np.array(y)
        y[y <= 3] = 0
        y[y > 3] = 1
        y = y.reshape([-1, 1, 1])
        yield process_vec_batch(x), y


if __name__ == '__main__':
    w2v = Word2Vec.load("../models/w2v_5dom.model")
    gen = train_data_generator(["../data/Kindle_Store_5.json.gz"],
                               line_counts={"../data/Kindle_Store_5.json.gz": 982_619},
                               chunk_size=100, w2v_model=w2v, autoencoder=False,
                               test_percent=0.3)
    x, y = next(gen)
    print(x.shape)
    print(np.unique(y, return_counts=True))
