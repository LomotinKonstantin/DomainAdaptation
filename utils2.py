from collections import OrderedDict
from datetime import datetime
import gzip
import json
import os

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from preprocessor import Preprocessor


def raw_chunk_generator(path: str,
                        chunk_size: int,
                        from_line: int = 0,
                        to_line: int = 0) -> pd.DataFrame:
    assert to_line >= from_line
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


def training_data_generator(files: list,
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
            y = chunk["overall"].values
            y[y <= 3] = 0
            y[y > 3] = 1
            if autoencoder:
                yield x, x
            else:
                yield x, y


def infinite_tr_vect_gen(files: list,
                         test_percent: float,
                         chunk_size: int,
                         w2v_model: Word2Vec,
                         line_counts: dict,
                         autoencoder: bool) -> np.array:
    while True:
        g = training_data_generator(files=files,
                                    chunk_size=chunk_size,
                                    w2v_model=w2v_model,
                                    line_counts=line_counts,
                                    autoencoder=autoencoder,
                                    test_percent=test_percent)
        for chunk in g:
            yield chunk


def train_model(model,
                train_files: list,
                batch_size: int,
                epochs: int,
                ae: bool,
                line_count_hint: dict,
                test_percent: float,
                w2v_model: Word2Vec,
                verbose=0,
                noise_decorator=None):
    callbacks = []
    generator = infinite_tr_vect_gen(files=train_files,
                                     test_percent=test_percent,
                                     chunk_size=batch_size,
                                     line_counts=line_count_hint,
                                     autoencoder=ae,
                                     w2v_model=w2v_model)
    steps_per_epoch = sum(line_count_hint.values()) * (1 - test_percent) / batch_size
    steps_per_epoch = int(steps_per_epoch)
    model.fit_generator(generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        verbose=verbose,
                        callbacks=callbacks)


if __name__ == '__main__':
    w2v = Word2Vec.load("../models/w2v_5dom.model")
    gen = vector_chunk_generator("../data/Kindle_Store_5.json.gz",
                                 chunk_size=10, to_line=10, from_line=100, w2v_model=w2v)
    df = next(gen)
    print(df)
