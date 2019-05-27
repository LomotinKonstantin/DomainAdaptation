import gzip
import os
import pickle

from gensim.models import Word2Vec
import numpy as np
import pandas as pd

from utils2 import vector_chunk_generator, pickle_vectors_generator

data_files = {
    # "../data/Apps_for_Android_5.json.gz": 752_937,
    # "../data/CDs_and_Vinyl_5.json.gz": 1_097_592,
    "../data/Electronics_5.json.gz": 1_689_188,
    "../data/Kindle_Store_5.json.gz": 982_619,
    "../data/Movies_and_TV_5.json.gz": 1_697_533
}


def create_vectors(in_file: str,
                   out_file: str,
                   batch_size: int,
                   w2v_model: Word2Vec):
    assert os.path.exists(in_file)
    gen = vector_chunk_generator(in_file,
                                 chunk_size=batch_size,
                                 w2v_model=w2v_model,
                                 pad=False)
    with gzip.open(out_file, "ab") as out_fp:
        for n, chunk in enumerate(gen, 1):  # type: pd.DataFrame
            # print(f"Chunk {n}")
            for line in chunk.index:
                line = np.array([chunk.loc[line, "vectors"],
                                chunk.loc[line, "overall"]])
                pickle.dump(line, out_fp)

def _test():
    file = "../data/reviews_Musical_Instruments_5.json.gz"
    out_file = "../data/reviews_Musical_Instruments_5_vectors.pkl"
    batch_size = 20
    w2v = Word2Vec.load("../models/w2v_movies_electr.model")
    # print("Creating vectors")
    # if os.path.exists(out_file):
    #     os.remove(out_file)
    # create_vectors(in_file=file, out_file=out_file,
    #                batch_size=batch_size, w2v_model=w2v)
    # print("Done!")
    print("Testing")
    first = None
    for n, (x, y) in enumerate(pickle_vectors_generator(pickle_file=out_file,
                                                        batch_size=1,
                                                        from_line=20, to_line=10000)):
        assert n < 10261 - 20 - 261 + 1, f"n: {n}"
        # assert (len(y) == len(x) == batch_size + 5) or (len(y) == len(x) == 410 % batch_size + 1), \
        #     f"[{n}] Chunk len check failed! Expected {batch_size + 5}, found ({len(x)}, {len(y)})"
        if first is None:
            print(f"X: {x}")
            print(f"y: {y}")
            first = 1
        assert len(x.shape) == 3, \
            f"[{n}] Vector shape check failed! Expected 3, found {x.shape}."


if __name__ == '__main__':
    for fname in data_files:
        print(f"Vectorizing {fname}")
        out_file = fname.split(".json.gz")[0] + "_vec.pkl.gz"
        w2v = Word2Vec.load("../models/w2v_5dom.model")
        create_vectors(in_file=fname,
                       out_file=out_file,
                       w2v_model=w2v,
                       batch_size=20)
