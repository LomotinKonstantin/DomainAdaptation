from collections import OrderedDict
import gzip
import json

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from preprocessor import Preprocessor


def raw_chunk_generator(path: str,
                        chunk_size: int,
                        from_line: int = 0,
                        to_line: int = 0) -> pd.DataFrame:
    chunk = []
    columns = ("reviewText", "overall")
    with gzip.open(path) as fp:
        for line_num, json_line in enumerate(fp):
            if line_num < from_line:
                continue
            if to_line > 0 and line_num <= to_line:
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
        chunk["vectors"] = text_to_matr(
            chunk["reviewText"].apply(preprocessor.preprocess),
            w2v_model
        )
        yield chunk


def text_to_matr(text: str, w2v_model) -> np.array:
    tokens = text.split()
    vec_lst = []
    for word in tokens:
        try:
            vec_lst.append(np.array(w2v_model.wv.get_vector(word)))
        except KeyError:
            pass
    return np.array(vec_lst)


if __name__ == '__main__':
    w2v = Word2Vec.load("../models/w2v_5dom.model")
    gen = vector_chunk_generator("../data/reviews_Musical_Instruments_5.json.gz",
                                 chunk_size=10, to_line=10, from_line=100, w2v_model=w2v)
    df = next(gen)
    print(df)
