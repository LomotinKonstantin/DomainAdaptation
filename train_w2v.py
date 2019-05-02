from gensim.models import Word2Vec
import pandas as pd

from utils2 import pp_chunk_generator


def sentence_generator(files: list, batch_size: int) -> pd.DataFrame:
    for fn in files:
        print(f"Training on {fn}")
        gen = pp_chunk_generator(fn, batch_size)
        for num, batch in gen:
            sentence = [text.split() for text in batch["reviewText"].values]
            yield sentence


def create_w2v_model(files: list, batch_size=1000, vector_size=128) -> Word2Vec:
    w2v = Word2Vec(size=vector_size, min_count=3)
    init = False
    for sentences in sentence_generator(files, batch_size):
        w2v.build_vocab(sentences, update=init)
        w2v.train(sentences, epochs=20, total_examples=len(sentences))
        init = True
    return w2v


if __name__ == '__main__':
    files = [
        "Apps_for_Android_5.json.gz",
        "CDs_and_Vinyl_5.json.gz",
        "Kindle_Store_5.json.gz",
        "Electronics_5.json.gz",
        "Movies_and_TV_5.json.gz"""
    ]
    for i in range(len(files)):
        files[i] = "../data/" + files[i]    # type: list
    model = create_w2v_model(files)
    model.save("../models/w2v_5dom_v1.model")
