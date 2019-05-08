import sys

from gensim.models import Word2Vec

from utils2 import pp_chunk_generator


def create_w2v_model(files: list, batch_size=1000, vector_size=128) -> Word2Vec:
    w2v = Word2Vec(size=vector_size, min_count=3)
    init = False
    for n, fn in enumerate(files, start=1):
        print(f"Training on {fn}")
        gen = pp_chunk_generator(fn, batch_size)
        for batch in gen:
            sentence = [text.split() for text in batch["reviewText"].values]
            w2v.build_vocab(sentence, update=init)
            w2v.train(sentence, epochs=20, total_examples=len(sentence))
            init = True
        w2v.save(f"../models/w2v_5dom_stage{n}.model")
    return w2v


if __name__ == '__main__':
    if len(sys.argv) == 1:
        files = [
            "Apps_for_Android_5.json.gz",
            "CDs_and_Vinyl_5.json.gz",
            "Kindle_Store_5.json.gz",
            "Electronics_5.json.gz",
            "Movies_and_TV_5.json.gz",
        ]
        for i in range(len(files)):
            files[i] = "../data/" + files[i]    # type: list
        model = create_w2v_model(files, batch_size=20)
        model.save("../models/w2v_5dom_v1.model")
    else:
        model_fp = sys.argv[1]
        train_file = sys.argv[2]
        out_file = sys.argv[3]
        w2v = Word2Vec.load(model_fp)
        gen = pp_chunk_generator(train_file, 50)
        for batch in gen:
            sentence = [text.split() for text in batch["reviewText"].values]
            w2v.build_vocab(sentence, update=True)
            w2v.train(sentence, epochs=20, total_examples=len(sentence))
        w2v.save(out_file)
