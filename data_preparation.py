import urllib.request
import os
import numpy as np
import json
from collections import OrderedDict
from nltk.corpus import stopwords
from nltk import word_tokenize
from string import punctuation
import re
import pandas as pd
from gensim.models import Word2Vec
import gzip


def load_data():
    urllib.request.urlretrieve("http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
                               "/reviews_Movies_and_TV_5.json.gz",
                               "../data/Movies_and_TV_5.json.gz")
    urllib.request.urlretrieve("http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5"
                               ".json.gz",
                               "../data/Electronics_5.json.gz")


# ## Датасеты
# <a href=http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Musical_Instruments_5.json.gz>Датасет с отзывами на музыкальные инструменты</a><br>
# <a href=http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Automotive_5.json.gz>Датасет с отзывами на автомобили</a><br>
# <a href=http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz>Датасет с отзывами на фильмы</a><br>
# <a href=http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz>Датасет с отзывами на электронику</a><br>
# 
# Чтобы все работало без редактирования, данные должны лежать в распакованном виде в ../data/


def load_json(path: str, columns=["helpful", "reviewText", "summary", "overall"]) -> pd.DataFrame:
    """
    Loads Amazon Review Dataset from the specified json file.
    Only includes fields, passed in 'columns' argument.
    Returns: DataFrame
    """
    if not os.path.exists(path):
        raise FileExistsError("'%s' does not exists!" % path)
    with open(path) as fp:
        lines = fp.readlines()
    lines = list(map(lambda s: json.loads(s, object_pairs_hook=OrderedDict), lines))
    return pd.DataFrame(lines).loc[:, columns]


def raw_data_generator(path: str, columns=["helpful", "reviewText", "summary", "overall"]) -> OrderedDict:
    if not os.path.exists(path):
        raise FileExistsError("'%s' does not exist!" % path)
    with gzip.open(path) as fp:
        for line in fp:
            parsed = json.loads(line, object_pairs_hook=OrderedDict)
            yield {key: parsed[key] for key in columns}


def batch_generator(path: str, columns=["helpful", "reviewText", "summary", "overall"], max_batch_size=500):
    lines = []
    for obj in raw_data_generator(path, columns):
        lines.append(obj)
        if len(lines) == max_batch_size:
            yield pd.DataFrame(lines)
            lines = []
    yield pd.DataFrame(lines)


def review_len(data: pd.DataFrame) -> np.array:
    """
    Returns the list of reviews' lengths
    """
    return np.array(list(map(lambda a: len(a.split()), data["reviewText"])))


# ## Предобработка
# * Перевод в нижний регистр
# * Удаление ненужных символов
# * Удаление стоп-слов
# * Удаление токенов короче 3 символов
# 
# Без лемматизации и стемминга


class Preprocessor:

    def __init__(self):
        self.stopwords = stopwords.words("english")
        self.to_drop = self.stopwords + list(punctuation) + list("0123456789“")

    def preprocess(self, text: str) -> str:
        # Lowercase conversion
        new_text = text.lower()
        new_text = re.sub("\.(?=\S)", " . ", new_text)
        tokens = word_tokenize(new_text)
        tokens = list(filter(lambda a: a not in self.to_drop and len(a) >= 3, tokens))
        return " ".join(tokens)


def test_pp():
    test_str = "One way would be to split the document into words by white space     " \
               "(as in “2. Split by Whitespace“), then use string translation to replace all    " \
               " punctuation with nothing (e.g. remove it)."
    p = Preprocessor()
    return p.preprocess(test_str)


# test_pp()


# In[5]:

# Preprocessing the dataframe
def preprocess_df(df: pd.DataFrame, merge_summary=False) -> pd.DataFrame:
    """
    If 'merge_summary' is True, merges 'Summary' column into 'reviewText'
    """
    p = Preprocessor()
    reviews = list(map(p.preprocess, df["reviewText"]))
    summaries = list(map(p.preprocess, df["summary"]))
    new_df = df.copy()
    new_df["reviewText"] = reviews
    new_df["summary"] = summaries
    if merge_summary:
        new_df["reviewText"] += " " + new_df["summary"]
    return new_df


# Создание векторной модели
# Для word embedding используется word2vec

# In[6]:


def train_w2v_model(dfs: list) -> Word2Vec:
    data = []
    for df in dfs:
        data += list(map(str.split, df["reviewText"].values))
    model = Word2Vec(sentences=data, size=128, min_count=3)
    model.train(data, total_examples=len(data), epochs=50)
    return model


# Сохраним векторы, чтобы в будущем работать только с ними
def create_vectors(df: pd.DataFrame, model: Word2Vec) -> pd.DataFrame:
    res = {
        "vectors": [],
        "overall": [],
    }
    for i in df.index:
        tokens = df.loc[i, "reviewText"].split()
        vec_lst = []
        for word in tokens:
                vec_lst.append(list(model.wv.get_vector(word)))
        res["vectors"].append(vec_lst)
        res["overall"].append(df.loc[i, "overall"])
    # Also adding the target variable
    return pd.DataFrame(res)


# Датасеты Movies и Electronics
# Для полной загрузки этих таблиц не хватает памяти, поэтому обработка построчная

# Нужно два раза пройтись по текстам: первый раз, чтобы обучить модель w2v, а второй - чтобы создать векторы

# In[8]:

def create_vectors_from_file(pp_file: str,
                             vector_file: str,
                             w2v_file: str,
                             batch_size=1000):
    w2v = Word2Vec.load(w2v_file)
    vector_fp = open(vector_file, "w")
    header = True
    for num, batch in enumerate(pd.read_csv(pp_file, sep="\t", chunksize=batch_size)):
        # clear_output(True)
        # display("Creating vectors")
        # display("Processing batch {}".format(num + 1))
        vector_batch = create_vectors(batch, w2v)
        vector_batch.to_csv(vector_fp, header=header, sep="\t", index=False)
        header = False


def process_file(raw_data_file: str,
                 pp_backup_file: str,
                 w2v_model_file: str,
                 create_w2v=False,
                 columns=["reviewText", "summary", "overall"],
                 batch_size=1000,
                 vector_size=128, ):
    data_gen = batch_generator(raw_data_file, columns, batch_size)
    if create_w2v:
        w2v = Word2Vec(size=vector_size, min_count=3)
    else:
        w2v = Word2Vec.load(w2v_model_file)
    header = True
    pp_file = open(pp_backup_file, "w")
    # Первый проход
    for num, batch in enumerate(data_gen):
        # Предобработка
        # clear_output(True)
        # display("Preprocessing and w2v training")
        print("Processing batch {}".format(num + 1))
        pp_batch = preprocess_df(batch, merge_summary=True)
        pp_batch.to_csv(pp_file, header=header, sep="\t", index=False)
        if header and create_w2v:
            # Признак первого батча 
            header = False
            w2v.build_vocab(pp_batch["reviewText"].values, update=False)
            w2v.train(pp_batch["reviewText"].values, total_examples=len(pp_batch["reviewText"]), epochs=20)
        # Обучение модели w2v
        w2v.build_vocab(pp_batch["reviewText"].values, update=True)
        w2v.train(pp_batch["reviewText"].values, total_examples=len(pp_batch["reviewText"]), epochs=20)
    w2v.save(w2v_model_file)


def balance_batch(df: pd.DataFrame):
    max_target = 0
    if df[df["target_bin"] == 1].shape[0] > df[df["target_bin"] == 0].shape[0]:
        max_target = 1
    ind_to_drop = np.random.choice(df[df["target_bin"] == max_target].index,
                                   size=(df.shape[0] - 2 * df[df["target_bin"] == 1 - max_target].shape[0]),
                                   replace=False)
    df.drop(ind_to_drop, axis=0, inplace=True)


def process_vector_batch(batch: pd.DataFrame):
    # To binary
    batch["target_bin"] = (batch["overall"] > 3).astype(int)
    #     batch["vectors"] = list(map(eval, batch["vectors"]))
    # Balancing
    balance_batch(batch)


def batch_vector_generator(pp_file: str,
                           w2v_file: str,
                           batch_size=1000):
    w2v = Word2Vec.load(w2v_file)
    for num, batch in enumerate(pd.read_csv(pp_file, sep="\t", chunksize=batch_size)):
        vector_batch = create_vectors(batch, w2v)
        vector_batch.drop(index=vector_batch[vector_batch["overall"] == "overall"].index,
                          inplace=True)
        vector_batch["overall"] = vector_batch["overall"].apply(float)
        yield vector_batch


def balanced_to_file(inp_file: str, output_file: str, batch_size: int, w2v_file: str):
    first = True
    with open(output_file, "w") as out_file:
        for num, batch in enumerate(batch_vector_generator(pp_file=inp_file,
                                                           batch_size=batch_size,
                                                           w2v_file=w2v_file)):
            print("Batch", num + 1)
            process_vector_batch(batch)
            len_0 = batch[batch["target_bin"] == 0].shape[0]
            len_1 = batch[batch["target_bin"] == 1].shape[0]
            if len_0 != len_1:
                print("Unbalanced!\n0: {}\n1: {}".format(len_0, len_1))
            batch.to_csv(out_file, header=first, index=False, sep="\t")
            first = False


if __name__ == '__main__':
    log = open("preparation.log", "w")
    try:

        print("Started")
        data_path = {
            "music": os.path.join("..", "data", "Musical_Instruments_5.json"),
            "auto": os.path.join("..", "data", "Automotive_5.json"),
            "movies": os.path.join("..", "data", "Movies_and_TV_5.json.gz"),
            "electr": os.path.join("..", "data", "Electronics_5.json.gz"),
        }
        balanced_output = {
            "movies": "../data/movies_vectors_balanced.csv",    # Source domain
            "electr": "../data/electr_vectors_balanced.csv",    # Target domain
        }
        columns = ["reviewText", "summary", "overall"]
        batch_size = 5000
        vector_size = 128
        w2v_file = "../models/w2v_movies_electr.model"

        electr_pp_file = os.path.join("..", "data", "electr_pp.csv")
        movies_pp_file = os.path.join("..", "data", "movies_pp.csv")

        electr_vec_file = os.path.join("..", "data", "electr_vectors.csv")
        movies_vec_file = os.path.join("..", "data", "electr_vectors.csv")

        # Загрузка данных
        # load_data()

        # Предобработка и обучение w2v
        # print("Preprocessing electronics")
        # process_file(raw_data_file=data_path["electr"],
        #              pp_backup_file=electr_pp_file,
        #              w2v_model_file=w2v_file,
        #              create_w2v=True,
        #              columns=columns,
        #              batch_size=batch_size,
        #              vector_size=vector_size)
        # print("Preprocessing movies")
        # process_file(raw_data_file=data_path["movies"],
        #              pp_backup_file=movies_pp_file,
        #              w2v_model_file=w2v_file,
        #              create_w2v=False,
        #              columns=columns,
        #              batch_size=batch_size,
        #              vector_size=vector_size, )

        # create_vectors_from_file(pp_file=electr_pp_file,
        #                          vector_file=electr_vec_file,
        #                          w2v_file=w2v_file,
        #                          batch_size=5000)
        #
        # create_vectors_from_file(pp_file=movies_pp_file,
        #                          vector_file=movies_vec_file,
        #                          w2v_file=w2v_file,
        #                          batch_size=5000)
        print("Creating and balancing electronics vectors", file=log)
        balanced_to_file(inp_file=electr_pp_file,
                         output_file=balanced_output["electr"],
                         batch_size=15000, w2v_file=w2v_file)
        print("Creating and balancing movies vectors", file=log)
        balanced_to_file(inp_file=movies_pp_file,
                         output_file=balanced_output["movies"],
                         batch_size=15000, w2v_file=w2v_file)
    except Exception as e:
        raise e
    finally:
        log.close()
