# coding: utf-8

# In[1]:

# get_ipython().magic('load_ext autoreload')
# get_ipython().magic('autoreload 2')


# In[2]:

import pandas as pd
import numpy as np
from IPython.display import display, clear_output
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, accuracy_score

from utils import load_vectors, rnd_labeled_data_generator, test_data_generator, conseq_labeled_data_generator

# Загружаем векторы

# In[3]:

vectors = {
    "auto": "../data/auto_vectors.csv",
    "mus": "../data/mus_vectors.csv"
}

# In[4]:
print("Loading vectors")
manually = True
auto_df = load_vectors(vectors["auto"], manually=manually, verbose=False)
if not manually:
    auto_df["vectors"].apply(lambda a: np.array(eval(a)))

# In[8]:

auto_df.groupby("overall").count()

# Классификация с 5 категориями, не бинарная

# In[8]:

auto_df["target"] = pd.get_dummies(auto_df["overall"]).values.tolist()
auto_df["target"] = auto_df["target"].apply(np.array)

# In[9]:

auto_df.head()

# Если делать zero-padding, очень быстро возникает MemoryError. Лучше учить по одному сэмплу

# In[5]:
#
#
# # In[136]:
#
#
#
# # In[137]:
#
# y_test = np.dstack(y_test)[0].T
# y_train = np.dstack(y_train)[0].T


# In[33]:
# # In[300]:
#
# # batch_size = 100
# # timesteps = timesteps_padding(auto_df["vectors"].values)
# # print("Timesteps:", timesteps)
# # print("Batch size:", batch_size)
# hidden_size1 = 80
# hidden_size2 = 32
#
# model = Sequential()
# model.add(LSTM(hidden_size1, return_sequences=True, input_shape=(None, 128)))
# # model.add(LSTM(hidden_size2, return_sequences=False))
# model.add(Dense(5, activation="softmax"))
# # model.add(Dense(num_classes, activation='softmax'))
#
# model.compile(loss='mean_absolute_error',
#               optimizer='adagrad', metrics=["accuracy"])
#
# model.fit_generator(rnd_labeled_data_generator(X_train, y_train),
#                     validation_data=rnd_labeled_data_generator(X_test, y_test),
#                     steps_per_epoch=30, epochs=150, verbose=1, validation_steps=100)
#
#
# # In[220]:
#
# def predict(X, model):
#     predictions = []
#     for i, sample in enumerate(X):
#         clear_output(True)
#         display("Retrieving {}/{}".format(i + 1, X.shape[0]))
#         predictions.append(model.predict(np.reshape(sample, [1, *sample.shape]), steps=1).reshape(-1, 5))
# #     predictions = np.array(predictions)
# #     predictions = np.reshape(predictions, (predictions.shape[0]))
#     return predictions
#
#
#
#
# # In[276]:
#
# def result_to_one_hot(prediction):
#     # Усреднение ответов для каждого слова внутри текста, потом выбор максимума
# #     max_i = np.argmax(np.mean(prediction, axis=0).reshape(-1))
# #     res = np.zeros(5)
# #     res[max_i] = 1
# #     return res.astype(int)
#     # Выбор оценки исходя из самого значимого слова
# #     max_i = np.argmax(np.max(prediction, axis=0))
# #     res = np.zeros(5)
# #     res[max_i] = 1
# #     return res.astype(int)
#     # Выбор по количеству максимальных
#     # значений среди слов в пределах рубрики
#     max_i = np.argmax(np.argmax(prediction, axis=0))
#     res = np.zeros(5)
#     res[max_i] = 1
#     return res.astype(int)
#
#
# def decode_one_hot(oh_matrix):
#     return np.where(oh_matrix == 1)[1] + 1
#
#
# # In[302]:
#
# res = predict(X_test, model)
#
#
# # In[303]:
#
# oh_res = np.array(list(map(result_to_one_hot, res)))
#
#
# # In[304]:
#
# print("Balanced accuracy:\t\t{}".format(balanced_accuracy_score(decode_one_hot(y_test),
#                                                                          decode_one_hot(oh_res), adjusted=False)))
# print("Balanced and adjusted accuracy:\t{}".format(balanced_accuracy_score(decode_one_hot(y_test),
#                                                                          decode_one_hot(oh_res), adjusted=True)))
# print("Unbalanced accuracy:\t\t{}".format(accuracy_score(decode_one_hot(y_test),
#                                                          decode_one_hot(oh_res))))
#
#
# # In[298]:
#
# decode_one_hot(y_test[:10])
#
#
# # In[299]:
#
# decode_one_hot(oh_res[:10])
#
#
# # In[294]:
#
# oh_res[:10]


# При классификации по 5 рубрикам качество остается низким при различных конфигурациях модели.
# Попробуем перейти к бинарной классификации.

# In[9]:

auto_df["target_bin"] = (auto_df["overall"] > 3).astype(int)
auto_df.head()

# Классы очень несбалансированны:

# In[10]:

auto_df.groupby("target_bin").count()

# In[11]:

ind_to_drop = np.random.choice(auto_df[auto_df["target_bin"] == 1].index,
                               size=(auto_df.shape[0] - 2 * auto_df[auto_df["target_bin"] == 0].shape[0]),
                               replace=False)

# In[12]:

all(auto_df.loc[ind_to_drop, "target_bin"])

# # In[13]:
#
# ind_to_drop.shape, auto_df.shape
#
#
# # In[14]:
#
# np.unique(auto_df.index).shape


# In[15]:

balanced_df = auto_df.drop(ind_to_drop, axis=0)
balanced_df.groupby("target_bin").count()

# In[16]:

X_train, X_test, y_train, y_test = train_test_split(balanced_df["vectors"].values,
                                                    balanced_df["target_bin"].values, test_size=0.3)


# In[17]:

def predict_bin(X, model):
    predictions = []
    for i, sample in enumerate(X):
        # clear_output(True)
        # display("Retrieving {}/{}".format(i + 1, X.shape[0]))
        predictions.append(model.predict(np.reshape(sample, [1, *sample.shape]), steps=1).reshape(-1))
    return predictions


def res_to_bin(y):
    return np.array(list(map(lambda a: np.round(np.mean(a)).astype(int), y)))


# In[18]:

hidden_size1 = 32
hidden_size2 = 150

model = Sequential()
model.add(LSTM(hidden_size1, return_sequences=True, input_shape=(None, 128)))
# model.add(LSTM(hidden_size2, return_sequences=True))
# model.add(Dense(hidden_size2, activation="hard_sigmoid"))
model.add(Dense(1, activation='hard_sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adagrad', metrics=["accuracy"])

model.fit_generator(rnd_labeled_data_generator(X_train, y_train),
                    validation_data=rnd_labeled_data_generator(X_test, y_test),
                    steps_per_epoch=1500, epochs=2, verbose=1, validation_steps=100)

res = predict_bin(X_test, model)
bin_res = res_to_bin(res)

np.bincount(bin_res)

print("Balanced accuracy:\t\t{}".format(balanced_accuracy_score(y_test, bin_res, adjusted=False)))
print("Balanced and adjusted accuracy:\t{}".format(balanced_accuracy_score(y_test, bin_res, adjusted=True)))
print("Unbalanced accuracy:\t\t{}".format(accuracy_score(y_test, bin_res, )))

model.save("classifier.hdf5")
