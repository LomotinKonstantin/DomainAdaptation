
# coding: utf-8

# In[1]:

import tensorflow as tf
import pandas as pd
import numpy as np
from IPython.display import display, clear_output

from keras.models import Sequential
from keras.layers import LSTM, Dense

from utils import load_vectors, ae_data_generator, rnd_labeled_data_generator


# Загружаем векторы

# In[2]:

vectors = {
    "mus": "../data/mus_vectors.csv",
    "auto": "../data/auto_vectors.csv"
}


# Отзывы об автомобилях примем за исходный домен, а отзывы о музыкальных инструментах - за целевой 

# In[3]:
print("Loading mus vectors")
mus_df = load_vectors(vectors["mus"], manually=False)


# In[4]:

# Кернель умирает при загрузке большого файла,
# Поэтому грузим вручную
print("Loading auto vectors")
auto_df = load_vectors(vectors["auto"], manually=True, verbose=False)


# Преобразуем векторы из строк в списки. По какой-то причине Series.apply наглухо стопорит кернель

# In[5]:
print("Turning str values to lists")
for i in range(mus_df.shape[0]):
    # clear_output(True)
    # display("{} / {}".format(i+1, mus_df.shape[0]))
    mus_df.at[i, "vectors"] = eval(mus_df.loc[i, "vectors"])


# Переходим к бинарной классификации и балансируем датасеты

# In[6]:

def balanced(df):
    ind_to_drop = np.random.choice(df[df["target_bin"] == 1].index, 
                                   size=(df.shape[0] - 2*df[df["target_bin"] == 0].shape[0]), 
                                   replace=False)
    return df.drop(ind_to_drop, axis=0)


# In[7]:
print("Binarization & balancing")
mus_df["target_bin"] = (mus_df["overall"] > 3).astype(int)
mus_df.head()


# In[8]:

mus_df.groupby("target_bin").count()


# In[9]:

balanced_mus_df = balanced(mus_df)
balanced_mus_df.groupby("target_bin").count()


# In[10]:

auto_df["target_bin"] = (auto_df["overall"] > 3).astype(int)
auto_df.head()


# In[11]:

balanced_auto_df = balanced(auto_df)
balanced_auto_df.groupby("target_bin").count()


# Обучаем LSTM-автоэнкодер на данных из двух доменов

# In[12]:

X_train = pd.concat([auto_df["vectors"], mus_df["vectors"]]).values


# # In[13]:
#
# X_train.shape


# In[14]:

# Converting to ndarray
for i in range(X_train.shape[0]):
    # clear_output(True)
    # display("{} / {}".format(i+1, X_train.shape[0]))
    X_train[i] = np.array([np.array(vec) for vec in X_train[i]])


# In[15]:

np.random.shuffle(X_train)


# In[ ]:

train_percent = 0.7


# Используем pretraining. Из-за того, что размер последовательностей не фиксирован, а батч для обучения должен быть
# тензором, приходится обучать модель по одному сэмплу за раз. Zero-padding приводит к MemoryError даже на маленьком
# сбалансированном датасете.

# In[ ]:


data_dim = 128
num_classes = 2
latent_space_dim = 32

print("Training AE layers")
# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(latent_space_dim, return_sequences=True,
               input_shape=(None, data_dim)))
model.add(LSTM(128, return_sequences=True))
# model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='mean_squared_error',
              optimizer='adagrad')
model.fit_generator(ae_data_generator(X_train), steps_per_epoch=3000, epochs=10, verbose=0)


# На всякий случай сохраняем модель

# In[ ]:

model.save("./lstm_v2.hdf5")


# Удаляем последний слой

# In[ ]:
#
# model.layers.pop()
# model.layers


# In[ ]:

X_train, X_test, y_train, y_test = train_test_split(balanced_auto_df["vectors"].values, 
                                                    balanced_auto_df["target_bin"].values, test_size=0.3)


# Добавляем классификатор со структурой, полученной в LSTM_classifier.ipynb

# In[ ]:

hidden_size1 = 32
hidden_size2 = 150

model.add(LSTM(hidden_size1, return_sequences=True, input_shape=(None, 128)))
# model.add(LSTM(hidden_size2, return_sequences=True))
# model.add(Dense(hidden_size2, activation="hard_sigmoid"))
model.add(Dense(1, activation='hard_sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adagrad', metrics=["accuracy"])

model.fit_generator(rnd_labeled_data_generator(X_train, y_train), 
                    validation_data=rnd_labeled_data_generator(X_test, y_test), 
                    steps_per_epoch=1500, epochs=2, verbose=1, validation_steps=100)


# Таким образом, точность предобученной модели на исходном домене составила 0.88
# Теперь проведем валидацию модели на объектах из целевого домена

# In[ ]:

def predict(X, model):
    predictions = []
    for i, sample in enumerate(X):
        # clear_output(True)
        # display("Retrieving {}/{}".format(i + 1, X.shape[0]))
        predictions.append(model.predict(np.reshape(sample, [1, *sample.shape]), steps=1).reshape(-1))
    return predictions


def res_to_bin(y):
    return np.array(list(map(lambda a: np.round(np.mean(a)).astype(int), y)))


def accuracy(bin_res, y_test):
    print("Balanced accuracy:\t\t{}".format(balanced_accuracy_score(y_test, bin_res, adjusted=False)))
    print("Balanced and adjusted accuracy:\t{}".format(balanced_accuracy_score(y_test, bin_res, adjusted=True)))
    print("Unbalanced accuracy:\t\t{}".format(accuracy_score(y_test, bin_res)))


def validate(X_test, y_test, model):
    res = predict_bin(X_test, model)
    bin_res = res_to_bin(res)
    accuracy(bin_res, y_test)


# In[ ]:

print("Source domain test accuracy:")
validate(X_test, y_test, model)
print("\nTarget domain test accuracy:")
validate(balanced_mus_df["vectors"].values, balanced_mus_df["target_bin"].values, model)


# Сохраняем модель

# In[ ]:

model.save("./lstm_dense_v2.hdf5")


# Точность достигла хороших значений для обоих доменов.
# > Source domain test accuracy: 0.8351154837766662<br>
# > Target domain test accuracy: 0.8674490487938746 
# 
