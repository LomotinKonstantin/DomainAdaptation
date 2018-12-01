import pandas as pd
import numpy as np
from IPython.display import display, clear_output


def load_vectors(path: str, manually=False) -> pd.DataFrame:
    if not manually:
        return pd.read_csv(path, sep="\t")
    else:
        first_line_skipped = False
        data = {
            "overall": [],
            "vectors": []
        }
        cntr = 1
        with open(path) as file:
            for line in file:
                clear_output(True)
                display("Reading line %d" % cntr)
                if not first_line_skipped:
                    first_line_skipped = True
                    continue
                score, vectors = line.split("\t")
                vector = eval(vectors)
                vector = np.array(list(map(np.array, vector)))
                #if type(vector) != type([]):
                #    raise ValueError("{} in line {}".format(vector, cntr))
                data["overall"].append(eval(score))
                data["vectors"].append(vector)
                cntr += 1
        return pd.DataFrame(data)


def ae_data_generator(X_train):
    while True:
        x = X_train[np.random.choice(np.arange(X_train.shape[0]))]
        x = np.array([x])
        # Попытка поймать баг
        if len(x.shape) != 3:
            print(x)
            yield x, x

            
def conseq_labeled_data_generator(X_array, y_array):
    index = 0
    while True:
        index = index % X_array.shape[0]
        x = X_array[index]
        x = np.reshape(x, (1, *x.shape))
        y = y_array[index]
        y = np.array([y])
        y = np.reshape(y, (1, *y.shape))
        yield x, y        

        
def rnd_labeled_data_generator(X_array, y_array):
    while True:
        index = np.random.choice(np.arange(X_array.shape[0]))
        x = X_array[index]
        x = np.reshape(x, (1, *x.shape))
        y = y_array[index]
        y = np.array([y])
        y = np.reshape(y, (1, *y.shape))
        yield x, y


def test_data_generator(X_test):
    for i, sample in enumerate(X_test):
        #answer = np.array(y_test[i])
        yield np.reshape(sample, (1, *sample.shape))#, np.reshape(answer, [1, *answer.shape])
 
