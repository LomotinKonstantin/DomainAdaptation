import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    test_ratio = float(sys.argv[1])
    files = sys.argv[2:]
    print("Started")
    print("Test ratio:", test_ratio)
    print("Files:")
    print("\t", "\n\t".join(files))
    for fname in files:
        if not os.path.exists(fname):
            print("File {} not found".format(fname))
            exit()
        fname_split = os.path.split(fname)[-1]
        train_path = os.path.join(os.path.dirname(fname),
                                  "train_" + fname_split)
        test_path = os.path.join(os.path.dirname(fname),
                                 "test_" + fname_split)
        with open(train_path, "w") as train_file, open(test_path, "w") as test_file:
            write_header = True
            for num, chunk in enumerate(pd.read_csv(fname, sep="\t")):
                print("Batch {} of file {}".format(num, fname_split))
                chunk_size = len(chunk.index)
                train_rows_num = int(chunk_size * (1 - test_ratio))
                train_chunk = chunk.iloc[:train_rows_num]
                train_chunk.to_csv(train_file,
                                   sep="\t",
                                   index=None,
                                   header=write_header)
                test_chunk = chunk.iloc[train_rows_num + 1:]
                test_chunk.to_csv(test_file,
                                  sep="\t",
                                  index=None,
                                  header=write_header)
                write_header = False
