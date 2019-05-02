from collections import OrderedDict
import gzip
import json

import pandas as pd

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
            parsed_line = {key: value for key, value in parsed_line.items() if key in columns}
            chunk.append(parsed_line)
            if len(chunk) == chunk_size:
                yield pd.DataFrame(chunk)
                chunk = []
    if chunk:
        yield pd.DataFrame(chunk)


def pp_chunk_generator(path: str,
                       chunk_size: int,
                       from_line: int = 0,
                       to_line: int = 0) -> pd.DataFrame:
    preprocessor = Preprocessor()
    for chunk in raw_chunk_generator(path,
                                     chunk_size,
                                     from_line=from_line,
                                     to_line=to_line):
        chunk["vector"] = chunk["reviewText"].apply(preprocessor.preprocess)
        yield chunk


def text_to_matr(text: str):
    return text


if __name__ == '__main__':
    gen = pp_chunk_generator("../data/reviews_Musical_Instruments_5.json.gz",
                             chunk_size=10, to_line=10, from_line=100)
    df = next(gen)
    print(df)
