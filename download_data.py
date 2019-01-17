import urllib.request

urllib.request.urlretrieve("http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz", 
                  "../data/Movies_and_TV_5.json.gz")
urllib.request.urlretrieve("http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz", 
                  "../data/Electronics_5.json.gz")
