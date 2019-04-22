import urllib.request

urllib.request.urlretrieve("http://snap.stanford.edu/"
                           "data/amazon/productGraph/"
                           "categoryFiles/reviews_CDs_and_Vinyl_5.json.gz",
                           "../data/CDs_and_Vinyl_5.json.gz")
urllib.request.urlretrieve("http://snap.stanford.edu/"
                           "data/amazon/productGraph/"
                           "categoryFiles/reviews_Kindle_Store_5.json.gz",
                           "Kindle_Store_5.json.gz")
urllib.request.urlretrieve("http://snap.stanford.edu/"
                           "data/amazon/productGraph/"
                           "categoryFiles/reviews_Apps_for_Android_5.json.gz",
                           "../data/Apps_for_Android_5.json.gz")
