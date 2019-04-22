import sys
from keras.models import load_model
from utils import append_timestamp, test_model


if __name__ == '__main__':
    model_path = sys.argv[1]
    test_path = sys.argv[2]
    report_path = append_timestamp(sys.argv[3])
    comment = sys.argv[4]
    model = load_model(model_path)
    test_model(model, test_path, report_path, 3000, comment=comment)
