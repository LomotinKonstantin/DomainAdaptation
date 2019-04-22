from keras.models import load_model
from utils import append_timestamp, test_model


if __name__ == '__main__':
    report_folder = "../reports/"
    model_folder = "../models/"
    models = {
        "AE_classifier_29-Mar-2019__17-49-01.hdf5": {
            "report": "AE"
        },
        "classifier_DANN_final_07-Apr-2019__00-43-33.hdf5": {
            "report": "DANN"
        },
        "SDAE_classifier_06-Apr-2019__15-51-40.hdf5": {
            "report": "SDAE"
        }
    }
    data = {
        "source": "../data/test_movies_vectors_balanced.csv",
        "target": "../data/test_electr_vectors_balanced.csv"
    }
    for model_name in models:
        model_path = model_folder + model_name
        model = load_model(model_path)
        for k in data:
            report = "{}_test_{}_V2.csv".format(k, models[model_name]["report"])
            report_path = append_timestamp(report_folder + report)
            comment = "Testing {} on {} domain with new predict processing"
            comment = comment.format(model_name, k)
            test_model(model, data[k], report_path, 1000, comment=comment, norm=True)
