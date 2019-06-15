from all_models_source_target import *


if __name__ == '__main__':
    model = load_model("../models/28-May-2019__03-06-08/LSTM_Apps_for_Android.hdf5")
    test_path = "../data/Apps_for_Android_5.json.gz"
    test_model(model=model,
               test_paths=[test_path],
               test_percent=0.1,
               line_count_hint=line_counts,
               w2v_model=w2v_model,
               batch_size=1000,
               report_path="../reports/debug_report.csv")
