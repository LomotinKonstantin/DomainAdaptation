from all_models_source_target import *


if __name__ == '__main__':
    timestamp = get_timestamp()
    data_folder = "../data/"
    report_folder = f"../reports/{timestamp}/"
    model_folder = f"../models/{timestamp}/"
    os.mkdir(report_folder)
    os.mkdir(model_folder)

    model = create_lstm_classifier()
    report_path = report_folder + "LSTM_source-source.csv"
    clear_path = model_folder + "LSTM_clear.hdf5"
    model.save(clear_path)
    print("Training LSTM")
    train_on_source(model, ae=False)
    model.save(model_folder + "LSTM_source.hdf5")
    print("Testing LSTM on source")
    test_on_source(model, report_path)
    report_path = report_folder + "LSTM_source-target_{}.csv"
    print("Testing LSTM on target")
    test_on_target(model, report_path)
    print("Training and testing LSTM on target")
    train_and_test_on_target(clear_model=clear_path, ae=False, model_name="LSTM",
                             report_folder=report_folder, model_folder=model_folder)
