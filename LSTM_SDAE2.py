from all_models_source_target import *


if __name__ == '__main__':
    timestamp = get_timestamp()
    data_folder = "../data/"
    report_folder = f"../reports/{timestamp}/"
    model_folder = f"../models/{timestamp}/"
    os.mkdir(report_folder)
    os.mkdir(model_folder)

    report_path = report_folder + "SDAE_LSTM_test_report.csv"
    model = None
    for latent_space_dim in [128, 72, 64]:
        model = create_AE_model(latent_space_dim, 128, model)
        print(f"Training AE layers {latent_space_dim}")
        train_model(model, train_files=train_files, batch_size=batch_size,
                    epochs=epochs, ae=True, line_count_hint=line_counts,
                    test_percent=test_percent, w2v_model=w2v_model,
                    noise_decorator=gaussian_noise)
        model.layers.pop()
    model = create_lstm_classifier(model)
    model.save(model_folder + "SDAE_LSTM_clear.hdf5")
    print("Training SDAE+LSTM on source")
    train_on_source(model, False)
    model.save(model_folder + "SDAE_LSTM_source.hdf5")
    print("Testing SDAE+LSTM on source")
    test_on_source(model, report_folder + "SDAE_LSTM_source-source.csv")
    print("Testing SDAE+LSTM on target")
    test_on_target(model, report_folder + "SDAE_LSTM_source-target.csv")
    print("Training and testing SDAE+LSTM on target")
    model = load_model(model_folder + "SDAE_LSTM_clear.hdf5")
    train_and_test_on_target(model, ae=False, model_name="AE_LSTM",
                             report_folder=report_folder, model_folder=model_folder)
    model.save(model_folder + "SDAE_LSTM_target.hdf5")
