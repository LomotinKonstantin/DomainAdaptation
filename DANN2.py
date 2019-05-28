import numpy as np

from all_models_source_target import *
from utils2 import train_data_generator


def train_dom_disc(dom_disc):
    source_gen = train_data_generator(files=[source_domain],
                                      chunk_size=batch_size,
                                      w2v_model=w2v_model,
                                      line_counts=line_counts,
                                      autoencoder=False,
                                      test_percent=test_percent)
    target_gen = train_data_generator(files=target_domains,
                                      chunk_size=batch_size,
                                      w2v_model=w2v_model,
                                      line_counts=line_counts,
                                      autoencoder=False,
                                      test_percent=test_percent)
    for i in range(epochs):
        print(f"DANN dom_disc training epoch #{i}")
        for j in range(line_counts[source_domain] // batch_size):
            X_s, y_s = next(source_gen)
            # Source domain кодируем 0
            domain_s = np.zeros(y_s.shape)
            X_t, y_t = next(target_gen)
            # Target domain кодируем 1
            domain_t = np.full(y_t.shape, fill_value=1)
            print(f"Fitting domain discriminator batch #{j}")
            dom_disc.fit(X_s, domain_s, epochs=1, verbose=1)
            dom_disc.fit(X_t, domain_t, epochs=1, verbose=1)


if __name__ == '__main__':
    timestamp = get_timestamp()
    data_folder = "../data/"
    report_folder = f"../reports/{timestamp}/"
    model_folder = f"../models/{timestamp}/"
    os.mkdir(report_folder)
    os.mkdir(model_folder)
    print(f"Source domain file: {source_domain}")
    print(f"Target domains file: {target_domains}")

    classifier, dom_discriminator, comb = create_DANN()
    print("Training domain discriminator on source and target")
    train_dom_disc(dom_discriminator)
    clear_path = model_folder + "DANN_class_clear_with_FE.hdf5"
    classifier.save(clear_path)
    print("Training DANN classifiers on source")
    train_on_source(classifier, False)
    classifier.save(model_folder + "DANN_classifier_source.hdf5")
    print("Testing DANN on source")
    test_on_source(classifier, report_folder + "DANN_source-source.csv")
    print("Testing DANN on target")
    test_on_target(classifier, report_folder + "DANN_source-target.csv")
    print("Training and testing DANN on target")
    train_and_test_on_target(clear_model=clear_path, ae=False, model_name="DANN",
                             report_folder=report_folder, model_folder=model_folder)
