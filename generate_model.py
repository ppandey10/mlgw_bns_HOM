from src.model import *
import matplotlib.pyplot as plt


def modes():
    modes_model = ModesModel(
        filename = "default_16hz",
        modes=[Mode(4,4)], 
        initial_frequency_hz=16.,
    )
    print(modes_model.modes)
    modes_model.generate(
        training_downsampling_dataset_size=2 ** 10, # 1024
        training_pca_dataset_size=2 ** 14,          # 16384
        training_nn_dataset_size=2 ** 16            # 65536
    )
    modes_model.set_hyper_and_train_nn()
    modes_model.save(include_training_data=True)

if __name__ == "__main__":
    modes()
    print("done!")
