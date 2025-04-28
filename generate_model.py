from src.model import *
import matplotlib.pyplot as plt


def modes():
    modes_model = ModesModel(
        filename = "fast_train_2k",
        modes=[Mode(2,2)], 
        initial_frequency_hz=20.,
    )
    print(modes_model.modes)
    modes_model.generate(128, 2000, 2000)
    modes_model.set_hyper_and_train_nn()
    modes_model.save(include_training_data=True)
    print("done!")

if __name__ == "__main__":
    modes()