from src.model import *
from src.downsampling_interpolation import RDPDownsamplingTraining

def modes():
    modes_model = ModesModel(
        filename="default_5hz",
        modes=[Mode(3,3)],
        initial_frequency_hz=5.0,
        # parameter_ranges=ParameterRanges(q_range=(1.35, 3.0)),
        # downsampling_training=RDPDownsamplingTraining,
    )
    print(modes_model.modes)
    modes_model.generate(
        training_downsampling_dataset_size=2**11, # 2048 = 2^11
        training_pca_dataset_size=2**14,          # 16384 = 2^14
        training_nn_dataset_size=2**16          # 65536 = 2^16
    )
    modes_model.set_hyper_and_train_nn()
    modes_model.save(include_training_data=True)
    print("done generating model!")

if __name__ == "__main__":
    modes()
    print("done!")
