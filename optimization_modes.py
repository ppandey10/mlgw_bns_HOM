from src.model import *
from src.hyperparameter_optimization import HyperparameterOptimization

def hyper_opt_fun():
    main_model = ModesModel(
        filename = "default_16hz", 
        modes=[Mode(2,2), Mode(2,1), Mode(3,3), Mode(4,4)]
    )
    main_model.load()
    mmodel = main_model.models[Mode(2,1)]
    ho = HyperparameterOptimization(mmodel)
    ho.optimize_and_save(1.5)


if __name__ == "__main__":
    hyper_opt_fun()
    print("done!")
