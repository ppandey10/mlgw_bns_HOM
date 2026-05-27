from src.model import *
from src.hyperparameter_optimization import HyperparameterOptimization

ind_mode = Mode(2,1)
def hyper_opt_fun():
    main_model = ModesModel(
        filename = "default_5hz", 
        modes=[ind_mode]
    )
    main_model.load()
    mmodel = main_model.models[ind_mode]
    ho = HyperparameterOptimization(mmodel, n_train_fixed=10000)
    ho.optimize_and_save(4.0)
    print("done optimizing!")
    print(f"best hyperparameters: {ho.best_hyperparameters()}")
    print("starting training with best hyperparameters...")
    mmodel.set_hyper_and_train_nn(ho.best_hyperparameters())
    mmodel.save(include_training_data=False)
    print("done training!")
if __name__ == "__main__":
    hyper_opt_fun()
