
curl http://algodev.matheusaraujo.com:8032/ohh.config --output ohh.config

python run_update_hyperparameters.py
python train_model.py /physionet_data/challenge/files/cross-validation-data-1-0-3/$1/train model
python run_model.py model /physionet_data/challenge/files/cross-validation-data-1-0-3/$1/test test_outputs
python evaluate_model.py /physionet_data/challenge/files/cross-validation-data-1-0-3/$1/test test_outputs/ model
python run_save_hyperparameters.py
