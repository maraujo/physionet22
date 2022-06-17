rm -r test_outputs
# python train_model.py /physionet_data/challenge/files/cross-validation-data-1-0-3-mini/$1/train model
# python run_model.py model /physionet_data/challenge/files/cross-validation-data-1-0-3-mini/$1/test test_outputs
# python evaluate_model.py /physionet_data/challenge/files/cross-validation-data-1-0-3-mini/$1/test test_outputs/ model

python train_model.py /physionet_data/challenge/files/cross-validation-data-1-0-3/$1/train model
# python run_model.py model /physionet_data/challenge/files/cross-validation-data-1-0-3-mini/$1/test test_outputs
# python evaluate_model.py /physionet_data/challenge/files/cross-validation-data-1-0-3-mini/$1/test test_outputs/ model