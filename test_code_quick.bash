python train_model.py /physionet_data/challenge/files/cross-validation-data-1-0-3-mini/0/train model
python run_model.py model /physionet_data/challenge/files/cross-validation-data-1-0-3-mini/0/test test_outputs
python evaluate_model.py /physionet_data/challenge/files/cross-validation-data-1-0-3-mini/0/test test_outputs/