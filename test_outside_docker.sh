python train_model.py /physionet_data/challenge/files/cross-validation-data-1-0-3-example/0/train /physionet_data/challenge/model
python run_model.py model /physionet_data/challenge/files/cross-validation-data-1-0-3-example/0/test /physionet_data/challenge/test_outputs
python evaluate_model.py /physionet_data/challenge/files/cross-validation-data-1-0-3-example/0/test /physionet_data/challenge/test_outputs /physionet_data/challenge/model