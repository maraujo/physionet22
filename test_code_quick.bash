python train_model.py /physionet_data/challenge/example/training_data model
python run_model.py model /physionet_data/challenge/example/test_data/
python evaluate_model.py ../../../challenge/example/test_data/ test_outputs/