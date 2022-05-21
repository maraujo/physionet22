#Quick train/evaluation mode (not all the data)
python train_model.py training_data model
python run_model.py model test_data test_outputs
python evaluate_model.py test_data test_outputs model