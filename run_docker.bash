docker build -t image .
<<<<<<< HEAD
# docker run -v /physionet_data/challenge/example/model/:/physionet/model -v /physionet_data/challenge/example/test_data/:/physionet/test_data -v /physionet_data/challenge/example/test_outputs/:/physionet/test_outputs/ -v /physionet_data/challenge/example/training_data/:/physionet/training_data image bash ./test_in_docker.bash 
docker run -v ./models/:/physionet/model -v /home/zeng/data/train_val_split_v1/test/:/physionet/test_data -v /home/zeng/code/physionet22/test_output/:/physionet/test_outputs/ -v /home/zeng/data/train_val_split_v1/train/:/physionet/training_data image bash ./test_in_docker.bash 
=======
docker run -v /physionet_data/challenge/example/model/:/physionet/model -v /physionet_data/challenge/example/test_data/:/physionet/test_data -v /physionet_data/challenge/example/test_outputs/:/physionet/test_outputs/ -v /physionet_data/challenge/example/training_data/:/physionet/training_data image bash ./test_in_docker.bash 
# docker run -v /home/zeng/code/physionet22/models/:/physionet/model -v /home/zeng/data/train_val_split_v1/test/:/physionet/test_data -v /home/zeng/code/physionet22/test_output/:/physionet/test_outputs/ -v /home/zeng/data/train_val_split_v1/train/:/physionet/training_data image bash ./test_in_docker.bash 
>>>>>>> 43302e0 (submission 3 tentative 1)
