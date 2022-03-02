#Install basics
yum install make glibc-devel gcc patch
#Install EPEL
sudo amazon-linux-extras install epel -y
#Enable EPEL
sudo yum-config-manager --enable epel
#Install packeges
sudo yum install hdf5
sudo yum install libsndfile-devel
pip install pandas==1.14.1
pip install ipdb==0.13.9
pip install xlrd==1.2.0
pip install openl3==0.4.1
pip install librosa==0.7.2
pip install numba==0.48
pip install pydub==0.25.1
pip install audiomentations==0.16.0
pip install tqdm==4.62.2
pip install ipywidgets==7.6.4
pip install loguru==0.5.3
pip install tensorflow==2.8.0