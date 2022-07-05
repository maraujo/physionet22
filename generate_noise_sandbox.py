import pandas as pd
import os 
import shutil
from tqdm import tqdm
import autokeras as ak
import sys
from team_code import generate_mel_wav_crops_v2
import glob
from multiprocessing import Pool


WORKERS = min(os.cpu_count() - 1, 8)

labels_df = pd.read_csv("project_noise_detection.csv")
# Remove unsure
labels_df = labels_df[~(labels_df["type"] == "Unsure")]
labels_df["has_noise"] = labels_df["type"] == "Has noise"

# Create a folder according to https://autokeras.com/tutorial/load/
ROOT_AUDIO = "/physionet_data/challenge/files/circor-heart-sound/1.0.3/training_data/"
ROOT_IMAGES = "/tmp/noise_images/"
if os.path.exists(ROOT_IMAGES):
    shutil.rmtree(ROOT_IMAGES)
os.makedirs(ROOT_IMAGES, exist_ok=True)
audio_files = glob.glob(ROOT_AUDIO + "*.wav")
destiny = [ROOT_IMAGES] * len(audio_files)
audio_destinys = zip(audio_files, destiny)
pool = Pool(processes=(min(WORKERS, 8)))
for _ in tqdm(pool.imap(generate_mel_wav_crops_v2, audio_destinys), total=len(audio_files)):
    pass
pool.close()

ROOT_FOLDER = "/physionet_data/challenge/files/noise_detection_sandbox/"
HAS_NOISE_FOLDER = ROOT_FOLDER + "positive/"
HAS_HEARTBEAT_FOLDER = ROOT_FOLDER + "negative/"
if not os.path.exists(HAS_NOISE_FOLDER):
    os.mkdir(HAS_NOISE_FOLDER)
else:
    shutil.rmtree(HAS_NOISE_FOLDER)
    os.mkdir(HAS_NOISE_FOLDER)
if not os.path.exists(HAS_HEARTBEAT_FOLDER):
    os.mkdir(HAS_HEARTBEAT_FOLDER)
else:
    shutil.rmtree(HAS_HEARTBEAT_FOLDER)
    os.mkdir(HAS_HEARTBEAT_FOLDER)
    
for row_index, row in tqdm(labels_df.iterrows()):
    image_path = ROOT_IMAGES + row["image"].split("/")[-1]
    if row["has_noise"]:
        shutil.copy2(image_path, HAS_NOISE_FOLDER)
    else:
        shutil.copy2(image_path, HAS_HEARTBEAT_FOLDER)
    print(row_index)
os.system("tar -cvzf noise_detection_sandbox.tar.gz {}".format(ROOT_FOLDER))
sys.exit(0)      
img_height = 360
img_width = 144
batch_size = 4    

train_data = ak.image_dataset_from_directory(
    ROOT_FOLDER,
    # Use 20% data as testing data.
    validation_split=0.2,
    subset="training",
    # Set seed to ensure the same split when loading testing data.
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

test_data = ak.image_dataset_from_directory(
    ROOT_FOLDER,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

clf = ak.ImageClassifier(overwrite=True, max_trials=100)
clf.fit(train_data)
print(clf.evaluate(test_data))
try:
    clf.save("model_autokeras_noise_detection.model", save_format="tf")
except Exception:
    clf.save("model_autokeras_noise_detection.h5")