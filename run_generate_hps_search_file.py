from team_code import *
import random
import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text
import uuid
random.seed(42)

K = 10000

parameters_search_space = {
TRAIN_FRAC_lbl : [(0.7, 1)],
ACTIVATION_FUNCTION_lbl : [("leaky_relu", 0.5), ("relu", 0.5)],
EMBDS_PER_PATIENTS_lbl : [(4, 0.1), (16, 0.2), (64, 0.5), (128, 0.2) ],
NORMALIZE_OUTCOMES_lbl : [(True, 0.5), (False, 0.5)],
VAL_FRAC_MURMUR_lbl : [(0.2, 0.5), (0.3, 0.5)],
NOISE_IMAGE_SIZE_lbl : [(64, 0.3), (108, 0.4), (256,0.3)],
MURMUR_IMAGE_SIZE_lbl : [(64, 0.3), (108, 0.4), (256,0.3)],
RESHUFFLE_PATIENT_EMBS_N_lbl : [(1, 0.3), (4, 0.4), (16, 0.3)],
class_weight_murmur_lbl : [(1, 0.5), (2, 0.25), (4, 0.25)],
class_weight_decision_lbl : [(1, 0.5), (2, 0.25), (4, 0.25)],
class_weight_outcome_lbl : [(1, 0.5), (2, 0.25), (4, 0.25)],
batch_size_murmur_lbl : [(128, 0.5), (256, 0.25), (64, 0.25)],
REMOVE_NOISE_lbl : [(True, 0.5), (False, 0.5)],
EMBS_SIZE_lbl : [(1, 0.1), (2, 0.2), (4, 0.2), (8, 0.2),(16, 0.2), (32, 0.1)],
N_DECISION_LAYERS_lbl : [(1, 0.25), (2, 0.5), (3, 0.25)],
N_OUTCOME_LAYERS_lbl : [(1, 0.25), (2, 0.5), (3, 0.25)],
NEURONS_DECISION_lbl : [(4, 0.1), (8, 0.2), (32, 0.2), (128, 0.3),(256, 0.2)],
IS_DROPOUT_IN_DECISION_lbl : [(True, 0.5), (False, 0.5)],
DROPOUT_VALUE_IN_DECISION_lbl : [(0.1, 0.25), (0.25, 0.5), (0.5, 0.25)],
NEURONS_OUTCOME_lbl :  [(4, 0.1), (8, 0.2), (32, 0.2), (128, 0.3),(256, 0.2)],
DROPOUT_IN_OUTCOME_lbl : [(True, 0.5), (False, 0.5)],
DROPOUT_VALUE_IN_OUTCOME_lbl : [(0.1, 0.25), (0.25, 0.5), (0.5, 0.25)],
N_MURMUR_CNN_NEURONS_LAYERS_lbl : [(64, 0.3), (8, 0.1), (32, 0.2), (128, 0.2),(256, 0.2)],
DROPOUT_VALUE_IN_MURMUR_lbl : [(0.1, 0.25), (0.25, 0.5), (0.5, 0.25)],
DROPOUT_VALUE_IN_MURMUR_lbl : [(True, 0.5), (False, 0.5)],
N_MURMUR_LAYERS_lbl : [(1, 0.25), (2, 0.5), (3, 0.25)],
UNKOWN_RANDOM_MIN_THRESHOLD_lbl : [(0.5, 0.25), (0.8, 0.5), (1, 0.25)],
BATCH_SIZE_DECISION_lbl  : [(32, 0.25), (4, 0.5), (8, 0.25)],
IMG_HEIGHT_RATIO_lbl : [(1, 0.5), (0.8, 0.25), (0.5, 0.25)]
}

def generate_random_parameters():
    new_parameters = {}
    for key in parameters_search_space.keys():
        parameter_search_space = dict(parameters_search_space[key])
        parameter_choice = random.choices([*parameter_search_space.keys()], [*parameter_search_space.values()], k=1)
        new_parameters[key] = parameter_choice[0]
    new_parameters["parameter_id"] = str(uuid.uuid4())
    return new_parameters

parameter_search_df = pd.DataFrame([generate_random_parameters()])

while parameter_search_df.shape[0] < K:
    new_paramter =  pd.DataFrame([generate_random_parameters()])
    parameter_search_df = pd.concat([parameter_search_df, new_paramter])
    logger.info(parameter_search_df.shape[0])
    if parameter_search_df.duplicated().any():
        logger.info("Duplicated.")
    parameter_search_df.drop_duplicates()

parameter_search_df['start_running_timestamp'] = pd.Series(dtype=str)
parameter_search_df['murmur_file'] = pd.Series(dtype=str)
parameter_search_df['outcome_file'] = pd.Series(dtype=str)
parameter_search_df['mean_murmur'] = pd.Series(dtype=float64)
parameter_search_df['std_murmur'] = pd.Series(dtype=float64)
parameter_search_df['std_outcome'] = pd.Series(dtype=float64)
parameter_search_df['mean_outcome'] = pd.Series(dtype=float64)

os.system("curl --create-dirs -o $HOME/.postgresql/root.crt -O https://cockroachlabs.cloud/clusters/6cadd36b-9892-418c-88c7-64a5781755ec/cert")
os.environ['DATABASE_URL'] = "cockroachdb://ohh:nZ-eJfpX1fro6l-b9szvCg@free-tier11.gcp-us-east1.cockroachlabs.cloud:26257/defaultdb?sslmode=verify-full&options=--cluster%3Damped-fox-1436"
engine = create_engine(os.environ["DATABASE_URL"])
conn = engine.connect()

res = conn.execute(text("SELECT now()")).fetchall()
print(res)
parameter_search_df.to_sql(name="parameters_search", con=conn, if_exists="replace")
logger.success("Parameters saved in SQL server")