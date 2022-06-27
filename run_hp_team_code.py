import os
from collections import OrderedDict
import json
from loguru import logger

# Embs Size : [16, 64, 256]
# Weight class murmur : [1, 1.5, 3, 5]
# Weight class decision : [1, 1.5, 3, 5]
# Murmur Image Size : [32, 72, 108, 216]
# Random embeddings per patient: [15, 50, 100, 128]
# Reshuffle for training: [1, 3, 5, 10]
embs_sizes = [16, 64, 256]
reshuffle_patients = [1, 3, 5, 10]
embs_per_patients = [15, 50, 100, 150]
murmur_image_sizes = [32, 72, 108, 216]
weight_class_murmurs = [1, 1.5, 3, 5]
weight_class_decisions = [1, 1.5, 3, 5]
parameters = OrderedDict()
parameters["embs_size"] = embs_sizes
parameters["reshuffle_patients"] = reshuffle_patients
parameters["embs_per_patients"] = embs_per_patients
parameters["murmur_image_sizes"] = murmur_image_sizes
parameters["weight_class_murmur"] = weight_class_murmurs
parameters["weight_class_decisions"] = weight_class_decisions


best_embs = {"best_param": None, "performance" : 0}

base = {
    "embs_size" : 64,
    "reshuffle_patient" : 1,
    "embs_per_patient" : 15,
    "murmur_image_size" : 108,
    "weight_class_murmur" : 1,
    "weight_class_decisions" : 1.5,
    "performance" : 0.0
}

def create_configfile_given_cofig(config):
    base_config = { "TEST_MODE": True }
    base_config.update(config)
    json_object = json.dumps(base_config, indent = 4) 
    ptr = open("ohh.config", "w")
    ptr.write(json_object)

def get_current_performace():
    data = None
    with open("decision_evaluation.json") as json_file:
        data = json.load(json_file)
    return data["auc"]

def run_model():
    # os.system("bash run_train_full.bash 0")
    os.system("rm -r recordings_aux")
    os.system("rm -r test_outputs")
    os.system("bash test_code_quick.bash 0")

for parameter_name in parameters.keys():
    parameters_values = parameters[parameter_name]
    for value in parameters_values:
        tentative_config = base.copy()
        tentative_config[parameter_name] = value
        logger.info("New Tentative: {} - {}", parameter_name, value)
        logger.info("Current Base: {}".format(base))
        create_configfile_given_cofig(tentative_config)
        run_model()
        current_performance = get_current_performace()
        logger.info("Previous performance: {}".format(base["performance"]))
        logger.info("New performance: {}".format(current_performance))
        if base["performance"] < current_performance:
            logger.info("Updated.")
            base["performance"] = current_performance
            base[parameter_name] = value