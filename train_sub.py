import os
import logging
from t3qai_client import T3QAI_TRAIN_OUTPUT_PATH, T3QAI_TRAIN_MODEL_PATH, T3QAI_TRAIN_DATA_PATH, T3QAI_TEST_DATA_PATH, T3QAI_MODULE_PATH



def exec_train():
    logging.info('[hunmin log] the start line of the function [exec_train]')
    logging.info('[hunmin log] T3QAI_TRAIN_DATA_PATH : {}'.format(
        T3QAI_TRAIN_DATA_PATH))

    list_files_directories(T3QAI_TRAIN_DATA_PATH)
    my_path = os.path.join(T3QAI_TRAIN_DATA_PATH, 'dataset') + '/'


def list_files_directories(path):
    dir_list = os.listdir(path)
    logging.info('[hunmin log] Files and directories in {} :'.format(path))
    logging.info('[hunmin log] dir_list : {}'.format(dir_list))
