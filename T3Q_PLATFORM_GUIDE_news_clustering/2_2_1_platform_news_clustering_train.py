# train.py
import logging
import os

from train_sub import exec_train
import t3qai_client as tc
from t3qai_client import T3QAI_TRAIN_OUTPUT_PATH, T3QAI_TRAIN_MODEL_PATH, \
    T3QAI_TRAIN_DATA_PATH, T3QAI_TEST_DATA_PATH, T3QAI_MODULE_PATH


def main():
    logging.info('[train.py] main() start')
    result = None
    result_msg = "success"
    tc.train_start()
    try:
        train()
    except Exception as e:
        result = e
        result_msg = e
        logging.info('error log : {}'.format(e))
    tc.train_finish(result, result_msg)


def train():
    exec_train()
    logging.info('[hunmin log] the end line of the function [train]')


if __name__ == '__main__':
    main()
