# inference_service.py


from inference_service_sub import exec_init_model, exec_inference_dataframe


import logging
logger = logging.getLogger()
logger.setLevel('INFO')


def init_model():
    params = exec_init_model()
    logging.info('[hunmin log] the end line of the function [init_model]')
    return {**params}


def inference_dataframe(df, model_info_dict):
    result = exec_inference_dataframe(df, model_info_dict)
    logging.info(
        '[hunmin log] the end line of the function [inference_dataframe]')
    return {**result}


#def inference_file(files, model_info_dict):
    #result = exec_inference_file(files, model_info_dict)
    #logging.info('[hunmin log] the end line of the function [inference_file]')
    #return result
