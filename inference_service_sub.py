from bertopic import BERTopic
import logging


def exec_init_model():
    # BERTopic model 로드
    model = BERTopic(embedding_model='bongsoo/kpf-sbert-128d-v1',
                     min_topic_size=5)
    model_info_dict = {
        "model": model
    }

    return model_info_dict


def exec_inference_dataframe(df, model_info_dict):
    logging.info(
        '[hunmin log] the start line of the function [exec_inference_dataframe]')

    model = model_info_dict['model']

    # data preprocess

    # data predict
    topics, probs = model.fit_transform()

    logging.info('[hunmin log] result : {}'.format(result))

    return result


def exec_inference_file(files, model_info_dict):
    """
    파일기반 추론함수는 files와 로드한 model을 전달받습니다.
    """
    logging.info(
        '[hunmin log] the start line of the function [exec_inference_file]')
    model = model_info_dict['model']

    inference_result = []
    for one_file in files:
        logging.info('[hunmin log] file inference')

        logging.info('[hunmin log] load model')

        # data predict


#     result = [DownloadFile(file_path=T3QAI_TRAIN_OUTPUT_PATH+'/Accuracy_Loss.png', file_name='result.jpg'),
#               DownloadFile(file_path=T3QAI_TRAIN_OUTPUT_PATH+'/Accuracy_Loss.png', file_name='result2.jpg')]
    result = {'inference': inference_result}
    return result
