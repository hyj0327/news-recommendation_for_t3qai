from bertopic import BERTopic
import logging


def exec_init_model():
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
    topics, probs = model.fit_transform(
        documents=train_paragraph_data['paragraph'], embeddings=train_paragraph_embeddings)

    logging.info('[hunmin log] result : {}'.format(result))

    return result


def exec_inference_file(files, model_info_dict):
    model = model_info_dict['model']
