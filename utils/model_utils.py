from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch import nn as nn


def get_router_model(action_num, model_path, **kwargs):
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=action_num, **kwargs)
    return model
