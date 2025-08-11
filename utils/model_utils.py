from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch import nn as nn


def get_router_model(action_num: int, model_path: str, **kwargs) -> AutoModelForSequenceClassification:
    """
    Get the router model.

    Args:
        action_num (int): The number of actions.
        model_path (str): The path to the model.
        **kwargs: The keyword arguments.

    Returns:
        AutoModelForSequenceClassification: The router model.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=action_num, **kwargs)
    return model
