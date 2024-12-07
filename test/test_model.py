import sys
sys.path.append("G:\workspace_github\dnn_models")

import pytest
import torch
import torch.nn as nn
from util.visualizer import show_model
from util.logger import logger



@pytest.mark.parametrize("model, input_to_model", [
        (nn.Conv2d(3, 16, 3), torch.randn(1, 3, 32, 32))
    ])
def test_show_model(model, input_to_model):
    # Call the function
    logger.info(f"{sys.path}")
    try:
        show_model(model, input_to_model)
    except Exception as e:
        logger.error(f"Error occurred while calling show_model: {str(e)}")
    
