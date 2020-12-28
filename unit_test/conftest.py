from pathlib import Path
import pytest
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np


@pytest.fixture
def mvtec_test_image():
    return Path(__file__).parent.joinpath("unit_test_data", "mvtec_bottle_train.png")


@pytest.fixture
def resnet50_model(mvtec_test_image):
    """Example from "Extract features with VGG16"
    see https://keras.io/api/applications/

    return:
        an output feature generate by ResNet50 model
    """
    model = ResNet50(weights='imagenet', include_top=False)
    img = image.load_img(str(mvtec_test_image), target_size=(900, 900))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    # get output feature of model
    output_features = model.predict(x)
    return output_features


@pytest.fixture
def random_seed():
    return
