from __future__ import absolute_import, print_function

from csbdeep.utils import backend_channels_last
from csbdeep.utils.tf import keras_import

from .model2d import Config2D, SplineDist2D, SplineDistData2D

K = keras_import("backend")
if not backend_channels_last():
    raise NotImplementedError(
        "Keras is configured to use the '%s' image data format, which is currently not supported. "
        "Please change it to use 'channels_last' instead: "
        "https://keras.io/getting-started/faq/#where-is-the-keras-configuration-file-stored"
        % K.image_data_format()
    )
del backend_channels_last, K

from csbdeep.models import (clear_models_and_aliases, register_aliases,
                            register_model)

# register pre-trained models and aliases (TODO: replace with updatable solution)

del register_model, register_aliases, clear_models_and_aliases
