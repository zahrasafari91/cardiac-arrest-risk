######################### Original Model ################################
# from tensorflow.keras.layers import (
#     Input, Conv1D, MaxPooling1D, Dropout, BatchNormalization, Activation, Add, Flatten, Dense)
# from tensorflow.keras.models import Model, load_model
# import numpy as np
# from tensorflow.keras.models import Model, load_model


# class ResidualUnit(object):
#     """Residual unit block (unidimensional)."""
#     def __init__(self, n_samples_out, n_filters_out, kernel_initializer='he_normal',
#                  dropout_keep_prob=0.8, kernel_size=17, preactivation=True,
#                  postactivation_bn=False, activation_function='relu'):
#         self.n_samples_out = n_samples_out
#         self.n_filters_out = n_filters_out
#         self.kernel_initializer = kernel_initializer
#         self.dropout_rate = 1 - dropout_keep_prob
#         self.kernel_size = kernel_size
#         self.preactivation = preactivation
#         self.postactivation_bn = postactivation_bn
#         self.activation_function = activation_function

#     def _skip_connection(self, y, downsample, n_filters_in):
#         """Implement skip connection."""
#         if downsample > 1:
#             y = MaxPooling1D(downsample, strides=downsample, padding='same')(y)
#         elif downsample == 1:
#             y = y
#         else:
#             raise ValueError("Number of samples should always decrease.")

#         if n_filters_in != self.n_filters_out:
#             y = Conv1D(self.n_filters_out, 1, padding='same',
#                        use_bias=False, kernel_initializer=self.kernel_initializer)(y)
#         return y

#     def _batch_norm_plus_activation(self, x):
#         if self.postactivation_bn:
#             x = Activation(self.activation_function)(x)
#             x = BatchNormalization(center=False, scale=False)(x)
#         else:
#             x = BatchNormalization()(x)
#             x = Activation(self.activation_function)(x)
#         return x

#     def __call__(self, inputs):
#         """Residual unit."""
#         x, y = inputs
#         n_samples_in = y.shape[1]
#         downsample = int(n_samples_in // self.n_samples_out)
#         n_filters_in = y.shape[2]
#         y = self._skip_connection(y, downsample, n_filters_in)
        
#         # 1st layer
#         x = Conv1D(self.n_filters_out, self.kernel_size, padding='same',
#                    use_bias=False, kernel_initializer=self.kernel_initializer)(x)
#         x = self._batch_norm_plus_activation(x)
#         if self.dropout_rate > 0:
#             x = Dropout(self.dropout_rate)(x)

#         # 2nd layer
#         x = Conv1D(self.n_filters_out, self.kernel_size, strides=downsample,
#                    padding='same', use_bias=False,
#                    kernel_initializer=self.kernel_initializer)(x)
#         if self.preactivation:
#             x = Add()([x, y])  # Sum skip connection and main connection
#             y = x
#             x = self._batch_norm_plus_activation(x)
#             if self.dropout_rate > 0:
#                 x = Dropout(self.dropout_rate)(x)
#         else:
#             x = BatchNormalization()(x)
#             x = Add()([x, y])  # Sum skip connection and main connection
#             x = Activation(self.activation_function)(x)
#             if self.dropout_rate > 0:
#                 x = Dropout(self.dropout_rate)(x)
#             y = x
#         return [x, y]

# def get_model(n_classes, last_layer='sigmoid'):
#     kernel_size = 16
#     kernel_initializer = 'he_normal'
#     # Update input shape to match MUSIC dataset (4096 samples, 3 leads)
#     signal = Input(shape=(4096, 3), dtype=np.float32, name='signal')
#     x = signal
#     x = Conv1D(64, kernel_size, padding='same', use_bias=False,
#                kernel_initializer=kernel_initializer)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x, y = ResidualUnit(1024, 128, kernel_size=kernel_size,
#                         kernel_initializer=kernel_initializer)([x, x])
#     x, y = ResidualUnit(256, 196, kernel_size=kernel_size,
#                         kernel_initializer=kernel_initializer)([x, y])
#     x, y = ResidualUnit(64, 256, kernel_size=kernel_size,
#                         kernel_initializer=kernel_initializer)([x, y])
#     x, _ = ResidualUnit(16, 320, kernel_size=kernel_size,
#                         kernel_initializer=kernel_initializer)([x, y])
#     x = Flatten()(x)
#     diagn = Dense(n_classes, activation=last_layer, kernel_initializer=kernel_initializer)(x)
#     model = Model(signal, diagn)
#     return model

# if __name__ == "__main__":
#     # Adjust number of classes based on MUSIC dataset (binary classification for cardiac arrest risk)
#     n_classes = 1  # Binary classification
#     model = get_model(n_classes)
#     model.summary()



#************************ Create New Model ********************************

# from tensorflow.keras.models import Model, load_model
# from tensorflow.keras.layers import (
#     Input, Conv1D, Dense, Flatten, BatchNormalization,
#     Activation, MaxPooling1D, Dropout, Add
# )
# from tensorflow.keras.regularizers import l2


# def residual_block(input_tensor, filters, kernel_size=16, strides=1, name_prefix="res_block"):
#     """
#     A residual block with a shortcut connection.
    
#     Args:
#         input_tensor (tf.Tensor): Input tensor.
#         filters (int): Number of filters for the Conv1D layers.
#         kernel_size (int): Kernel size for the Conv1D layers.
#         strides (int): Stride for the Conv1D layers.
#         name_prefix (str): Prefix for layer names.

#     Returns:
#         tf.Tensor: Output tensor.
#     """
#     # Save the input as the shortcut
#     shortcut = input_tensor

#     # First convolutional layer
#     x = Conv1D(
#         filters=filters,
#         kernel_size=kernel_size,
#         strides=strides,
#         padding="same",
#         name=f"{name_prefix}_conv1"
#     )(input_tensor)
#     x = BatchNormalization(name=f"{name_prefix}_bn1")(x)
#     x = Activation("relu", name=f"{name_prefix}_act1")(x)

#     # Second convolutional layer
#     x = Conv1D(
#         filters=filters,
#         kernel_size=kernel_size,
#         strides=1,
#         padding="same",
#         name=f"{name_prefix}_conv2"
#     )(x)
#     x = BatchNormalization(name=f"{name_prefix}_bn2")(x)

#     # Adjust shortcut dimensions if necessary
#     if input_tensor.shape[-1] != filters:
#         shortcut = Conv1D(
#             filters=filters,
#             kernel_size=1,  # Pointwise convolution to adjust channel dimensions
#             strides=strides,
#             padding="same",
#             name=f"{name_prefix}_shortcut_conv"
#         )(shortcut)
#         shortcut = BatchNormalization(name=f"{name_prefix}_shortcut_bn")(shortcut)

#     # Add the shortcut connection
#     x = Add(name=f"{name_prefix}_add")([x, shortcut])
#     x = Activation("relu", name=f"{name_prefix}_act2")(x)

#     return x


# def load_pretrained_model(weights_path, n_classes=1):
#     """
#     Adapt the pre-trained model for transfer learning to handle combined input (4096, 6).
    
#     Args:
#         weights_path: Path to the pre-trained model.
#         n_classes: Number of target classes.

#     Returns:
#         Model: Modified Keras model.
#     """
#     # Load the pre-trained model
#     base_model = load_model(weights_path, compile=False)

#     # Input for combined data
#     combined_input = Input(shape=(4096, 6), name="Combined_Input")

#     # Process Combined Input
#     x = Conv1D(64, kernel_size=16, strides=1, padding="same", activation="relu", name="conv1")(combined_input)
#     x = BatchNormalization(name="batch_norm1")(x)
#     x = MaxPooling1D(pool_size=4, strides=4, name="maxpool1")(x)
#     x = residual_block(x, filters=64, name_prefix="res1")
#     x = Dropout(0.3, name="dropout1")(x)

#     x = residual_block(x, filters=128, name_prefix="res2")
#     x = MaxPooling1D(pool_size=4, strides=4, name="maxpool2")(x)
#     x = Dropout(0.5, name="dropout2")(x)

#     # Flatten and Dense layers
#     x = Flatten(name="flatten")(x)
#     x = Dense(128, activation="relu", kernel_regularizer=l2(0.001), name="dense1")(x)
#     x = Dropout(0.5, name="dropout3")(x)
#     output = Dense(n_classes, activation="sigmoid", name="output")(x)

#     # Create the final model
#     model = Model(inputs=combined_input, outputs=output)

#     # Freeze layers from the pre-trained model
#     for layer in base_model.layers:
#         if not isinstance(layer, BatchNormalization):
#             layer.trainable = False

#     return model


# if __name__ == "__main__":
#     weights_path = "/home/zahra.safarialamoti/projects/cardiac_arrest_risk/model/model.hdf5"
#     model = load_pretrained_model(weights_path, n_classes=1)
#     model.summary()



#************************ transfer model **************************

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Dropout, BatchNormalization,
    Activation, Add, Flatten, Dense
)

#ResidualUnit  

class ResidualUnit(object):
    """One-dimensional residual block (same as your older code)."""
    def __init__(self, n_samples_out, n_filters_out,
                 kernel_initializer='he_normal',
                 dropout_keep_prob=0.8,
                 kernel_size=17,
                 preactivation=True,
                 postactivation_bn=False,
                 activation_function='relu'):
        self.n_samples_out = n_samples_out
        self.n_filters_out = n_filters_out
        self.kernel_initializer = kernel_initializer
        self.dropout_rate = 1 - dropout_keep_prob
        self.kernel_size = kernel_size
        self.preactivation = preactivation
        self.postactivation_bn = postactivation_bn
        self.activation_function = activation_function

    def _skip_connection(self, y, downsample, n_filters_in):
        if downsample > 1:
            y = MaxPooling1D(downsample, strides=downsample, padding='same')(y)
        elif downsample == 1:
            pass
        else:
            raise ValueError("Number of samples should always decrease.")
        if n_filters_in != self.n_filters_out:
            y = Conv1D(self.n_filters_out, 1, padding='same',
                       use_bias=False, kernel_initializer=self.kernel_initializer)(y)
        return y

    def _batch_norm_plus_activation(self, x):
        if self.postactivation_bn:
            x = Activation(self.activation_function)(x)
            x = BatchNormalization(center=False, scale=False)(x)
        else:
            x = BatchNormalization()(x)
            x = Activation(self.activation_function)(x)
        return x

    def __call__(self, inputs):
        x, y = inputs
        n_samples_in = y.shape[1]
        downsample = n_samples_in // self.n_samples_out
        n_filters_in = y.shape[2]
        y = self._skip_connection(y, downsample, n_filters_in)

        # 1st conv
        x = Conv1D(self.n_filters_out, self.kernel_size, padding='same',
                   use_bias=False, kernel_initializer=self.kernel_initializer)(x)
        x = self._batch_norm_plus_activation(x)
        if self.dropout_rate > 0:
            x = Dropout(self.dropout_rate)(x)

        # 2nd conv
        x = Conv1D(self.n_filters_out, self.kernel_size, strides=downsample,
                   padding='same', use_bias=False,
                   kernel_initializer=self.kernel_initializer)(x)
        if self.preactivation:
            x = Add()([x, y])
            y = x
            x = self._batch_norm_plus_activation(x)
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate)(x)
        else:
            x = BatchNormalization()(x)
            x = Add()([x, y])
            x = Activation(self.activation_function)(x)
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate)(x)
            y = x

        return [x, y]


#Get old model 

def get_old_model(n_classes=6, last_layer='sigmoid'):
    """
    Returns the old model architecture with input shape (4096, 12).
    """
    kernel_size = 16
    kernel_initializer = 'he_normal'
    
    signal = Input(shape=(4096, 12), dtype=np.float32, name='signal')
    x = signal
    
    # initial conv
    x = Conv1D(64, kernel_size, padding='same', use_bias=False,
               kernel_initializer=kernel_initializer, name='old_conv1')(x)
    x = BatchNormalization(name='old_bn1')(x)
    x = Activation('relu', name='old_relu1')(x)

    # residual blocks
    x, y = ResidualUnit(1024, 128, kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer)([x, x])
    x, y = ResidualUnit(256, 196, kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer)([x, y])
    x, y = ResidualUnit(64, 256, kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer)([x, y])
    x, _ = ResidualUnit(16, 320, kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer)([x, y])

    x = Flatten(name='old_flatten')(x)
    output = Dense(n_classes, activation=last_layer,
                   kernel_initializer=kernel_initializer,
                   name='old_output')(x)

    model = Model(inputs=signal, outputs=output)
    return model


#Get new model

def get_new_model(n_classes=1, last_layer='sigmoid'):
    """
    Returns the new model architecture with input shape (4096, 6).
    """
    kernel_size = 16
    kernel_initializer = 'he_normal'
    
    signal = Input(shape=(4096, 6), dtype=np.float32, name='signal')
    x = signal
    
    # initial conv
    x = Conv1D(64, kernel_size, padding='same', use_bias=False,
               kernel_initializer=kernel_initializer, name='new_conv1')(x)
    x = BatchNormalization(name='new_bn1')(x)
    x = Activation('relu', name='new_relu1')(x)

    # residual blocks
    x, y = ResidualUnit(1024, 128, kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer)([x, x])
    x, y = ResidualUnit(256, 196, kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer)([x, y])
    x, y = ResidualUnit(64, 256, kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer)([x, y])
    x, _ = ResidualUnit(16, 320, kernel_size=kernel_size,
                        kernel_initializer=kernel_initializer)([x, y])

    x = Flatten(name='new_flatten')(x)
    output = Dense(n_classes, activation=last_layer,
                   kernel_initializer=kernel_initializer,
                   name='new_output')(x)

    model = Model(inputs=signal, outputs=output)
    return model

#Load pretrained model for Transfer Learning

def load_pretrained_model(old_weights_path, n_classes=1, freeze_until=5):
    """
    1) Load the old model (pre-trained weights).
    2) Build the new model with shape (4096,6).
    3) Transfer weights from old --> new (partial for the first Conv1D).
    4) Freeze desired layers (by index).
    5) Return new model for fine-tuning.
    """
    # --- Load old model weights ---
    old_model = get_old_model(n_classes=6)  # old model has 6 output classes
    old_model.load_weights(old_weights_path)
    print("Loaded old model weights.")

    # --- Build new model for single-label classification ---
    new_model = get_new_model(n_classes=n_classes)
    print("Created new model with input shape (4096, 6).")

    # --- Transfer weights layer-by-layer if shapes match ---
    for i in range(len(old_model.layers)):
        old_layer = old_model.layers[i]
        new_layer = new_model.layers[i] if i < len(new_model.layers) else None
        if not new_layer:
            break  # no corresponding layer in new_model

        old_w = old_layer.get_weights()
        new_w = new_layer.get_weights()
        # If either layer has no weights (like Flatten or Activation, etc.), skip
        if not old_w or not new_w:
            continue

        # If shapes match exactly
        if all(ow.shape == nw.shape for ow, nw in zip(old_w, new_w)):
            new_layer.set_weights(old_w)
            print(f"Transferred weights: {old_layer.name} --> {new_layer.name}")
        else:
            print(f"Shape mismatch: {old_layer.name} vs {new_layer.name}, skipping exact copy.")

    # --- Special partial transfer for the first Conv1D (12 -> 6 channels) ---
    # old_conv1 kernel
    old_conv1_weights = old_model.get_layer('old_conv1').get_weights()
    if old_conv1_weights:
        # old_conv1_weights[0] is kernel of shape [kernel_size, 12, filters], old_conv1_weights[1] is BN or bias if used
        kernel_12 = old_conv1_weights[0]  # shape (kernel_size, 12, 64)
        # We'll slice the first 6 channels
        kernel_6 = kernel_12[:, :6, :]  # shape (kernel_size, 6, 64)

        new_conv1_layer = new_model.get_layer('new_conv1')
        new_conv1_weights = new_conv1_layer.get_weights()
        # new_conv1_weights[0] shape is (kernel_size, 6, 64)
        if len(new_conv1_weights) > 0 and kernel_6.shape == new_conv1_weights[0].shape:
            new_conv1_layer.set_weights([kernel_6])  # set just the kernel
            print("Partially transferred weights for the first Conv1D layer (old_conv1 -> new_conv1).")

    # --- Freeze the first N layers if requested ---
    for idx, layer in enumerate(new_model.layers[:freeze_until]):
        layer.trainable = False
        print(f"Froze layer {idx} -> {layer.name}")

    print("New model is now ready for fine-tuning.")
    return new_model

#print("model summary", load_pretrained_model("/home/zahra.safarialamoti/projects/cardiac_arrest_risk/model/model.hdf5", n_classes=1, freeze_until=5).summary())



