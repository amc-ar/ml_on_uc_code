import tensorflow as tf
from tensorflow_model_optimization.python.core.keras.compat import keras


def pytorch_conv_kernel_init():
    # Matches PyTorch Conv2d default: kaiming_uniform_(a=sqrt(5))
    return keras.initializers.VarianceScaling(
        scale=1.0 / 3.0,
        mode="fan_in",
        distribution="uniform"
    )

class ConvBNReLU(keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding="same", bias=True, use_bn=True, **kwargs):
        super().__init__(**kwargs)
        self.conv = keras.layers.Conv2D(
            filters,
            kernel_size,
            padding=padding,
            use_bias=bias,
            kernel_initializer=pytorch_conv_kernel_init(),
            bias_initializer="zeros",
            kernel_regularizer=keras.regularizers.l2(1e-4),
        )
        self.bn = keras.layers.BatchNormalization(
            momentum=0.95,
            epsilon=1e-5
        ) if use_bn else None
        self.relu = keras.layers.ReLU()

    def call(self, x, training=False):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x, training=training)
        return self.relu(x)

class CifarModelTest400kTF(keras.Model):
    def __init__(self, num_classes=10, bias=True, use_bn=True, **kwargs):
        super().__init__(**kwargs)

        self.conv1 = ConvBNReLU(32, 3, bias=bias, use_bn=use_bn)

        self.conv2 = ConvBNReLU(64, 3, bias=bias, use_bn=use_bn)
        self.conv3 = ConvBNReLU(64, 3, bias=bias, use_bn=use_bn)

        self.pool1 = keras.layers.MaxPooling2D(pool_size=2, strides=2, padding="valid")
        self.conv4 = ConvBNReLU(64, 3, bias=bias, use_bn=use_bn)
        self.conv5 = ConvBNReLU(64, 3, bias=bias, use_bn=use_bn)

        self.pool2 = keras.layers.MaxPooling2D(pool_size=2, strides=2, padding="valid")
        self.conv7 = ConvBNReLU(96, 3, bias=bias, use_bn=use_bn)
        self.conv8 = ConvBNReLU(96, 3, bias=bias, use_bn=use_bn)
        self.conv9 = ConvBNReLU(96, 1, padding="valid", bias=bias, use_bn=use_bn)

        self.conv10 = ConvBNReLU(32, 3, bias=bias, use_bn=use_bn)

        self.pool3 = keras.layers.MaxPooling2D(pool_size=8, strides=8, padding="valid")
        self.conv11 = ConvBNReLU(32, 1, padding="valid", bias=bias, use_bn=use_bn)
        self.dropout12 = keras.layers.Dropout(0.5)
        self.classifier = keras.layers.Conv2D(
            num_classes,
            kernel_size=1,
            padding="valid",
            use_bias=bias,
            kernel_initializer=pytorch_conv_kernel_init(),
            bias_initializer="zeros",
            kernel_regularizer=keras.regularizers.l2(1e-4),

        )
        self.flatten = keras.layers.Flatten()

    def call(self, x, training=False):
        x = self.conv1(x, training=training)

        res = self.conv2(x, training=training)
        x = self.conv3(res, training=training) + res
        # x = self.dropout1(x, training=training)

        x = self.pool1(x)
        res = self.conv4(x, training=training)
        x = self.conv5(res, training=training) + res
        # x = self.dropout2(x, training=training)

        x = self.pool2(x)
        x = self.conv7(x, training=training)

        res = self.conv8(x, training=training)
        x = self.conv9(res, training=training) + res
        # x = self.dropout3(x, training=training)

        x = self.conv10(x, training=training)
        x = self.pool3(x)
        x = self.conv11(x, training=training)
        x = self.dropout12(x, training=training)

        x = self.classifier(x)

        return self.flatten(x)


def cifarmodeltest_400k_tf_do50(num_classes=10, bias=True, use_bn=True):
    return CifarModelTest400kTF(
        num_classes=num_classes,
        bias=bias,
        use_bn=use_bn,
        name="cifarmodeltest_400k_tf",
    )


def _functional_conv_bn_relu(x, filters, kernel_size, padding="same", bias=True, use_bn=True, name=None):
    x = keras.layers.Conv2D(
        filters,
        kernel_size,
        padding=padding,
        use_bias=bias,
        kernel_initializer=pytorch_conv_kernel_init(),
        bias_initializer="zeros",
        kernel_regularizer=keras.regularizers.l2(1e-4),
        name=f"{name}_conv",
    )(x)
    if use_bn:
        x = keras.layers.BatchNormalization(
            momentum=0.95,
            epsilon=1e-5,
            name=f"{name}_bn",
        )(x)
    return keras.layers.ReLU(name=f"{name}_relu")(x)


def cifarmodeltest_400k_tf_do50_functional(num_classes=10, bias=True, use_bn=True):
    inputs = keras.Input(shape=(32, 32, 3), name="input")

    x = _functional_conv_bn_relu(inputs, 32, 3, bias=bias, use_bn=use_bn, name="conv1")

    res = _functional_conv_bn_relu(x, 64, 3, bias=bias, use_bn=use_bn, name="conv2")
    x = _functional_conv_bn_relu(res, 64, 3, bias=bias, use_bn=use_bn, name="conv3")
    x = keras.layers.Add(name="resid1")([x, res])

    x = keras.layers.MaxPooling2D(pool_size=2, strides=2, padding="valid", name="pool1")(x)
    res = _functional_conv_bn_relu(x, 64, 3, bias=bias, use_bn=use_bn, name="conv4")
    x = _functional_conv_bn_relu(res, 64, 3, bias=bias, use_bn=use_bn, name="conv5")
    x = keras.layers.Add(name="resid2")([x, res])

    x = keras.layers.MaxPooling2D(pool_size=2, strides=2, padding="valid", name="pool2")(x)
    x = _functional_conv_bn_relu(x, 96, 3, bias=bias, use_bn=use_bn, name="conv7")

    res = _functional_conv_bn_relu(x, 96, 3, bias=bias, use_bn=use_bn, name="conv8")
    x = _functional_conv_bn_relu(res, 96, 1, padding="valid", bias=bias, use_bn=use_bn, name="conv9")
    x = keras.layers.Add(name="resid3")([x, res])

    x = _functional_conv_bn_relu(x, 32, 3, bias=bias, use_bn=use_bn, name="conv10")
    x = keras.layers.MaxPooling2D(pool_size=8, strides=8, padding="valid", name="pool3")(x)
    x = _functional_conv_bn_relu(x, 32, 1, padding="valid", bias=bias, use_bn=use_bn, name="conv11")
    x = keras.layers.Dropout(0.5, name="dropout12")(x)
    x = keras.layers.Conv2D(
        num_classes,
        kernel_size=1,
        padding="valid",
        use_bias=bias,
        kernel_initializer=pytorch_conv_kernel_init(),
        bias_initializer="zeros",
        kernel_regularizer=keras.regularizers.l2(1e-4),
        name="classifier",
    )(x)

    outputs = keras.layers.Flatten(name="flatten")(x)
    return keras.Model(inputs, outputs, name="cifarmodeltest_400k_tf")


def copy_cifarmodeltest_400k_weights(source_model, target_model):
    for block_name in ("conv1", "conv2", "conv3", "conv4", "conv5", "conv7", "conv8",
                       "conv9", "conv10", "conv11"):
        source_block = getattr(source_model, block_name)
        target_model.get_layer(f"{block_name}_conv").set_weights(source_block.conv.get_weights())
        if source_block.bn is not None:
            target_model.get_layer(f"{block_name}_bn").set_weights(source_block.bn.get_weights())

    target_model.get_layer("classifier").set_weights(source_model.classifier.get_weights())
    return target_model