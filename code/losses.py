import tensorflow as tf

class CustomMAE(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
    def call(self, y_true, y_pred):
        """
        Mean Absolute Error Loss
        """
        difference = tf.abs(y_true - y_pred)
        return tf.reduce_sum(difference, axis=-1)

class CustomMSE(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
    def call(self, y_true, y_pred):
        """
        Mean Square Error Loss
        """
        difference = tf.math.square(y_true - y_pred)
        return tf.reduce_sum(difference, axis=-1)

class SISDRLoss(tf.keras.losses.Loss):
    def __init__(self, zero_mean=True, eps=1e-8):
        super().__init__()
        self.zero_mean = zero_mean
        self.eps = eps
    def call(self, y_true, y_pred):
        """
        SISDRLoss for time domain variables
        """
        if self.zero_mean:
            input_mean = tf.repeat(tf.expand_dims(tf.math.reduce_mean(y_pred, axis=-1), axis=-1), y_pred.shape[1], axis=1)
            target_mean = tf.repeat(tf.expand_dims(tf.math.reduce_mean(y_true, axis=-1), axis=-1), y_true.shape[1], axis=1)
            input = tf.subtract(y_pred, input_mean)
            target = tf.subtract(y_true, target_mean)
            target = tf.cast(target, dtype=tf.float32)

        alpha = tf.reduce_sum(tf.multiply(input, target), axis=-1) / tf.add(tf.reduce_sum(tf.pow(target, 2), axis=-1), self.eps)
        target = tf.multiply(target, tf.expand_dims(alpha, axis=-1))
        res = input - target

        loss = 10 * tf.experimental.numpy.log10(
            tf.add(tf.reduce_sum(tf.pow(target, 2), axis=-1) / tf.add(tf.reduce_sum(tf.pow(res, 2), axis=-1), self.eps), self.eps)
        )
        return -loss

def modelLoss(stdct_target, predictions):
    """
    Attentive MultiResUNet Loss
    """
    mse = CustomMSE()
    loss_stdct = mse(stdct_target, predictions)
    return loss_stdct
