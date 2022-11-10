import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from losses import modelLoss
from settings import *

def train_step(input, stdct_target, time_target, model, optimiser):
    """
    Train step for each batch
    """
    with tf.GradientTape() as tape:
        predictions = model(input, training=True)
        loss = modelLoss(stdct_target, tf.squeeze(predictions))
    grad = tape.gradient(loss, model.trainable_variables)
    optimiser.apply_gradients(zip(grad, model.trainable_variables))
    return loss

def valid_step(input, stdct_target, time_target, model):
    """
    Validation step for each batch
    """
    predictions = model(input, training=False)
    loss= modelLoss(stdct_target, tf.squeeze(predictions))
    return loss

def save_output(input, target, model):
    """
    Generates Output Image
    """
    output_spec = model(np.expand_dims(input, axis=0), training=False)
    plt.subplot(1,2,1)
    plt.imshow(np.flipud(np.log(np.abs(target[:MAX_BINS // 2, :] + 1e-7))))
    plt.axis('off')
    plt.title('Target')
    plt.subplot(1,2,2)
    plt.imshow(np.flipud(np.log(np.abs(np.squeeze(output_spec + 1e-7)[:MAX_BINS // 2, :] ))))
    plt.axis('off')
    plt.title('Output Spect')
    plt.savefig('output.png', bbox_inches='tight')
    plt.close()
