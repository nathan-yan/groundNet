"""
    GroundNet
    For use in SLI 2018 to classify ground features
    Nathan Yan, 2017
"""

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

class GroundNet:
    def __init__(self, input_shape, layer_info, activation = tf.nn.relu, classes = 3):
        # output_shape is the same as input_shape, except for the channel number, which will be equal to classes

        self.input_shape = input_shape
        self.layer_info = layer_info
        self.classes = classes

        self.activation = activation

        self.weights = []

        # Initialize weights of GroundNet
        for l in range (len(self.layer_info)):
            # Use glorot uniform for ReLU
            weight = tf.get_variable("layer_" + str(l), layer_info[l],
                                     dtype = tf.float32)

            self.weights.append(weight)

        # Produce a final weight that takes the last layer and transforms it into a tensor suitable for classification
        final_weight = tf.get_variable("prediction_layer", [1, 1, layer_info[-1][-2], self.classes],
                                 dtype = tf.float32)
        self.weights.append(final_weight)

    def inference(self, inp):
        X = inp

        for l in range (len(self.weights)):
            X = tf.nn.conv2d(input = X,
                             filter = self.weights[l],
                             strides = [1, 1, 1, 1],
                             padding = "SAME")

            # If this isn't the last layer, ReLU
            if (l != len(self.weights) - 1):
                X = self.activation(X)
            else:
                # X at the end is a bs x h x w x c 4-tensor. We want to perform softmax along the channel axis
                X = tf.nn.softmax(X, dim = -1)

        return X

def main():
    BS = 32

    groundNet = GroundNet(None, [[3, 3, 3, 10], [3, 3, 10, 10]])

    # Create computational graph
    # Both inp and target should be 4-tensors of size bs x h x w x c
    inp = tf.placeholder(tf.float32)
    target = tf.placeholder(tf.float32)

    prediction = groundNet.inference(inp)

    # Log-liklihood loss
    # We want to minimize the loss. Log-liklihood is the log of the probability of the prediction being the target if the prediction parameterized a multinoulli distribution. As a result, we want to make that probability negative, so the more likely the prediction is the target, the smaller the loss (good!)

    # We're going to take the sum of all of these losses but not across the batchsize axis
    loss = -tf.reduce_mean(tf.reduce_sum(tf.log(prediction) * target +
                    tf.log(1 - prediction) * (1 - target),
                    axis = [1, 2, 3]))

    # And now for my favorite optimizer...
    train = tf.train.RMSPropOptimizer(1e-3).minimize(loss)

    # Quick inference/train test
    with tf.Session() as sess:
        # Init variables
        sess.run(tf.global_variables_initializer())

        # Generate a random image
        img = np.random.randn(1, 200, 200, 3)
        tar = np.random.randn(1, 200, 200, 3)

        for i in range (10):
            # Hopefully you should see a monotonic decrease in loss
            print(sess.run([train, loss], {inp : img, target : tar}))

main()
