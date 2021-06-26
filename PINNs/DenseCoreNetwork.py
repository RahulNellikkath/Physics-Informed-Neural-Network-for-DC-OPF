# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 14:12:38 2021

@author: Rahul N
"""

import tensorflow as tf

class DenseCoreNetwork(tf.keras.models.Model):
    """
    This constitutes the core neural network with the PINN model. It outputs the angle for each generator based on
    the inputs for each generator's power, initial angle and frequency deviation. Additionally a common time input
    represents the time instance that shall be predicted.
    """

    def __init__(self, n_gbus, n_line, neurons_in_hidden_layers_Pg, neurons_in_hidden_layers_Lm):

        super(DenseCoreNetwork, self).__init__()
        # self.n_gbus=n_gbus
        # self.n_line=n_line
        # self.neurons_in_hidden_layer_1=neurons_in_hidden_layer_1
        # self.neurons_in_hidden_layer_Lm = neurons_in_hidden_layer_Lm
        
        self.hidden_layer_0 = tf.keras.layers.Dense(units=neurons_in_hidden_layers_Pg[0],
                                                            activation=tf.keras.activations.relu,
                                                            use_bias=True,
                                                            kernel_initializer=tf.keras.initializers.glorot_normal(),
                                                            bias_initializer=tf.keras.initializers.zeros(),
                                                    name='first_layer')
        self.hidden_layer_1 = tf.keras.layers.Dense(units=neurons_in_hidden_layers_Pg[1],
                                                            activation=tf.keras.activations.relu,
                                                            use_bias=True,
                                                            kernel_initializer=tf.keras.initializers.glorot_normal(),
                                                            bias_initializer=tf.keras.initializers.zeros(),
                                                    name='hidden_layer_1')
        self.hidden_layer_2 = tf.keras.layers.Dense(units=neurons_in_hidden_layers_Pg[2],
                                                            activation=tf.keras.activations.relu,
                                                            use_bias=True,
                                                            kernel_initializer=tf.keras.initializers.glorot_normal(),
                                                            bias_initializer=tf.keras.initializers.zeros(),
                                                    name='hidden_layer_2')
                                                    
        self.dense_output_layer_g = tf.keras.layers.Dense(units=n_gbus,
                                                        activation=tf.keras.activations.linear,
                                                        use_bias=True,
                                                        kernel_initializer=tf.keras.initializers.glorot_normal(),
                                                        name='output_layer_g')
        
##  Lg NN
        self.hidden_layer_Lm_0 = tf.keras.layers.Dense(units=neurons_in_hidden_layers_Lm[0],
                                                            activation=tf.keras.activations.relu,
                                                            use_bias=True,
                                                            kernel_initializer=tf.keras.initializers.glorot_normal(),
                                                            bias_initializer=tf.keras.initializers.zeros(),
                                                    name='hidden_layer_Lm_0')
        self.hidden_layer_Lm_1 = tf.keras.layers.Dense(units=neurons_in_hidden_layers_Lm[1],
                                                            activation=tf.keras.activations.relu,
                                                            use_bias=True,
                                                            kernel_initializer=tf.keras.initializers.glorot_normal(),
                                                            bias_initializer=tf.keras.initializers.zeros(),
                                                    name='hidden_layer_Lm_1')
        self.hidden_layer_Lm_2 = tf.keras.layers.Dense(units=neurons_in_hidden_layers_Lm[2],
                                                            activation=tf.keras.activations.relu,
                                                            use_bias=True,
                                                            kernel_initializer=tf.keras.initializers.glorot_normal(),
                                                            bias_initializer=tf.keras.initializers.zeros(),
                                                    name='hidden_layer_Lm_2')


        self.dense_output_layer_l = tf.keras.layers.Dense(units=1,
                                                        activation=tf.keras.activations.linear,
                                                        use_bias=True,
                                                        kernel_initializer=tf.keras.initializers.glorot_normal(),
                                                        name='output_layer_l')
        self.dense_output_layer_a_u = tf.keras.layers.Dense(units=n_gbus,
                                                        activation=tf.keras.activations.linear,
                                                        use_bias=True,
                                                        kernel_initializer=tf.keras.initializers.glorot_normal(),
                                                        name='output_layer_a_u')
        self.dense_output_layer_a_d = tf.keras.layers.Dense(units=n_gbus,
                                                        activation=tf.keras.activations.linear,
                                                        use_bias=True,
                                                        kernel_initializer=tf.keras.initializers.glorot_normal(),
                                                        name='output_layer_a_d')
        self.dense_output_layer_b_u = tf.keras.layers.Dense(units=n_line,
                                                        activation=tf.keras.activations.linear,
                                                        use_bias=True,
                                                        kernel_initializer=tf.keras.initializers.glorot_normal(),
                                                        name='output_layer_b_u')
        self.dense_output_layer_b_d = tf.keras.layers.Dense(units=n_line,
                                                        activation=tf.keras.activations.linear,
                                                        use_bias=True,
                                                        kernel_initializer=tf.keras.initializers.glorot_normal(),
                                                        name='output_layer_b_d')       
    def call(self, inputs, training=None, mask=None):
        x_power = inputs
        hidden_layer_0_output = self.hidden_layer_0(x_power)
        hidden_layer_1_output = self.hidden_layer_1(hidden_layer_0_output)
        hidden_layer_2_output = self.hidden_layer_2(hidden_layer_1_output)
        network_output_g = self.dense_output_layer_g(hidden_layer_2_output)

        hidden_layer_Lm_0_output = self.hidden_layer_Lm_0(x_power)
        hidden_layer_Lm_1_output = self.hidden_layer_Lm_1(hidden_layer_Lm_0_output)
        hidden_layer_Lm_2_output = self.hidden_layer_Lm_2(hidden_layer_Lm_1_output)
        network_output_l = self.dense_output_layer_l(hidden_layer_Lm_2_output)
        network_output_a_u = self.dense_output_layer_a_u(hidden_layer_Lm_2_output)
        network_output_a_d = self.dense_output_layer_a_d(hidden_layer_Lm_2_output)
        network_output_b_u = self.dense_output_layer_b_u(hidden_layer_Lm_2_output)
        network_output_b_d = self.dense_output_layer_b_d(hidden_layer_Lm_2_output)
        
        return network_output_g,network_output_l,network_output_a_u,network_output_a_d,network_output_b_u,network_output_b_d

    def call_inference(self, inputs, training=None, mask=None):
        x_power = inputs
        hidden_layer_0_output = self.hidden_layer_0(x_power)
        hidden_layer_1_output = self.hidden_layer_1(hidden_layer_0_output)
        hidden_layer_2_output = self.hidden_layer_2(hidden_layer_1_output)
        network_output_g = self.dense_output_layer_g(hidden_layer_2_output)

        hidden_layer_Lm_0_output = self.hidden_layer_Lm_0(x_power)
        hidden_layer_Lm_1_output = self.hidden_layer_Lm_1(hidden_layer_Lm_0_output)
        hidden_layer_Lm_2_output = self.hidden_layer_Lm_2(hidden_layer_Lm_1_output)
        network_output_l = self.dense_output_layer_l(hidden_layer_Lm_2_output)
        network_output_a_u = self.dense_output_layer_a_u(hidden_layer_Lm_2_output)
        network_output_a_d = self.dense_output_layer_a_d(hidden_layer_Lm_2_output)
        network_output_b_u = self.dense_output_layer_b_u(hidden_layer_Lm_2_output)
        network_output_b_d = self.dense_output_layer_b_d(hidden_layer_Lm_2_output)
        
        return network_output_g,network_output_l,network_output_a_u,network_output_a_d,network_output_b_u,network_output_b_d
