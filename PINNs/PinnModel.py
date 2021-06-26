from PINNs.PinnLayer import PinnLayer

import tensorflow as tf


class PinnModel(tf.keras.models.Model):

    def __init__(self,weight1,weight2,simulation_parameters):
        super(PinnModel, self).__init__()


        self.PinnLayer = PinnLayer(simulation_parameters=simulation_parameters)
        n_data_points = simulation_parameters['data_creation']['n_data_points']
        n_collocation = simulation_parameters['data_creation']['n_collocation']
        n_lbus = simulation_parameters['general']['n_lbus']
        n_total = n_data_points + n_collocation

        loss_weights = [n_total / n_data_points, weight1, weight1, weight1, weight1, weight1, weight2*(10**(-8))]

        self.compile(optimizer=tf.keras.optimizers.Adam(),
                     loss=tf.keras.losses.mean_absolute_error,
                     loss_weights=loss_weights)

        self.build(input_shape=[(None, n_lbus), (None, 1)])

    def call(self, inputs, training=None, mask=None):
        L_val, x_type = inputs
        network_output_g, n_o_l, n_o_a_u, n_o_a_d, n_o_b_u, n_o_b_d, network_output_physics = self.PinnLayer(L_val)
        #loss in gen prediction
        loss_network_output_Gen = tf.multiply(network_output_g, x_type)
        #loss in dual variables prediction
        loss_network_output_l   = tf.multiply(n_o_l, x_type)
        loss_network_output_a_u = tf.multiply(n_o_a_u, x_type)
        loss_network_output_a_d = tf.multiply(n_o_a_d, x_type)
        loss_network_output_b_u = tf.multiply(n_o_b_u, x_type)
        loss_network_output_b_d = tf.multiply(n_o_b_d, x_type)
        #KKT error 
        loss_network_output_physics = network_output_physics

        loss_output = (loss_network_output_Gen,
                       loss_network_output_l,
                       loss_network_output_a_u,
                       loss_network_output_a_d,
                       loss_network_output_b_u,
                       loss_network_output_b_d,
                        loss_network_output_physics)

        return loss_output
