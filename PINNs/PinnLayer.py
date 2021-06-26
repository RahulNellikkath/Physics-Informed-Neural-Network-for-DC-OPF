import tensorflow as tf
import numpy as np
from PINNs.DenseCoreNetwork import DenseCoreNetwork

    
class PinnLayer(tf.keras.layers.Layer):
    """
    This layer includes the prediction
    """

    def __init__(self, simulation_parameters):
        super(PinnLayer, self).__init__()

        self.n_buses = simulation_parameters['general']['n_buses']
        self.n_gbus = simulation_parameters['general']['n_gbus']
        self.n_line = simulation_parameters['general']['n_line']
        self.neurons_in_hidden_layers_Pg = simulation_parameters['training']['neurons_in_hidden_layers_Pg']
        self.neurons_in_hidden_layers_Lm = simulation_parameters['training']['neurons_in_hidden_layers_Lm']
        self.DenseCoreNetwork = DenseCoreNetwork(n_gbus =self.n_gbus, n_line =self.n_line,
                                                 neurons_in_hidden_layers_Pg=self.neurons_in_hidden_layers_Pg,
                                                 neurons_in_hidden_layers_Lm=self.neurons_in_hidden_layers_Lm)
        self.C_Pg=simulation_parameters['true_system']['C_Pg']
        self.Pg_max=simulation_parameters['true_system']['Pg_max']
        self.Pg_min=simulation_parameters['true_system']['Pg_min']
        self.Pl_max=simulation_parameters['true_system']['Pl_max']
        self.Pg_max_act=simulation_parameters['true_system']['Pg_max_act']
        self.PTDF=simulation_parameters['true_system']['PTDF']
        self.L_max= simulation_parameters['general']['L_max']
        self.Lg_Max=simulation_parameters['Lg_Max']
        self.Map_g=simulation_parameters['true_system']['Map_g']
        self.Map_L=simulation_parameters['true_system']['Map_L']
    def Get_KKT_error(self,P_Gens,P_Loads,n_o_l, n_o_a_u, n_o_a_d, n_o_b_u, n_o_b_d):       
        # KKT primal conditions
        KKT_error = tf.abs(tf.reduce_sum(tf.multiply(P_Gens,self.Pg_max_act), axis=1) - tf.reduce_sum(self.L_max*0.6 + tf.multiply(P_Loads,self.L_max*0.4), axis=1))/np.max(self.L_max)
        KKT_error = KKT_error + tf.reduce_sum(tf.nn.relu(P_Gens - self.Pg_max), axis=1)/self.n_gbus
        KKT_error = KKT_error + tf.reduce_sum(tf.nn.relu(self.Pg_min-P_Gens), axis=1)/self.n_gbus
        KKT_error = KKT_error + tf.reduce_sum(tf.nn.relu(tf.matmul((tf.matmul(tf.multiply(P_Gens,self.Pg_max_act),self.Map_g)-tf.matmul(self.L_max*0.6 + tf.multiply(P_Loads,self.L_max*0.4),self.Map_L)),self.PTDF) - self.Pl_max), axis=1)/(np.max(self.L_max)*self.n_line)
        KKT_error = KKT_error + tf.reduce_sum(tf.nn.relu(tf.matmul((tf.matmul(self.L_max*0.6 + tf.multiply(P_Loads,self.L_max*0.4),self.Map_L) - tf.matmul(tf.multiply(P_Gens,self.Pg_max_act),self.Map_g)),self.PTDF) -self.Pl_max), axis=1)/(np.max(self.L_max)*self.n_line)
        
        #KKT stationarity condition
        KKT_error = KKT_error + tf.reduce_sum(n_o_l, axis=1)*self.Lg_Max[0]/100 +  tf.reduce_sum(tf.abs(np.matmul(self.C_Pg,self.Map_g) + tf.matmul(n_o_a_d,self.Map_g)*self.Lg_Max[2] - tf.matmul(n_o_a_u,self.Map_g)*self.Lg_Max[1] - tf.matmul(n_o_b_u*self.Lg_Max[3],self.PTDF, transpose_b=True) + tf.matmul(n_o_b_d*self.Lg_Max[4],self.PTDF, transpose_b=True)), axis=1)/100 
        
        #KKT complementary slackness conditions
        KKT_error = KKT_error + tf.reduce_sum(tf.abs(tf.multiply(n_o_a_u,P_Gens - self.Pg_max)), axis=1)/self.n_gbus
        KKT_error = KKT_error + tf.reduce_sum(tf.abs(tf.multiply(n_o_a_d,(self.Pg_min-P_Gens))), axis=1)/self.n_gbus
        KKT_error = KKT_error + tf.reduce_sum(tf.abs(tf.multiply(n_o_b_u,tf.matmul((tf.matmul(tf.multiply(P_Gens,self.Pg_max_act),self.Map_g)-tf.matmul(self.L_max*0.6 + tf.multiply(P_Loads,self.L_max*0.4),self.Map_L)),self.PTDF) - self.Pl_max)), axis=1)/(np.max(self.L_max)*self.n_line)
        KKT_error = KKT_error + tf.reduce_sum(tf.abs(tf.multiply(n_o_b_d,tf.matmul((tf.matmul(self.L_max*0.6 + tf.multiply(P_Loads,self.L_max*0.4),self.Map_L) - tf.matmul(tf.multiply(P_Gens,self.Pg_max_act),self.Map_g)),self.PTDF) -self.Pl_max)), axis=1)/(np.max(self.L_max)*self.n_line)

        #KKT dual variables
        KKT_error = KKT_error + tf.reduce_sum(tf.nn.relu(tf.math.negative(n_o_a_u)), axis=1)
        KKT_error = KKT_error + tf.reduce_sum(tf.nn.relu(tf.math.negative(n_o_a_d)), axis=1)
        KKT_error = KKT_error + tf.reduce_sum(tf.nn.relu(tf.math.negative(n_o_b_u)), axis=1)
        KKT_error = KKT_error + tf.reduce_sum(tf.nn.relu(tf.math.negative(n_o_b_d)), axis=1)
            
        return KKT_error
        
    def call(self, inputs, **kwargs):
        
        L_Val = tf.convert_to_tensor(inputs)
        
        network_output_g, n_o_l, n_o_a_u, n_o_a_d, n_o_b_u, n_o_b_d= self.DenseCoreNetwork.call_inference(L_Val, **kwargs)

        KKT_error = self.Get_KKT_error(network_output_g,L_Val,n_o_l, n_o_a_u, n_o_a_d, n_o_b_u, n_o_b_d)
        
        return network_output_g, n_o_l, n_o_a_u, n_o_a_d, n_o_b_u, n_o_b_d, KKT_error

