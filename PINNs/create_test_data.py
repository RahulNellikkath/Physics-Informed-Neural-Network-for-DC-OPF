import numpy as np
import pandas as pd
from PINNs.KKT import find_kkt_Lg

def create_test_data(simulation_parameters):
    
    n_buses = simulation_parameters['general']['n_buses']
    n_gbus = simulation_parameters['general']['n_gbus']
    n_line = simulation_parameters['general']['n_line']
    
    n_test_data_points = simulation_parameters['data_creation']['n_test_data_points']
    n_collocation = simulation_parameters['data_creation']['n_collocation']
    n_data_points = simulation_parameters['data_creation']['n_data_points']
    n_total = n_data_points + n_collocation

    Lg_Max=simulation_parameters['Lg_Max']
    
    L_Val=pd.read_csv('Data_File/'+str(n_buses)+'/NN_input.csv').to_numpy()[n_total:n_total+n_test_data_points][:]
    L_type_data = np.ones((n_test_data_points, 1))
    
    results = pd.read_csv('Data_File/'+str(n_buses)+'/NN_output.csv').to_numpy()[n_total:n_total+n_test_data_points][:]
    y_gen_data = np.array(results) 

    KKT_Lg=find_kkt_Lg(n_buses,L_Val,results)
    y_Lg = np.array(KKT_Lg)
    
    a_u_s=1
    a_u_e=n_gbus+1
    
    a_d_s=n_gbus+1
    a_d_e=2*n_gbus+1
    
    b_u_s=2*n_gbus+1
    b_u_e=2*n_gbus+n_line+1
    
    b_d_s=2*n_gbus+n_line+1
    b_d_e=2*n_gbus+2*n_line+1
    
    y_l_Lg  =y_Lg[:,0]/Lg_Max[0]
    y_a_u_Lg=y_Lg[:,a_u_s:a_u_e]/Lg_Max[1]
    y_a_d_Lg=y_Lg[:,a_d_s:a_d_e]/Lg_Max[2]
    y_b_u_Lg=y_Lg[:,b_u_s:b_u_e]/Lg_Max[3]
    y_b_d_Lg=y_Lg[:,b_d_s:b_d_e]/Lg_Max[4]

    
    x_test = [np.concatenate([L_Val], axis=0),
                  np.concatenate([L_type_data], axis=0)]

    y_test= [y_gen_data, y_l_Lg, y_a_u_Lg, y_a_d_Lg, y_b_u_Lg, y_b_d_Lg, np.zeros((n_test_data_points , 1))]

    np.savetxt('MILP_For_Worst_Case_Guarantees/Trained_Neural_Networks/case'+str(n_buses)+'_DCOPF/1/features_test.csv',L_Val, fmt='%s', delimiter=',')
    np.savetxt('MILP_For_Worst_Case_Guarantees/Trained_Neural_Networks/case'+str(n_buses)+'_DCOPF/1/labels_test.csv',y_gen_data, fmt='%s', delimiter=',')
    return x_test, y_test