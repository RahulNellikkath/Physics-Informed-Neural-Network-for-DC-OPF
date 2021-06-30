import numpy as np
import pandas as pd
from PINNs.KKT import find_kkt_Lg
from PINNs.Check_KKT import Check_kkt

def create_data(simulation_parameters):

    n_buses = simulation_parameters['general']['n_buses']
    n_gbus = simulation_parameters['general']['n_gbus']
    n_line = simulation_parameters['general']['n_line']
    n_collocation = simulation_parameters['data_creation']['n_collocation']
    n_data_points = simulation_parameters['data_creation']['n_data_points']
    n_total = n_data_points + n_collocation
    
    # Checking if KKT equations to find dual variables is solvable and removing datapoints for which KKT is not solvable
    Check_kkt(n_buses)
    #P_d or the Input data points
    NN_input=pd.read_csv('Data_File/'+str(n_buses)+'/NN_input.csv').to_numpy()
    L_Val=NN_input[0:n_data_points+n_collocation][:] 

    
    L_type_data = np.ones((n_data_points, 1))
    L_type_collocation = np.zeros((n_collocation, 1))

    #OPF output 
    NN_output=pd.read_csv('Data_File/'+str(n_buses)+'/NN_output.csv').to_numpy()
    results = NN_output[0:n_data_points][:]
    y_gen_data = results 
    y_gen_collocation = np.zeros((n_collocation, n_gbus))
    y_gen = np.concatenate([y_gen_data, y_gen_collocation], axis=0)
    
    #Finding the dual variables
    KKT_Lg=find_kkt_Lg(n_buses,L_Val[0:n_data_points][:],results)
    y_Lg_data = KKT_Lg
    y_Lg_collocation = np.zeros((n_collocation, 2*n_gbus+2*n_line+1))
    y_Lg = np.concatenate([y_Lg_data, y_Lg_collocation], axis=0)

    # assigning column numbers for each sets of dual variable.
    a_u_s=1
    a_u_e=n_gbus+1
    
    a_d_s=n_gbus+1
    a_d_e=2*n_gbus+1
    
    b_u_s=2*n_gbus+1
    b_u_e=2*n_gbus+n_line+1
    
    b_d_s=2*n_gbus+n_line+1
    b_d_e=2*n_gbus+2*n_line+1
    
    
    Lg_Max=[]
    
    #all the dual variables are normalised
    lg_max=np.max(y_Lg[:,0])+1 #adding one to avvoid division by zero when max of Lg is zero
    y_l_Lg  =y_Lg[:,0]/lg_max
    Lg_Max.append(lg_max)
    
    lg_max=np.max(y_Lg[:,a_u_s:a_u_e])+1
    y_a_u_Lg=y_Lg[:,a_u_s:a_u_e]/lg_max
    Lg_Max.append(lg_max)
    
    lg_max=np.max(y_Lg[:,a_d_s:a_d_e])+1
    y_a_d_Lg=y_Lg[:,a_d_s:a_d_e]/lg_max
    Lg_Max.append(lg_max)
    
    lg_max=np.max(y_Lg[:,b_u_s:b_u_e])+1
    y_b_u_Lg=y_Lg[:,b_u_s:b_u_e]/lg_max
    Lg_Max.append(lg_max)
    
    lg_max=np.max(y_Lg[:,b_d_s:b_d_e])+1
    y_b_d_Lg=y_Lg[:,b_d_s:b_d_e]/lg_max
    Lg_Max.append(lg_max)
    
    
    x_training = [L_Val,
                  np.concatenate([L_type_data, L_type_collocation], axis=0)]

    y_training = [y_gen, y_l_Lg , y_a_u_Lg , y_a_d_Lg , y_b_u_Lg, y_b_d_Lg, np.zeros((n_total, 1))]

    np.savetxt('MILP_For_Worst_Case_Guarantees/Trained_Neural_Networks/case'+str(n_buses)+'_DCOPF/1/features_train.csv',L_Val[0:n_data_points][:], fmt='%s', delimiter=',')
    np.savetxt('MILP_For_Worst_Case_Guarantees/Trained_Neural_Networks/case'+str(n_buses)+'_DCOPF/1/labels_train.csv',np.array(results), fmt='%s', delimiter=',')
    np.savetxt('MILP_For_Worst_Case_Guarantees/Trained_Neural_Networks/case'+str(n_buses)+'_DCOPF/1/NN_input.csv',NN_input, fmt='%s', delimiter=',')
    np.savetxt('MILP_For_Worst_Case_Guarantees/Trained_Neural_Networks/case'+str(n_buses)+'_DCOPF/1/NN_output.csv',NN_output, fmt='%s', delimiter=',')

    return x_training, y_training,Lg_Max
