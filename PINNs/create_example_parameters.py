import numpy as np
from numpy import genfromtxt
import pandas as pd

def create_example_parameters(n_buses: int):
    """
    creates a basic set of parameters that are used in the following processes:
    * data creation if measurements are to be simulated
    * setting up the neural network model
    * training procedure

    :param n_buses: integer number of buses in the system
    :return: simulation_parameters: dictonary that holds all parameters
    """

    if n_buses == 39:
        Gen = pd.read_csv('Data_File/39/Gen_39.csv',index_col=0)
        g_bus=Gen.index[Gen['Pg_max']!=0].to_numpy()
        n_gbus=len(g_bus)
        Pg_max_act=Gen['Pg_max_act'].to_numpy().reshape((1, n_gbus))
        Pg_max=Gen['Pg_max'].to_numpy().reshape((1, n_gbus))
        Pg_min=Gen['Pg_min'].to_numpy().reshape((1, n_gbus))
        C_Pg=Gen['C_Pg'].to_numpy().reshape((1, n_gbus))
        Map_g = np.zeros((n_gbus,n_buses))
        gen_no=0
        for g in g_bus:
            Map_g[gen_no][g-1]=1
            gen_no+=1
            
        Bus = pd.read_csv('Data_File/39/Bus_39.csv')
        l_bus=l_bus=Bus['Node'].to_numpy()
        n_lbus=len(l_bus)
        L_max=Bus['Pl'].to_numpy().reshape((1, n_lbus))
        Map_L = np.zeros((n_lbus,n_buses))
        l_no=0
        for l in l_bus:
            Map_L[l_no][l-1]=1
            l_no+=1
            
        branches = pd.read_csv('Data_File/39/branches_39.csv')
        n_line=branches.shape[0]
        fbus = branches['fbus'].to_numpy().reshape((n_line,1))
        tbus = branches['tbus'].to_numpy().reshape((n_line,1))        
        Pl_max=branches['branch_flowlimit'].to_numpy().reshape((1,n_line))
        Xline = branches['branch_x'].to_numpy().reshape((n_line,1))
        B= 1/ Xline

        Bline=np.zeros((n_line,n_buses))
        for i in range(n_line):
            Bline[i][fbus[i]-1] = B[i]
            Bline[i][tbus[i]-1] = -B[i]
        
        Bbus = np.zeros((n_buses,n_buses))
        for i in range(n_line):
            f=fbus[i][0]-1
            t=tbus[i][0]-1
            Bbus[f][f] = Bbus[f][f] + B[i]
            Bbus[f][t] = - B[i]
            Bbus[t][t] = Bbus[t][t] + B[i]
            Bbus[t][f] = - B[i]
        Bbus_inv = np.zeros((n_buses,n_buses))
        Bbus_inv[1:,1:] = np.linalg.inv(Bbus[1:,1:])
        PTDF=np.transpose(np.matmul(Bline,Bbus_inv))
    
    elif n_buses == 118:
        Gen = pd.read_csv('Data_File/118/Gen_118.csv',index_col=0)
        g_bus=Gen.index[Gen['Pg_max']!=0].to_numpy()
        n_gbus=len(g_bus)
        Pg_max_act=Gen['Pg_max_act'].to_numpy().reshape((1, n_gbus))
        Pg_max=Gen['Pg_max'].to_numpy().reshape((1, n_gbus))
        Pg_min=Gen['Pg_min'].to_numpy().reshape((1, n_gbus))
        C_Pg=Gen['C_Pg'].to_numpy().reshape((1, n_gbus))
        Map_g = np.zeros((n_gbus,n_buses))
        gen_no=0
        for g in g_bus:
            Map_g[gen_no][g-1]=1
            gen_no+=1
            
        Bus = pd.read_csv('Data_File/118/Bus_118.csv')
        l_bus=l_bus=Bus['Node'].to_numpy()
        n_lbus=len(l_bus)
        L_max=Bus['Pl'].to_numpy().reshape((1, n_lbus))
        Map_L = np.zeros((n_lbus,n_buses))
        l_no=0
        for l in l_bus:
            Map_L[l_no][l-1]=1
            l_no+=1
            
        branches = pd.read_csv('Data_File/118/branches_118.csv')
        n_line=branches.shape[0]
        fbus = branches['fbus'].to_numpy().reshape((n_line,1))
        tbus = branches['tbus'].to_numpy().reshape((n_line,1))        
        Pl_max=branches['branch_flowlimit'].to_numpy().reshape((1,n_line))
        Xline = branches['branch_x'].to_numpy().reshape((n_line,1))
        B= 1/ Xline
        PTDF=genfromtxt('Data_File/118/PTDF.csv', delimiter=',')
        
    elif n_buses == 162:
        Gen = pd.read_csv('Data_File/162/Gen_162.csv',index_col=0)
        g_bus=Gen.index[Gen['Pg_max']!=0].to_numpy()
        n_gbus=len(g_bus)
        Pg_max_act=Gen['Pg_max_act'].to_numpy().reshape((1, n_gbus))
        Pg_max=Gen['Pg_max'].to_numpy().reshape((1, n_gbus))
        Pg_min=Gen['Pg_min'].to_numpy().reshape((1, n_gbus))
        C_Pg=Gen['C_Pg'].to_numpy().reshape((1, n_gbus))
        Map_g = np.zeros((n_gbus,n_buses))
        gen_no=0
        for g in g_bus:
            Map_g[gen_no][g-1]=1
            gen_no+=1
            
        Bus = pd.read_csv('Data_File/162/Bus_162.csv')
        l_bus=l_bus=Bus['Node'].to_numpy()
        n_lbus=len(l_bus)
        L_max=Bus['Pl'].to_numpy().reshape((1, n_lbus))
        Map_L = np.zeros((n_lbus,n_buses))
        l_no=0
        for l in l_bus:
            Map_L[l_no][l-1]=1
            l_no+=1
            
        branches = pd.read_csv('Data_File/162/branches_162.csv')
        n_line=branches.shape[0]
        fbus = branches['fbus'].to_numpy().reshape((n_line,1))
        tbus = branches['tbus'].to_numpy().reshape((n_line,1))        
        Pl_max=branches['branch_flowlimit'].to_numpy().reshape((1,n_line))
        Xline = branches['branch_x'].to_numpy().reshape((n_line,1))
        B= 1/ Xline
        PTDF=genfromtxt('Data_File/162/PTDF.csv', delimiter=',')

   

    # -----------------------------------------------------------------------------------------------
    # general parameters of the power system that are assumed to be known in the identification process
    # n_buses: integer number of buses in the system
    # n_line: integer number of lines in the system
    # n_lbus: integer number of loads in the system
    # l_bus: vector with location of all the loads
    # L_max: vector with Max of all the loads
    # n_gbus: integer number of generators in the system
    # g_bus: vector with location of all the generators    
    # -----------------------------------------------------------------------------------------------

    
    general_parameters = {'n_buses': n_buses,
                          'n_line': n_line,
                          'n_lbus': n_lbus,
                          'l_bus': l_bus,
                          'L_max': L_max,
                          'n_gbus':n_gbus,
                          'g_bus': g_bus
                          
                          }
    # -----------------------------------------------------------------------------------------------
    # true parameters of the power system that are used in the PINN Layer
    # Pg_max: vector with Max possible active generation of all the genertors(Normalised)
    # Pg_min: vector with Min possible active generation of all the genertors(Normalised)
    # Pg_max_act: vector with Max possible active generation of all the genertors
    # Pl_max: vector with Max line active powerflow limits
    # PTDF: Power transmission distribution factors of the system
    # C_Pg: Active power generation cost vector
    # Map_g: Maping of generators in the system
    # Map_L: Maping of loads in the system
    # -----------------------------------------------------------------------------------------------
                          
    true_system_parameters = {'Pg_max': Pg_max,
                              'Pg_min': Pg_min,
                              'Pg_max_act': Pg_max_act,
                              'Pl_max': Pl_max,
                              'PTDF':PTDF,
                              'C_Pg':C_Pg,
                              'Map_g':Map_g,
                              'Map_L':Map_L}

    # -----------------------------------------------------------------------------------------------
    # parameters for the training data creation 
    # n_data_points: number of data points where measurements are present
    # n_collocation_points: number of points where the physics are evaluated at (additional to the data points)
    # n_test_data_points: number of unseen data points 
    # -----------------------------------------------------------------------------------------------
    n_data_points = 20000
    n_test_data_points=25000
    n_collocation_points = 50000

    data_creation_parameters = {'n_data_points': n_data_points,
                                'n_collocation': n_collocation_points,
                                'n_test_data_points': n_test_data_points}

    # -----------------------------------------------------------------------------------------------
    # parameters for the scheduled training process and the network architecture
    # epoch_schedule: number of epochs per batch size
    # batching_schedule: batch size
    # neurons_in_hidden_layers: number of neurons for each hidden layer
    # -----------------------------------------------------------------------------------------------
    n_total = n_data_points + n_collocation_points

    epoch_schedule = [500]

    batching_schedule = [int(np.ceil(n_total / 2))]
                         
    training_parameters = {'epoch_schedule': epoch_schedule,
                           'batching_schedule': batching_schedule,
                           'neurons_in_hidden_layers_Pg': [20,20,20],
                           'neurons_in_hidden_layers_Lm': [30,30,30]}

    # -----------------------------------------------------------------------------------------------
    # combining all parameters in a single dictionary
    # -----------------------------------------------------------------------------------------------
    simulation_parameters = {'true_system': true_system_parameters,
                             'general': general_parameters,
                             'data_creation': data_creation_parameters,
                             'training': training_parameters}

    return simulation_parameters
