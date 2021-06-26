import numpy as np
from PINNs.create_example_parameters import create_example_parameters
import pandas

def find_kkt_Lg(n_buses,P_Loads,P_Gens):
    
    simulation_parameters = create_example_parameters(n_buses)

    n_buses=simulation_parameters['general']['n_buses']
    n_line=simulation_parameters['general']['n_line']
    n_gbus=simulation_parameters['general']['n_gbus']    
    n_lbus=simulation_parameters['general']['n_lbus']

    Pg_min=simulation_parameters['true_system']['Pg_min']
    Pg_max=simulation_parameters['true_system']['Pg_max_act'] 
    C_Pg=simulation_parameters['true_system']['C_Pg']
    g_bus=simulation_parameters['general']['g_bus']
    Map_g=simulation_parameters['true_system']['Map_g']
    
    Pl_max=np.transpose(simulation_parameters['true_system']['Pl_max'])    
    Map_L=simulation_parameters['true_system']['Map_L']
    L_max= simulation_parameters['general']['L_max']
    
    PTDF=simulation_parameters['true_system']['PTDF']
    l_bus=simulation_parameters['general']['l_bus']

    Lg_val=[]
    P_Load_new = P_Loads
    P_Gen_new = P_Gens
    rem=0
    for l in range(np.size(P_Loads,0)):
        
        P_Load=np.zeros((1,n_lbus))
        P_Load = L_max*0.6 + np.multiply(P_Loads[l],L_max*0.4)
        
        P_Gen=np.zeros((1,n_gbus))
        P_Gen=np.multiply(P_Gens[l],Pg_max)

        # X= [lambda(1), alpha_up(n_gbus), alpha_dowm(n_gbus),
        #     beta_up(n_line), beta_down(n_line)]'
        
        lamda_start=0
        lambda_end=1
        
        alpha_u_start=1
        alpha_u_end=n_gbus+1
        
        alpha_d_start=n_gbus+1
        alpha_d_end=2*n_gbus+1
        
        beta_u_start=2*n_gbus+1
        beta_u_end=2*n_gbus+n_line+1
        
        beta_d_start=2*n_gbus+n_line+1
        beta_d_end=2*n_gbus+2*n_line+1
    
        A=np.zeros([2*n_gbus+2*n_line+1,2*n_gbus+2*n_line+1])
        B=np.zeros([2*n_gbus+2*n_line+1,1])
    
        #   Stationarity condition
        A[0:n_gbus,lamda_start:lambda_end]= np.ones(n_gbus).reshape((n_gbus,1))
        A[0:n_gbus,alpha_u_start:alpha_u_end]= np.diag(-np.ones(n_gbus))
        A[0:n_gbus,alpha_d_start:alpha_d_end]= np.diag(np.ones(n_gbus))
        A[0:n_gbus,beta_u_start:beta_u_end]=-PTDF[g_bus-1,:]
        A[0:n_gbus,beta_d_start:beta_d_end]=PTDF[g_bus-1,:]
        B[0:n_gbus,:]=np.transpose(C_Pg)[0:n_gbus,:]
        
        #  Complementary slackness condition for the generators not at their limit
        row=n_gbus
        for i in range(n_gbus):
            if Pg_max[0][i] - P_Gen[0][i] > 0 :
                A[row,alpha_u_start+i] = Pg_max[0][i] - P_Gen[0][i]
                row+=1
            if P_Gen[0][i] - Pg_min[0][i] > 0.0:
                A[row,alpha_d_start+i] = P_Gen[0][i] - Pg_min[0][i]
                row+=1
        #  Complementary slackness condition for the line flows not at their limit            
        for i in range(0,n_line):
            if Pl_max[i] - np.matmul((np.matmul(P_Gen,Map_g)-np.matmul(P_Load,Map_L)),PTDF)[0][i] > 0.1 or Pl_max[i] - np.matmul((np.matmul(P_Gen,Map_g)-np.matmul(P_Load,Map_L)),PTDF)[0][i]< -0.1:
                A[row,beta_u_start+i] = Pl_max[i] - np.matmul((np.matmul(P_Gen,Map_g)-np.matmul(P_Load,Map_L)),PTDF)[0][i]
                row+=1
            if np.matmul((np.matmul(P_Gen,Map_g)-np.matmul(P_Load,Map_L)),PTDF)[0][i] + Pl_max[i] > 0.1 or np.matmul((np.matmul(P_Gen,Map_g)-np.matmul(P_Load,Map_L)),PTDF)[0][i] + Pl_max[i] < -0.1 :
                A[row,beta_d_start+i] = np.matmul((np.matmul(P_Gen,Map_g)-np.matmul(P_Load,Map_L)),PTDF)[0][i] + Pl_max[i] 
                row+=1
        # Checking if number of equations equal to number of variables         
        if row == 2*n_gbus+2*n_line+1:
            Lg = np.linalg.solve(A,B)
            Lg_val.append(Lg.reshape(1,2*n_gbus+2*n_line+1)[0])
            rem=rem+1
        else :
            print(l)
    return Lg_val      