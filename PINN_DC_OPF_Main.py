import numpy as np
import time
# import tensorflow as tf
# from tensorflow import keras

from PINNs.create_example_parameters import create_example_parameters
from PINNs.create_data import create_data
from PINNs.PinnModel import PinnModel
from PINNs.create_test_data import create_test_data


def PINN_DC_OPF():
    
    n_buses=118
    
    W_Lm=0.05
    W_PINN=0.05
    
    simulation_parameters = create_example_parameters(n_buses)

    x_training, y_training,Lg_Max = create_data(simulation_parameters=simulation_parameters)

    simulation_parameters.update({'Lg_Max':Lg_Max})
    
    x_test, y_test = create_test_data(simulation_parameters=simulation_parameters)


    model = PinnModel(W_Lm,W_PINN,simulation_parameters=simulation_parameters)

    np.set_printoptions(precision=3)
    print('Starting training')
    total_start_time = time.time()

    for n_epochs, batch_size in zip(simulation_parameters['training']['epoch_schedule'],
                                simulation_parameters['training']['batching_schedule']):

        epoch_start_time = time.time()
        model.fit(x_training,
                  y_training,
                  epochs=n_epochs,
                  batch_size=batch_size,
                  verbose=0,
                  shuffle=True)
    epoch_end_time = time.time()

    results = model.evaluate(x_test, y_test, verbose=0)
    
    y_pred=model.predict(x_test)
    mae=np.sum(np.absolute(y_test[0]-y_pred[0]))*100/np.sum(y_test[0])
    

    #os.makedirs('Test_output/39/L')
    for j in range(0, 4):
        weights = model.get_weights()[2*j]
        biases = model.get_weights()[2*j+1]
        np.savetxt('MILP_For_Worst_Case_Guarantees/Trained_Neural_Networks/case'+str(n_buses)+'_DCOPF/1'+'/W_'+str(j)+'.csv',weights, fmt='%s', delimiter=',')
        np.savetxt('MILP_For_Worst_Case_Guarantees/Trained_Neural_Networks/case'+str(n_buses)+'_DCOPF/1'+'/b_'+str(j)+'.csv',biases, fmt='%s', delimiter=',')


    print("Test Data Loss (MAE)", mae)
    # print(results)
    total_end_time = time.time()
    print(f'Total training time: {total_end_time - total_start_time:.1f} seconds')


if __name__ == "__main__":
    PINN_DC_OPF()