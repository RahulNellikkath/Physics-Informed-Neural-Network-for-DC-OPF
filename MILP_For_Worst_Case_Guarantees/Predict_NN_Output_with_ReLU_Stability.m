function pg_pred = Predict_NN_Output_with_ReLU_Stability(pd_NN,W_input,bias,W,W_output,ReLU_layers,ReLU_stability_active,ReLU_stability_inactive)
% compute the neural network prediction considering ReLU stability
zk_hat = W_input*(pd_NN.') + bias{1};
zk = max(zk_hat,0);
% always active ReLUs are the identity function
zk(squeeze(ReLU_stability_active(1,1,:)),1) =  zk_hat(squeeze(ReLU_stability_active(1,1,:)),1);
% always inactive ReLUs are always zero
zk(squeeze(ReLU_stability_inactive(1,1,:)),1) =  0;
for j = 1:ReLU_layers-1
    zk_hat = W{j}*zk + bias{j+1};
    zk = max(zk_hat,0);
    % always active ReLUs are the identity function
    zk(squeeze(ReLU_stability_active(1,j+1,:)),1) =  zk_hat(squeeze(ReLU_stability_active(1,j+1,:)),1);
    % always inactive ReLUs are always zero
    zk(squeeze(ReLU_stability_inactive(1,j+1,:)),1) =  0;
end
pg_pred = W_output*zk + bias{end};
end