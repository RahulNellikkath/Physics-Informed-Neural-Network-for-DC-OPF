function pg_pred = Predict_NN_Output(pd_NN,W_input,bias,W,W_output,ReLU_layers)
% compute the neural network prediction
zk_hat = W_input*(pd_NN.') + bias{1};
zk = max(zk_hat,0);
for j = 1:ReLU_layers-1
    zk_hat = W{j}*zk + bias{j+1};
    zk = max(zk_hat,0);
end
pg_pred = W_output*zk + bias{end};
end