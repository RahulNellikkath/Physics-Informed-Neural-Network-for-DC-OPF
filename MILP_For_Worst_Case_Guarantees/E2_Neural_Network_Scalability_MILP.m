% Compute the tightening of the bounds on the MILP formulation
% and the ReLU stability
% First: ReLU stability: eliminate inactive and fix active ReLUs
% Second: Interval Arithmetic to tighten bounds
% Third: LP relaxation to tighten bounds
% Fourth: Full MILP to tighten bounds

close all;
clear all;

% add Gurobi to path
addpath(genpath('C:\gurobi902\win64'));

cases =  {'case39_DCOPF'};

nr_cases = size(cases,1);

% Number of runs --> this is to rule out individual fluctuations
nr_iter = 1;

Time_Scalability = zeros(nr_cases,nr_iter);
Share_active_ReLUs = zeros(nr_cases,nr_iter);
Share_inactive_ReLUs = zeros(nr_cases,nr_iter);

for c=1:nr_cases %loop over cases
    
    for iter = 1:nr_iter % loop over runs
        tElapsed=tic();
        mpc = eval(cases{c});
        
        path_input = strcat('.\Trained_Neural_Networks\',cases{c},'\',num2str(iter),'\');
        
        
        % Load the neural network weights and biases
        W_input = csvread(strcat(path_input,'W_0.csv')).';
        W_output = csvread(strcat(path_input,'W_3.csv')).'; % not clear how the indexing works here (going from layer 1 to layer 2)
        W{1} = csvread(strcat(path_input,'W_1.csv')).';
        W{2} = csvread(strcat(path_input,'W_2.csv')).';
        bias{1} = csvread(strcat(path_input,'b_0.csv')); %net.b; % bias
        bias{2} = csvread(strcat(path_input,'b_1.csv'));
        bias{3} = csvread(strcat(path_input,'b_2.csv'));
        bias{4} = csvread(strcat(path_input,'b_3.csv'));
        
        % Load the entire neural network dataset
        Input_NN = csvread(strcat(path_input,'NN_input.csv'));
        Output_NN = csvread(strcat(path_input,'NN_output.csv'));
        
        % number of hidden layers
        ReLU_layers = 3;
        % number of neurons per layer
        nr_neurons = 20;
        
        % compute the ReLU activativity
        
        % how many of the ReLUs are always on/off --> fix those variables in the
        % formulation
        % loop over all training and test data
        Nr_samples = size(Input_NN,1);
        ReLU_stability = ones(Nr_samples,ReLU_layers,nr_neurons);
        TOL_ReLU = 0;
        for i = 1:Nr_samples
            zk_hat = W_input*(Input_NN(i,:).') + bias{1};
            zk = max(zk_hat,0);
            ReLU_stability(i,1,:) = zk>TOL_ReLU;
            for j = 1:ReLU_layers-1
                zk_hat = W{j}*zk + bias{j+1};
                zk = max(zk_hat,0);
                ReLU_stability(i,j+1,:) = zk>TOL_ReLU;
            end
        end
        
        % all ReLUs that are always active
        ReLU_stability_active=sum(ReLU_stability,1)==Nr_samples;
        % all ReLUs that are always inactive
        ReLU_stability_inactive=sum(ReLU_stability,1)==0;
        
        
        share_active = sum(sum(ReLU_stability_active))/(ReLU_layers*nr_neurons)*100;
        share_inactive = sum(sum(ReLU_stability_inactive))/(ReLU_layers*nr_neurons)*100;
        fprintf('The share of always active ReLUs: %5.2f %% \n', share_active)
        fprintf('The share of always inactive ReLUs: %5.2f %% \n', share_inactive)
        Share_active_ReLUs(c,iter) = share_active;
        Share_inactive_ReLUs(c,iter) = share_inactive;

        
        % options for post-processing of neural network
        interval_arithmetic = true;
        
        zk_hat_max = ones(nr_neurons,1,ReLU_layers)*(10000);% upper bound on zk_hat (Here we will need to use some bound tightening)
        zk_hat_min = ones(nr_neurons,1,ReLU_layers)*(-10000);% lower bound on zk_hat (Here we will need to use some bound tightening)
        
        if interval_arithmetic == true
            % use interval arithmetic to compute tighter bounds
            % initial input bounds
            u_init = ones(size(Input_NN,2),1);
            l_ini = zeros(size(Input_NN,2),1);
            zk_hat_max(:,1,1) = max(W_input,0)*u_init+min(W_input,0)*l_ini+bias{1};
            zk_hat_min(:,1,1) = min(W_input,0)*u_init+max(W_input,0)*l_ini+bias{1};
            for j = 1:ReLU_layers-1
                zk_hat_max(:,1,j+1) = max(W{j},0)*max(zk_hat_max(:,1,j),0)+min(W{j},0)*max(zk_hat_min(:,1,j),0)+bias{j+1};
                zk_hat_min(:,1,j+1) = min(W{j},0)*max(zk_hat_max(:,1,j),0)+max(W{j},0)*max(zk_hat_min(:,1,j),0)+bias{j+1};
            end
            
        end
        
        fprintf('\n Tightening of the ReLU bounds \n')
        
        zk_hat_max_cur = zk_hat_max;
        zk_hat_min_cur = zk_hat_min;
        
        for run = 1:2
            LP_tightening = true;
            % tighten the bounds of each individual RELU
            % MILP_tightening
            if LP_tightening == true
                tic();
                % build relaxed/MILP optimization problem
                size_input = size(Input_NN,2);
                
                if run == 1
                    % first solve LP relaxation
                    LP_relax = true;
                else
                    % second solve full MILP formulation
                    LP_relax = false;
                end
                % construct otpimization problem of neural network
                pd_NN = sdpvar(size_input,1);
                if LP_relax == true %integer relaxation
                    ReLU = sdpvar(nr_neurons,1,ReLU_layers);
                else
                    ReLU = binvar(nr_neurons,1,ReLU_layers);
                end
                
                zk_hat = sdpvar(nr_neurons,1,ReLU_layers);
                zk = sdpvar(nr_neurons,1,ReLU_layers);
                
                constr_tightening = [];
                %input restrictions
                constr_tightening = [constr_tightening;...
                    0.0 <= pd_NN <= 1.0];
                %input layer
                constr_tightening = [constr_tightening; ...
                    zk_hat(:,:,1) == W_input*pd_NN + bias{1}];
                
                for i = 1:ReLU_layers-1
                    constr_tightening = [constr_tightening; ...
                        zk_hat(:,:,i+1) == W{i}*zk(:,:,i) + bias{i+1}];
                end
                if LP_relax == true %integer relaxation
                    % % integer relaxation
                    constr_tightening = [constr_tightening; ...
                        0<= ReLU <=1 ];
                end
                
                % ReLU stability
                
                for k = 2:ReLU_layers
                    for m = 1:nr_neurons
                        constr_tightening_cur = constr_tightening;
                        
                        for i = 1:ReLU_layers
                            for jj = 1:nr_neurons
                                
                                if ReLU_stability_active(1,i,jj) == 1
                                    % this RELU is assumed to be stable and active
                                    constr_tightening_cur = [constr_tightening_cur; ...
                                        zk(jj,1,i) == zk_hat(jj,1,i)];
                                elseif ReLU_stability_inactive(1,i,jj) == 1
                                    % this RELU is assumed to be stable and inactive
                                    constr_tightening_cur = [constr_tightening_cur; ...
                                        zk(jj,1,i) == 0];
                                else
                                    
                                    % ReLU (rewriting the max function)
                                    constr_tightening_cur = [constr_tightening_cur; ...
                                        zk(jj,1,i) <= zk_hat(jj,1,i) - zk_hat_min_cur(jj,1,i).*(1-ReLU(jj,1,i));...1
                                        zk(jj,1,i) >= zk_hat(jj,1,i);...
                                        zk(jj,1,i) <= zk_hat_max_cur(jj,1,i).*ReLU(jj,1,i);...
                                        zk(jj,1,i) >= 0];
                                end
                            end
                        end
                        
  
                        % solve for lower bound, i.e. minimize
                        obj = zk_hat(m,1,k);
                        options = sdpsettings('solver','gurobi','verbose',0);
                        
                        diagnostics = optimize(constr_tightening_cur,obj,options);
                        if diagnostics.problem ~= 0 
                            error('some issue with solving MILP 1');
                        else
                            zk_hat_min_cur(m,1,k)=min(value(zk_hat(m,1,k)), zk_hat_max_cur(m,1,k)-10^-3);
                        end
                        
                        
                        % solve for upper bound, i.e. maximize
                        obj = -zk_hat(m,1,k);
                        diagnostics = optimize(constr_tightening_cur,obj,options);
                        if diagnostics.problem ~= 0 
                            error('some issue with solving MILP 2');
                        else 
                            % we need to avoid having both 0 because then we get numerical
                            % problems
                            zk_hat_max_cur(m,1,k)=max(value(zk_hat(m,1,k)),zk_hat_min_cur(m,1,k)+10^-3);
                        end
                        
                    end
                end
                toc();
            end
            
            % this shows by how much we reduced the bounds 
            mean((zk_hat_max_cur-zk_hat_min_cur)./(zk_hat_max-zk_hat_min))
            zk_hat_max = zk_hat_max_cur;
            zk_hat_min = zk_hat_min_cur;
        end
        
        fprintf('\n Tightening of the ReLU bounds finished \n')
        
        save(strcat(path_input,'zk_hat_min'),'zk_hat_min');
        save(strcat(path_input,'zk_hat_max'),'zk_hat_max');
        save(strcat(path_input,'ReLU_stability_inactive'),'ReLU_stability_inactive');
        save(strcat(path_input,'ReLU_stability_active'),'ReLU_stability_active');
        
        Time_Scalability(c,iter)=toc(tElapsed);
    end
end
% average time in minutes
fprintf('Averaged time to compute the tightened bounds for the MILP reformulation of the trained neural network\n')
mean(mean(Time_Scalability./60))
% average share of inactive ReLUs
fprintf('average share of inactive ReLUs\n')
mean(mean(Share_active_ReLUs))
% average share of active ReLUs
fprintf('average share of active ReLUs\n')
mean(mean(Share_inactive_ReLUs))

save('Workspace_S4_Neural_Network_Scalability_MILP')

