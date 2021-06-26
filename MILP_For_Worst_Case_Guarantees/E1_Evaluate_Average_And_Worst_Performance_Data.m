% Evaluate the average performance on the testing dataset and the
% worst-case performance on the entire dataset
% The atter serves as empirical lower bound to the exact worst-case
% performance over the entire input domain
close all;
clear all;

% define named indices into data matrices
[PQ, PV, REF, NONE, BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, ...
    VA, BASE_KV, ZONE, VMAX, VMIN, LAM_P, LAM_Q, MU_VMAX, MU_VMIN] = idx_bus;
[GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN, ...
    MU_PMAX, MU_PMIN, MU_QMAX, MU_QMIN, PC1, PC2, QC1MIN, QC1MAX, ...
    QC2MIN, QC2MAX, RAMP_AGC, RAMP_10, RAMP_30, RAMP_Q, APF] = idx_gen;
[F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, ...
    TAP, SHIFT, BR_STATUS, PF, QF, PT, QT, MU_SF, MU_ST, ...
    ANGMIN, ANGMAX, MU_ANGMIN, MU_ANGMAX] = idx_brch;
[PW_LINEAR, POLYNOMIAL, MODEL, STARTUP, SHUTDOWN, NCOST, COST] = idx_cost;

cases =  {'case39_DCOPF'};

nr_cases = size(cases,1);

% Number of runs --> this is to rule out individual fluctuations
nr_iter = 1;

% OUTPUT FOR TABLE: AVERAGE PERFORMANCE ON TEST SET
% mean absolute error (test | train | entire dataset)
MAE_ = zeros(nr_cases,nr_iter,3);
% maximum generator constraint violation
v_g_ = zeros(nr_cases,nr_iter,3);
% maximum line constraint violation
v_line_ = zeros(nr_cases,nr_iter,3);
% distance of predicted to optimal generator set-points
v_dist_ = zeros(nr_cases,nr_iter,3);
% sub-optimality
v_opt_ = zeros(nr_cases,nr_iter,3);

% OUTPUT FOR TABLE: WORST-CASE GUARANTEES OF PHYSICAL CONSTRAINT VIOLATIONS
% worst-case maximum generator constraint violation on entire data set
v_g_wc_data = zeros(nr_cases,nr_iter,3);
%  worst-case  maximum line constraint violation on entire data set
v_line_wc_data = zeros(nr_cases,nr_iter,3);

% OUTPUT FOR TABLE: WORST-CASE GUARANTEES OF DISTANCE OF DECISION VARIABLES AND SUB-OPTIMALITY
% worst-case distance of predicted to optimal generator set-points on entire data set
v_dist_wc_data = zeros(nr_cases,nr_iter,3);
% worst-case  sub-optimality  on entire data set
v_opt_wc_data = zeros(nr_cases,nr_iter,3);

tElapse = tic();
for c=1:nr_cases %loop over cases
    
    for iter = 1:nr_iter % loop over runs
        
        mpc = eval(cases{c});
        
        path_input = strcat('.\Trained_Neural_Networks\',cases{c},'\',num2str(iter),'\');
        
        nb = size(mpc.bus,1);
        ng = size(mpc.gen,1);
        nl = size(mpc.branch,1);
        
        % identify the loads which are non-zero
        ID_loads = find(mpc.bus(:,PD)~=0);
        
        nloads = size(ID_loads,1);
        %map from loads to buses
        M_d = zeros(nb,nloads);
        for i = 1:nloads
            M_d(ID_loads(i),i) = 1;
        end
        
        %map from generators to buses
        M_g = zeros(nb,ng);
        ID_gen = mpc.gen(:,GEN_BUS);
        for i = 1:ng
            M_g(ID_gen(i),i) = 1;
        end
        
        pd_max = mpc.bus(ID_loads,PD);
        pd_min =  pd_max.*0.6;
        pd_delta = pd_max.*0.4;
        
        mpopt = mpoption;
        mpopt.out.all = 0;
        
        pg_delta = mpc.gen(1:end,PMAX)-mpc.gen(1:end,PMIN);
        
        % Load the neural network weights and biases
        W_input = csvread(strcat(path_input,'W_0.csv')).';
        W_output = csvread(strcat(path_input,'W_3.csv')).'; % not clear how the indexing works here (going from layer 1 to layer 2)
        W{1} = csvread(strcat(path_input,'W_1.csv')).';
        W{2} = csvread(strcat(path_input,'W_2.csv')).';
        bias{1} = csvread(strcat(path_input,'b_0.csv')); %net.b; % bias
        bias{2} = csvread(strcat(path_input,'b_1.csv'));
        bias{3} = csvread(strcat(path_input,'b_2.csv'));
        bias{4} = csvread(strcat(path_input,'b_3.csv'));
        
        Input_NN = csvread(strcat(path_input,'NN_input.csv'));
        Output_NN = csvread(strcat(path_input,'NN_output.csv'));
        
        ReLU_layers = 3;
        nr_neurons = 20;
        
        % how many of the ReLUs are always on/off --> fix those variables in the
        % formulation
        % loop over all training and test data
        Nr_samples = size(Input_NN,1);
        ReLU_stability = ones(Nr_samples,ReLU_layers,nr_neurons);
        TOL_ReLU = 0.0;
        for i = 1:Nr_samples
            pd_NN = Input_NN(i,:);
            zk_hat = W_input*(pd_NN.') + bias{1};
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
        
        fprintf('The share of always active ReLUs: %5.2f %% \n', sum(sum(ReLU_stability_active))/(ReLU_layers*nr_neurons)*100)
        fprintf('The share of always inactive ReLUs: %5.2f %% \n', sum(sum(ReLU_stability_inactive))/(ReLU_layers*nr_neurons)*100)
        %y_pred_ReLU = max(y_pred_ReLU,0);
        
        
        for i = 1:3
            
            
            mpc = eval(cases{c});
            
            % load data from Python
            if i == 1
                % test data
                Input_NN = csvread(strcat(path_input,'features_test.csv'));
                Output_NN = csvread(strcat(path_input,'labels_test.csv'));
            elseif i == 2
                % training data
                Input_NN = csvread(strcat(path_input,'features_train.csv'));
                Output_NN = csvread(strcat(path_input,'labels_train.csv'));
            elseif i == 3
                % all data
                Input_NN = csvread(strcat(path_input,'NN_input.csv'));
                Output_NN = csvread(strcat(path_input,'NN_output.csv'));
            end
            
            
            Nr_samples = size(Input_NN,1);
            Output_pred = zeros(Nr_samples,size(Output_NN,2));
            for j = 1:Nr_samples
                pd_NN = Input_NN(j,:);
                Output_pred(j,:) = Predict_NN_Output_with_ReLU_Stability(pd_NN,W_input,bias,W,W_output,ReLU_layers,ReLU_stability_active,ReLU_stability_inactive);
                
                Output_pred_ReLU = Predict_NN_Output(pd_NN,W_input,bias,W,W_output,ReLU_layers);
                
                % here we check that the ReLU stability does not impact the neural
                % network prediction on the entire dataset
                % this is just a double check
                if max(abs(Output_pred_ReLU-Output_pred(j,:).'))>10^-4
                    error('error_1')
                end
            end
            
            MAE_(c,iter,i) = mean(mean(abs(Output_pred - Output_NN)));
            v_dist_(c,iter,i) = mean(max(abs(Output_pred - Output_NN),[],2));
            v_dist_wc_data(c,iter,i) = max(max(abs(Output_pred - Output_NN),[],2));
            % compare the cost (and compute sub-optimality)
            % compute the total demand
            cost_pred = zeros(Nr_samples,1);
            cost_true = zeros(Nr_samples,1);
            
            mpopt = mpoption;
            mpopt.out.all = 0;
            results_dcopf= rundcopf(mpc,mpopt);
            cost_ref =results_dcopf.f;
            for j = 1:Nr_samples
                pd_sum = (Input_NN(j,:).').*pd_delta+pd_min;
                pg_pred = (Output_pred(j,:).').*(pg_delta);
                pg_true = (Output_NN(j,:).').*(pg_delta);

                cost_pred(j,1) = sum(pg_pred.*mpc.gencost(:,6));
                cost_true(j,1) = sum(pg_true.*mpc.gencost(:,6));
            end
            
            v_opt_(c,iter,i) = mean((cost_pred-cost_true)./cost_ref);
            v_opt_wc_data(c,iter,i) = max((cost_pred-cost_true)./cost_ref);
            
            % compute the constraint violations
            pg_up_viol = zeros(Nr_samples,ng);
            pg_down_viol = zeros(Nr_samples,ng);
            pline_viol = zeros(Nr_samples,nl);
            
            pline_max = mpc.branch(:,RATE_A)./mpc.baseMVA;
            pg_min = mpc.gen(:,PMIN)./mpc.baseMVA;
            pg_max = mpc.gen(:,PMAX)./mpc.baseMVA;
            
            % get relevant bus and line admittance matrices
            mpc.bus(ID_loads,PD) = pd_min + (Input_NN(j,:).').*pd_delta;
            [B, Bf, Pbusinj, Pfinj] = makeBdc(mpc.baseMVA, mpc.bus, mpc.branch);
            if sum(abs(Pbusinj))+sum(abs(Pfinj)) > 10^-4
                error('Code cannot handle add. injections');
            end
            % compute inverse of bus admittance matrix; removing slack
            % bus --> first generator bus
            slack_bus = mpc.gen(1,GEN_BUS);
            B_red = B;
            % remove row and column corresponding to slack bus
            B_red(slack_bus,:) = [];
            B_red(:,slack_bus) = [];
            B_inv = inv(B_red);
            mpc_original = mpc;
            for j = 1:Nr_samples
                mpc = mpc_original;
                
                % compute slack bus dispatch
                %pg_slack= (sum(pd_min)+Input_NN(j,:)*pd_delta-Output_pred(j,:)*pg_delta)./mpc.baseMVA;
                
                % nodal power injections without slack bus
                pg = ((Output_pred(j,:).').*(pg_delta)/mpc.baseMVA);
                pd = (pd_min/mpc.baseMVA + (Input_NN(j,:).').*pd_delta/mpc.baseMVA);
                pinj = M_g * pg - M_d * pd;
                pinj_woslack = pinj;
                pinj_woslack(slack_bus) =[];
                % compute DC power line flow
                theta_woslack = B_inv*pinj_woslack;
                % insert 0 at the slack bus position
                theta = zeros(nb,1);
                theta(1:slack_bus-1,1) = theta_woslack(1:slack_bus-1,1);
                theta(slack_bus,1) = 0;
                theta(slack_bus+1:nb,1) = theta_woslack(slack_bus:end,1);
                % compute the line flow
                pline=Bf*theta;
                
                
                % compute line flow violation
                pline_viol(j,:) = max(max(pline-pline_max,0),max(-pline_max-pline,0));
                % compute generator violations
                pg_cur = (Output_pred(j,:).').*pg_delta./mpc.baseMVA;
                pg_up_viol(j,:) = max(pg_cur-pg_max,0);
                pg_down_viol(j,:) = max(pg_min-pg_cur,0);
                
                % For debugging:
                % random comparison to the results of dc opf
                % compare on expectation every 50st sample
                if rand(1)>=0.98
                    % set the genreator set points
                    mpc.gen(1:end,PG) = mpc.gen(1:end,PMIN)+(Output_pred(j,:).').*(pg_delta);
                    % set the active power demand
                    mpc.bus(ID_loads,PD) = pd_min + (Input_NN(j,:).').*pd_delta;
                    % rundcpf
                    results_dcpf = rundcpf(mpc,mpopt);
                    
                    % evaluate line flow
                    pline_dcpf = results_dcpf.branch(:,PF)./mpc.baseMVA;
                    % evaluate generator dispatch
                    pg_dcpf = results_dcpf.gen(:,PG)./mpc.baseMVA;
                    % if that does not match throw error
                    if (max(abs(pline_dcpf-pline))+max(abs(pg_dcpf-pg_cur)))>10
                        max(abs(pline_dcpf-pline))+max(abs(pg_dcpf-pg_cur))
                        error('Mismatch in constraint violation computation')
                    end
                end
                
                
            end
            v_line_(c,iter,i) = mean(max(pline_viol,[],2));
            v_g_(c,iter,i) = mean(max([pg_up_viol pg_down_viol],[],2));
            v_line_wc_data(c,iter,i) = max(max(pline_viol,[],2));
            v_g_wc_data(c,iter,i) = max(max([pg_up_viol pg_down_viol],[],2));
            
        end
        
    end
end
tElapsed = toc(tElapse)

% OUTPUT FOR TABLE: AVERAGE PERFORMANCE ON TEST SET

% average mean absolute error on test set (percent)
fprintf('average mean absolute error on test set  MAE  (percent)\n');
mean(MAE_(:,:,1),2)*100
% average maximum generator constraint violation on test set (MW)
fprintf('average maximum generator constraint violation on test set v_g (MW) \n');
mean(v_g_(:,:,1),2)*100
% average maximum line constraint violation on test set (MW)
fprintf('average maximum line constraint violation on test set v_line (MW)\n');
mean(v_line_(:,:,1),2)*100
% average distance of predicted to optimal generator set-points on test set
% (percent)
fprintf('average distance of predicted to optimal generator set-points on test set v_dist \n');
mean(v_dist_(:,:,1),2)*100
% average sub-optimality on test set (percent)
fprintf('average sub-optimality on test set (percent) v_opt \n');
mean(v_opt_(:,:,1),2) *100


% constraint violations w.r.t total maximum system loading
sum_p_d_max_ = zeros(nr_cases,1);

for c = 1:nr_cases
    mpc = eval(cases{c});
    sum_p_d_max_(c,1) = sum(mpc.bus(:,PD));
end

% average largest violation of generator limits normalized by total system
% loading
fprintf('average largest violation of generator limits normalized by total system loading(%%) \n');
max(mean(v_g_(:,:,1),2)*100./sum_p_d_max_)*100
% average largest violation of line limits
fprintf('average largest violation of line limits normalized by total system loading(%%) \n');
max(mean(v_line_(:,:,1),2)*100./sum_p_d_max_)*100


% OUTPUT FOR TABLE: WORST-CASE GUARANTEES OF PHYSICAL CONSTRAINT VIOLATIONS
% worst-case maximum generator constraint violation on entire data set (MW)
fprintf('worst-case maximum generator constraint violation on entire data set (MW) \n')
mean(v_g_wc_data(:,:,3),2)*100
%  worst-case  maximum line constraint violation on entire data set (MW)
fprintf('worst-case maximum line constraint violation on entire data set (MW) \n')
mean(v_line_wc_data(:,:,3),2)*100

% OUTPUT FOR TABLE: WORST-CASE GUARANTEES OF DISTANCE OF DECISION VARIABLES AND SUB-OPTIMALITY
% worst-case distance of predicted to optimal generator set-points on
% entire data set (percent)
fprintf('worst-case distance of predicted to optimal generator set-points on entire data set (percent) \n')
mean(v_dist_wc_data(:,:,3),2)*100
% worst-case  sub-optimality  on entire data set (percent)
fprintf('worst-case  sub-optimality  on entire data set (percent) \n')
mean(v_opt_wc_data(:,:,3),2)*100


save('Workspace_S3_Evaluate_Average_And_Worst_Performance_Data')