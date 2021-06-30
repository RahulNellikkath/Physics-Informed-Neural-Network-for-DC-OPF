% Compute worst-case guarantees on distance from predicted to optimal
% generation dispatch and on sub-optimality of cost function

close all;
clear all;

% add Gurobi to path
addpath(genpath('C:\gurobi902\win64'));
addpath(genpath('C:\Users\Andreas\Gurobi'));

cases =  {'case39_DCOPF'};

nr_cases = size(cases,1);

% Number of runs --> this is to rule out individual fluctuations
nr_iter = 1;

Time_Scalability = zeros(nr_cases,nr_iter);

delta = 0; % no input domain reduction

% Time to solve MILP to compute worst-case distance from predicted to
% optimal decision variables
v_dist_time_ = zeros(nr_cases,nr_iter);
%  worst-case distance from predicted to optimal decision variables
v_dist_wc_ = zeros(nr_cases,nr_iter);
% gen_ID of generator that corresponds to the worst-case distance from predicted to optimal decision variables
v_dist_ID_ = zeros(nr_cases,nr_iter);
% Time to solve MILP to compute worst-case sub-optimality
v_opt_time_ = zeros(nr_cases,nr_iter);
% Worst-case sub-optimality
v_opt_wc_= zeros(nr_cases,nr_iter);

success = zeros(nr_cases,nr_iter);

% maximum time for each MILP in seconds
time_MILP_max = 10^5;

tic();
for c=1:nr_cases %loop over cases
    
    for iter = 1:nr_iter% loop over runs
        iter
        mpc = eval(cases{c});
        
        path_input = strcat('.\Trained_Neural_Networks\',cases{c},'\',num2str(iter),'\');
        
        [v_info] = Compute_Worst_Case_Distance_Sub_Optimality(mpc,path_input,delta,time_MILP_max);
        
        v_info_{c,iter}=v_info;
        
        v_dist_time_(c,iter) = v_info.v_dist_time;
        v_dist_wc_(c,iter) = v_info.v_dist_wc;
        v_dist_ID_(c,iter) = v_info.v_dist_ID;
        v_opt_time_(c,iter) = v_info.v_opt_time;
        v_opt_wc_(c,iter) = v_info.v_opt_wc;
        success(c,iter)=1;
    end
    
end
toc();

save('Workspace_S6_Worst_Case_Distance_Optimality');
  
% worst-case distance from predicted to optimal decision variables in %
fprintf('worst-case distance from predicted to optimal decision variables in %% \n');
mean(v_dist_wc_,2).*100

% Worst-case line constraint violation for defined input domain
fprintf('worst-case sub-optimality in %% \n');
mean(v_opt_wc_,2).*100



