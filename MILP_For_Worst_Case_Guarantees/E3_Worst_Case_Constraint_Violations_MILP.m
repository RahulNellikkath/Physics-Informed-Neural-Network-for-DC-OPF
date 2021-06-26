% Compute the exact worst-case constraint violations for generator and line
% violations
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
nr_neurons = 20; % number of neurons for each hidden layer

% Time to solve MILP to compute worst-case line constraint violation
v_line_time_ = zeros(nr_cases,nr_iter);
% Worst-case line constraint violation for defined input domain
v_line_wc_ = zeros(nr_cases,nr_iter);
% Branch_ID of line that corresponds to the worst_case line violation
v_line_ID_ = zeros(nr_cases,nr_iter);
% Time to solve MILP to compute worst-case generator constraint violation
v_g_time_ = zeros(nr_cases,nr_iter);
% Worst-case generator constraint violation for defined input domain
v_g_wc_= zeros(nr_cases,nr_iter);
% gen_ID of generator that corresponds to the worst_case generator violation
v_g_ID_= zeros(nr_cases,nr_iter);

tic();
for c=1:nr_cases %1:nr_cases %loop over cases
    
    for iter = 1:nr_iter % loop over runs
        
        mpc = eval(cases{c});
        
        path_input = strcat('.\Trained_Neural_Networks\',cases{c},'\',num2str(iter),'\');
        
        
        [v_info,Pl_value] = Compute_Worst_Case_Constraint_Violations(mpc,path_input,delta,nr_neurons);
        
        v_info_{c,iter} = v_info;
        v_line_time_(c,iter) = v_info.v_line_time;
        v_line_wc_(c,iter) = v_info.v_line_wc;
        v_line_ID_(c,iter) = v_info.v_line_ID;
        v_g_time_(c,iter) = v_info.v_g_time;
        v_g_wc_(c,iter) = v_info.v_g_wc;
        v_g_ID_(c,iter) = v_info.v_g_ID;
    end
    
end
toc();

dataset_folder = strcat(pwd,'/Data_Sets/case',num2str(39),'_DCOPF/');
csvwrite(strcat(dataset_folder,'New_input.csv'),Pl_value);

save('Workspace_S5_Worst_Case_Constraint_Violations_MILP');

% constraint violations w.r.t total maximum system loading
sum_p_d_max_ = zeros(nr_cases,1);

for c = 1:nr_cases
    mpc = eval(cases{c});
    sum_p_d_max_(c,1) = sum(mpc.bus(:,3));
end

% Worst-case generator constraint violation for defined input domain
fprintf('Worst-case generator constraint violation for defined input domain (MW) \n');
mean(v_g_wc_,2)

% Worst-case line constraint violation for defined input domain
fprintf('Worst-case line constraint violation for defined input domain (MW) \n');
mean(v_line_wc_,2)

% compute violations w.r.t system loading
fprintf('compute worst-case violations w.r.t maximum system loading (%%) \n')
mean(v_g_wc_,2)./sum_p_d_max_*100
mean(v_line_wc_,2)./sum_p_d_max_*100
fprintf('compute average worst-case violations w.r.t maximum system loading (%%) \n')
% compute average violations w.r.t system loading
mean([mean(v_g_wc_,2)./sum_p_d_max_*100;   mean(v_line_wc_,2)./sum_p_d_max_*100])

