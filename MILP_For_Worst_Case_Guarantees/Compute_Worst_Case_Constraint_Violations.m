function [v_info,Pl_values] = Compute_Worst_Case_Constraint_Violations(mpc,path_input,delta,nr_neurons)

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


nb = size(mpc.bus,1);
ng = size(mpc.gen,1);
nline = size(mpc.branch,1);

% identify the loads which are non-zero
ID_loads = find(mpc.bus(:,PD)~=0);
nd=size(ID_loads,1);
pd_max = mpc.bus(ID_loads,PD);

%map from buses to loads
M_d = zeros(nb,nd);
for i = 1:nd
    M_d(ID_loads(i),i) = 1;
end

%map from generators to buses
M_g = zeros(nb,ng);
ID_gen = mpc.gen(:,GEN_BUS);
for i = 1:ng
    M_g(ID_gen(i),i) = 1;
end

% Here we assume that the loading ranges from 60% to 100%
pd_min =  pd_max.*0.6;
pd_delta = pd_max.*0.4;
pg_delta = mpc.gen(1:end,PMAX)-mpc.gen(1:end,PMIN);

% options
mpopt = mpoption;
mpopt.verbose = 0;
mpopt.out.all = 0;

% Load the neural network weights and biases
W_input = csvread(strcat(path_input,'W_0.csv')).';
W_output = csvread(strcat(path_input,'W_3.csv')).'; % not clear how the indexing works here (going from layer 1 to layer 2)
W{1} = csvread(strcat(path_input,'W_1.csv')).';
W{2} = csvread(strcat(path_input,'W_2.csv')).';
bias{1} = csvread(strcat(path_input,'b_0.csv')); %net.b; % bias
bias{2} = csvread(strcat(path_input,'b_1.csv'));
bias{3} = csvread(strcat(path_input,'b_2.csv'));
bias{4} = csvread(strcat(path_input,'b_3.csv'));


Input_NN = csvread(strcat(path_input,'features_test.csv'));
Output_NN = csvread(strcat(path_input,'labels_test.csv'));

% load tightened ReLU bounds
load(strcat(path_input,'zk_hat_min'));
load(strcat(path_input,'zk_hat_max'))

% load Relu stability (active/inactive ReLUs)
load(strcat(path_input,'ReLU_stability_inactive'));
load(strcat(path_input,'ReLU_stability_active'));

ReLU_layers = 3;

% Build mixed-integer linear represenation of trained neural networks
size_input = size(Input_NN,2);
LP_relax = false;
pd_NN = sdpvar(size_input,1);
if LP_relax == true %integer relaxation
    ReLU = sdpvar(nr_neurons,1,ReLU_layers);
else
    ReLU = binvar(nr_neurons,1,ReLU_layers);
end

zk_hat = sdpvar(nr_neurons,1,ReLU_layers);
zk = sdpvar(nr_neurons,1,ReLU_layers);
pg_pred = sdpvar(size(Output_NN,2),1);

constr = [];
%input restrictions
constr = [constr;...
    0.0+delta <= pd_NN <= 1.0-delta];
%input layer
constr = [constr; ...
    zk_hat(:,:,1) == W_input*pd_NN + bias{1}];

for i = 1:ReLU_layers
    for jj = 1:nr_neurons
        
        if ReLU_stability_active(1,i,jj) == 1
            % this RELU is assumed to be stable and active
            constr = [constr; ...
                zk(jj,1,i) == zk_hat(jj,1,i)];
        elseif ReLU_stability_inactive(1,i,jj) == 1
            % this RELU is assumed to be stable and inactive
            constr = [constr; ...
                zk(jj,1,i) == 0];
        else
            
            % ReLU (rewriting the max function)
            constr = [constr; ...
                zk(jj,1,i) <= zk_hat(jj,1,i) - zk_hat_min(jj,1,i).*(1-ReLU(jj,1,i));...1
                zk(jj,1,i) >= zk_hat(jj,1,i);...
                zk(jj,1,i) <= zk_hat_max(jj,1,i).*ReLU(jj,1,i);...
                zk(jj,1,i) >= 0];
        end
    end
end
for i = 1:ReLU_layers-1
    constr = [constr; ...
        zk_hat(:,:,i+1) == W{i}*zk(:,:,i) + bias{i+1}];
end
if LP_relax == true %integer relaxation
    % % integer relaxation
    constr = [constr; ...
        0<= ReLU <=1 ];
end
% output layer
constr = [constr; ...
    pg_pred == W_output * zk(:,:,end) + bias{end}];

options = sdpsettings('solver','gurobi','verbose',0,'savesolveroutput',1);

% DC power flow equations
theta = sdpvar(nb,1);
pline = sdpvar(nline,1);


baseMVA = mpc.baseMVA;
[Bbus, Bline, ~, ~] = makeBdc(baseMVA, mpc.bus, mpc.branch);

% DC power balance to find line flow
constr = [constr;...
    M_g*(pg_pred.*(pg_delta)/baseMVA) - M_d * (pd_NN.*pd_delta+pd_min) / baseMVA  == Bbus*theta];
% Active line flows
constr = [constr;...
    pline == Bline * theta];
% slack bus constraint
constr = [constr; ...
    theta(mpc.gen(1,GEN_BUS))==0];


v_g_wc_max = zeros(ng,1);
v_g_time_max = zeros(ng,1);

Pl_values = zeros(2*ng+nline,nd);

fprintf('Solving MILP for PGMAX Violations \n');
for i = 1:ng
    options.gurobi.BestBdStop = 0.0;
    i
    obj = (-1)*(pg_pred(i)*mpc.gen(i,PMAX)-mpc.gen(i,PMAX));

    diagnostics = optimize(constr,obj,options);
    v_g_time_max(i,1) = diagnostics.solvertime;
    diagnostics.solvertime
    Pl_values(i,:) = value(pd_NN);
    
    % MILP_TIME
    if diagnostics.problem ~= 0 && diagnostics.problem ~= -1
        
        % rerun the simulation with presolve set to conservative
        options.gurobi.Presolve = 1;
        diagnostics = optimize(constr,obj,options);
        % if issues persist abort
        
        if diagnostics.problem ~= 0 && diagnostics.problem ~= 3 && diagnostics.problem ~= -1
            diagnostics
            error('some issue with solving MILP PGMAX');
        end
        
        
    end
    if diagnostics.problem == 0
        v_g_wc_max(i,1) = value(obj)*(-1);
        options.gurobi.BestBdStop = value(obj);
        
        % Double check that the computed neural network input does lead to the
        % worst-case violation
        mpc_test = mpc;
        pg_pred_NN = Predict_NN_Output_with_ReLU_Stability(value(pd_NN).',W_input,bias,W,W_output,ReLU_layers,ReLU_stability_active,ReLU_stability_inactive);
        if sum(abs(pg_pred_NN-value(pg_pred))) > 1
            error('Mismatch in neural network prediction -- PGMAX ');
        end
        mpc_test.bus(ID_loads,PD) = (value(pd_NN)).*pd_delta +pd_min ;
        mpc_test.gen(1:end,PG) = pg_delta.*pg_pred_NN;
        results_dcpf = rundcpf(mpc_test,mpopt);
        pg_viol_max = results_dcpf.gen(i,PG)-results_dcpf.gen(i,PMAX);
        if abs(pg_viol_max-v_g_wc_max(i,1))>1000
            error('Mismatch in worst-case violation -- PGMAX');
        end
        if diagnostics.solveroutput.result.mipgap>10^-4
            error('MILP gap larger than 10^-4')
        end
        
    elseif diagnostics.problem == -1
        v_g_wc_max(i,1) = options.gurobi.BestBdStop-10^-4;
    end
    
end

v_g_wc_min = zeros(ng,1);
v_g_time_min = zeros(ng,1);

fprintf('Solving MILP for PGMIN Violations \n');
for i = 1:ng
    options.gurobi.BestBdStop = 0.0;
    i
    obj = (-1)*(mpc.gen(i,PMIN)-(pg_pred(i))*mpc.gen(i,PMAX));

    diagnostics = optimize(constr,obj,options);
    v_g_time_min(i,1) = diagnostics.solvertime;
    diagnostics.solvertime
    Pl_values(ng+i,:) = value(pd_NN);
    if diagnostics.problem ~= 0 && diagnostics.problem ~= -1
        % rerun the simulation with presolve set to conservative
        options.gurobi.Presolve = 1;
        diagnostics = optimize(constr,obj,options);
        % if issues persist abort
        if diagnostics.problem ~= 0 && diagnostics.problem ~= 3 && diagnostics.problem ~= -1
            diagnostics
            error('some issue with solving MILP PGMIN');
        end
        
    end
    
    if diagnostics.problem == 0
        v_g_wc_min(i,1) = value(obj)*(-1);
        options.gurobi.BestBdStop = value(obj);
        
        % Double check that the computed neural network input does lead to the
        % worst-case violation
        mpc_test = mpc;
        pg_pred_NN = Predict_NN_Output_with_ReLU_Stability(value(pd_NN).',W_input,bias,W,W_output,ReLU_layers,ReLU_stability_active,ReLU_stability_inactive);
        if sum(abs(pg_pred_NN-value(pg_pred))) > 1
            error('Mismatch in neural network prediction -- PGMIN ');
        end
        mpc_test.bus(ID_loads,PD) = (value(pd_NN)).*pd_delta +pd_min ;
        mpc_test.gen(1:end,PG) = pg_delta.*pg_pred_NN;
        results_dcpf = rundcpf(mpc_test,mpopt);
        pg_viol_max = results_dcpf.gen(i,PMIN)-results_dcpf.gen(i,PG);
        if abs(pg_viol_max-v_g_wc_min(i,1))>1
            error('Mismatch in worst-case violation -- PGMIN');
        end
        if diagnostics.solveroutput.result.mipgap>10^-4
            error('MILP gap larger than 10^-4')
        end
        
    elseif diagnostics.problem == -1
        v_g_wc_min(i,1) = options.gurobi.BestBdStop-10^-4;
    end
    
    
end

v_line_wc = zeros(nline,1);
v_line_time = zeros(nline,1);

fprintf('Solving MILP for PLINE Violations \n');
for i = 1:nline
    solved = false; % some instances have issues with the GUROBI presolve;
    % if solving with presolve fails we re-run without presolve and check again
    for runs = 1:2
        if solved == false
            
            options.gurobi.BestBdStop = 0.0;
            if runs == 1
                %automatic presolve: this speeds up the MILP in most cases, in very few
                %it malfunctions
                options.gurobi.Presolve=-1;
            else
                %this disables presolve
                options.gurobi.Presolve=0;
            end
            
            i
            obj = (-1)*max(abs(pline(i).*baseMVA) - mpc.branch(i,RATE_A));
            diagnostics = optimize(constr,obj,options);
            v_line_time(i,1) = diagnostics.solvertime;
            diagnostics.solvertime
            Pl_values(2*ng+i,:) = value(pd_NN);
            if diagnostics.problem ~= 0 && diagnostics.problem ~= -1
                % rerun the simulation with presolve set to conservative
                options.gurobi.Presolve = 1;
                diagnostics = optimize(constr,obj,options);
                % if issues persist abort
                if diagnostics.problem ~= 0 && diagnostics.problem ~= 3 && diagnostics.problem ~= -1
                    diagnostics
                    error('some issue with solving MILP PLINE');
                end
            end
            
            if diagnostics.problem == 0
                % Double check that the computed neural network input does lead to the
                % worst-case violation
                mpc_test = mpc;
                pg_pred_NN = Predict_NN_Output_with_ReLU_Stability(value(pd_NN).',W_input,bias,W,W_output,ReLU_layers,ReLU_stability_active,ReLU_stability_inactive);
                if sum(abs(pg_pred_NN-value(pg_pred))) > 10^-3
                    diagnostics.problem
                    sum(abs(pg_pred_NN-value(pg_pred)))
                    if runs == 1
                        fprintf('With Presolve: Mismatch in neural network prediction -- PLINE \n');
                    else
                        error('Without Presolve: Mismatch in neural network prediction -- PLINE \n');
                    end
                    solved = false;
                else
                    solved = true;
                end
                
                if solved == true
                    
                    mpc_test.bus(ID_loads,PD) = (value(pd_NN)).*pd_delta +pd_min ;
                    mpc_test.gen(1:end,PG) = pg_delta.*pg_pred_NN;
                    results_dcpf = rundcpf(mpc_test,mpopt);
                    pline_viol_max = abs(results_dcpf.branch(i,PT))-results_dcpf.branch(i,RATE_A);
                    if abs(pline_viol_max-(value(obj)*(-1)))>10^-2
                        if runs == 1
                            fprintf('With Presolve: Mismatch in worst-case violation -- PLINE  \n');
                        else
                            error('Without Presolve: Mismatch in worst-case violation -- PLINE \n');
                        end
                        solved = false;
                        
                        
                    else
                        
                    end
                    
                    if diagnostics.solveroutput.result.mipgap>10^-4
                        error('MILP gap larger than 10^-4')
                    end
                    v_line_wc(i,1) = value(obj)*(-1);
                    options.gurobi.BestBdStop = value(obj);
                end
                
            elseif diagnostics.problem == -1
                solved = true;
                v_line_wc(i,1) = options.gurobi.BestBdStop-10^-4;
            end
        end
    end
end



% Line violation
v_info.v_line_time=sum(v_line_time);
[v_info.v_line_wc,v_info.v_line_ID]=max(v_line_wc);

% Generator violation
v_info.v_g_time = sum(v_g_time_min+v_g_time_max);
% identify whether upper or lower generator bound violations are larger
if max(v_g_wc_min)>max(v_g_wc_max)
    [v_info.v_g_wc,v_info.v_g_ID]=max(v_g_wc_min);
else
    [v_info.v_g_wc,v_info.v_g_ID]=max(v_g_wc_max);
end

end

