function [v_info] = Compute_Worst_Case_Distance_Sub_Optimality(mpc,path_input,delta,time_MILP_max)

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

% neural network size
ReLU_layers =3;
nr_neurons = 20;

% identify the loads which are non-zero
ID_loads = find(mpc.bus(:,PD)~=0);
nd=size(ID_loads,1);
pd_max = mpc.bus(ID_loads,PD);

ng= size(mpc.gen,1);
nb = size(mpc.bus,1);
nline = size(mpc.branch,1);


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

pd_min = pd_max.*0.6;
pd_delta = pd_max.*0.4;
pg_delta = mpc.gen(1:end,PMAX) - mpc.gen(1:end,PMIN);

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

size_input = size(Input_NN,2);
LP_relax = false;
% construct otpimization problem of neural network
pd_NN = sdpvar(size_input,1);
if LP_relax == true %integer relaxation
    ReLU = sdpvar(hidden_layer,1,ReLU_layers);
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


v_dist_max = zeros(ng,1);
v_dist_time = zeros(ng,1);
MILP_gap_v_dist_max = zeros(ng,1);
MILP_exact_v_dist_max = zeros(ng,1);
v_opt_max = zeros(1,1);

v_opt_time = zeros(1,1);

pg = sdpvar(ng,1);
pg_slack_error = sdpvar(1,1);

cost = mpc.gencost(:,6)*mpc.baseMVA;
theta = sdpvar(nb,1);


pgmin = mpc.gen(:,PMIN)./mpc.baseMVA;
pgmax = mpc.gen(:,PMAX)./mpc.baseMVA;
plinemin = (-1)*mpc.branch(:,RATE_A)./mpc.baseMVA;
plinemax = mpc.branch(:,RATE_A)./mpc.baseMVA;

[Bbus, Bline, Pbusinj, Pfinj] = makeBdc(mpc.baseMVA, mpc.bus, mpc.branch);

if sum(abs(Pbusinj)) ~= 0.0
    error('Pbusinj not zero -- not supported');
end
if sum(abs(Pfinj)) ~= 0.0
    error('Pfinj not zero -- not supported');
end

% primal constraints DC-OPF
constr_primal = [M_g*pg- M_d*(pd_NN.*pd_delta/mpc.baseMVA + pd_min/mpc.baseMVA) == Bbus*theta; ...
    plinemin <= Bline * theta <= plinemax; ...
    pgmin <= pg <= pgmax;
    theta(mpc.gen(1,GEN_BUS))==0];

obj = cost.' *pg

% dual variables
mu_g_min = sdpvar(ng,1);
mu_g_max = sdpvar(ng,1);
mu_line_max = sdpvar(nline,1);
mu_line_min = sdpvar(nline,1);
lambda = sdpvar(nb,1);

% stationarity
constr_stationarity = [cost-mu_g_min+mu_g_max+ M_g.' * lambda == 0;...
    -Bline.' * mu_line_min + Bline.' * mu_line_max - Bbus*lambda == 0];
% dual feasibility
constr_dual = [mu_g_min >= 0; mu_g_max >= 0; mu_line_max >= 0; mu_line_min>=0];
% complementary slackness
M_pg_min=10^4;
b_pg_min=binvar(ng,1);
M_pg_max=10^4;
b_pg_max=binvar(ng,1);
M_pline_min=10^4;
b_pline_min=binvar(nline,1);
M_pline_max=10^4;
b_pline_max=binvar(nline,1);

% These are the complementary conditions
% (pgmin-pg)*mu_g_min
% (pg-pgmax)*mu_g_max
% (plinemin-Bline*theta)*mu_line_min
% (Bline*theta-plinemax)*mu_line_max

% We have three methods to implement the complentary slackness
% conditions
% a) Fortuny-Amat McCarl linearization (big-M reformulation)
% b) complements command in YALMIP
% c) kkt command in YALMIP


% a) Fortuny-Amat McCarl linearization (big-M reformulation)

constr_compl_slack = [...
    (pgmin-pg)>=-b_pg_min*M_pg_min;...
    mu_g_min<=(1-b_pg_min)*M_pg_min;...
    (pg-pgmax)>=-b_pg_max*M_pg_max;...
    mu_g_max<=(1-b_pg_max)*M_pg_max;...
    (plinemin-Bline*theta)>=-b_pline_min*M_pline_min;...
    mu_line_min<=(1-b_pline_min)*M_pline_min;...
    (Bline*theta-plinemax)>=-b_pline_max*M_pline_max;...
    mu_line_max<=(1-b_pline_max)*M_pline_max;...
    ];


% % b) complements command in YALMIP
% % using build in command complements of YALMIP
% constr_compl_slack_yalmip = [...
%     complements((pgmin-pg)<=0,mu_g_min>=0);...
%     complements((pg-pgmax)<=0,mu_g_max>=0);...
%     complements((plinemin-Bline*theta)<=0,mu_line_min>=0);...
%     complements((Bline*theta-plinemax)<=0,mu_line_max>=0);...
%     ];
%
%
% % c) kkt command in YALMIP
%


KKTsystem = [constr_primal; constr_stationarity; constr_dual; constr_compl_slack];


% This constraint we have defined a term slack_error to compensate for the differance in power balance after PINN prediction bus generation
% That is added to the predicted value of generation at the slack bus

constr_slack =   [sum(pd_NN.*pd_delta + pd_min)/mpc.baseMVA == sum([pg_pred(1)+pg_slack_error; pg_pred(2:end)].*pg_delta/mpc.baseMVA)];
options = sdpsettings('solver','gurobi','verbose',0,'savesolveroutput',1);
options.gurobi.TimeLimit = time_MILP_max;
options.gurobi.BestBdStop = 100000;
options.verbose = 1;


max_compl = zeros(ng+1,1);
max_dual = zeros(ng+1,1);
mpc_ver = mpc;
tic();
for g = 1:ng
    % automatic presolve -- in most instances this leads to a significant
    % speed-up; can however cause numerical issues in few instances
    options.gurobi.Presolve = -1;
    
    g
    if g == 1
        constr_overall = [constr;  constr_slack; KKTsystem];
        obj_overall = -abs(pg_slack_error+pg_pred(g)- pg(g)*mpc.baseMVA/pg_delta(g));
    else
        constr_overall = [constr;  KKTsystem];
        obj_overall = -abs(pg_pred(g)-pg(g)./mpc.gen(g,PMAX)*mpc.baseMVA);
    end
    
    diagnostics = optimize(constr_overall,obj_overall,options);
    
    % MILP_TIME
    if diagnostics.problem ~= 0 && diagnostics.problem ~= 3 && diagnostics.problem ~= -1
        % rerun the simulation with presolve set to conservative
        options.gurobi.Presolve = 1;
        diagnostics = optimize(constr_overall,obj_overall,options);
        % if issues persist abort
        if diagnostics.problem ~= 0 && diagnostics.problem ~= 3 && diagnostics.problem ~= -1
            
            % rerun the simulation with presolve disabled
            options.gurobi.Presolve = 0;
            diagnostics = optimize(constr_overall,obj_overall,options);
            % if issues persist abort
        
        if diagnostics.problem ~= 0 && diagnostics.problem ~= 3 && diagnostics.problem ~= -1
            diagnostics
            error('KKT MIL solve issue');
        end
        end 
    end
    
    
    if diagnostics.problem == 0
        
        % we need to double check that the
        % a) primal variables do not violate the derived bounds in the kkt
        % command
        if any(value(pgmin-pg)>=(M_pg_min-10^-2)) || any(value(pg-pgmax)>=(M_pg_min-10^-2)) ||any(value (plinemin-Bline*theta)>=(M_pg_min-10^-2)) ||any(value(Bline*theta-plinemax)>=(M_pg_min-10^-2))
            error('Primal bounds are binding')
        end
        max_dual(g,1) = max(max(value([mu_g_min;mu_g_max;mu_line_min;mu_line_max])));
        % b) dual variables do not violate the derived bounds in the kkt
        % command
        if any(value(mu_g_min)>=(M_pg_min-10^-2)) || any(value(mu_g_max)>=(M_pg_min-10^-2)) ||any(value (mu_line_min)>=(M_pg_min-10^-2)) ||any(value(mu_line_max)>=(M_pg_min-10^-2))
            error('Dual bounds are binding')
        end
        
        % c) the complementary slackness conditions hold
        % max_compl(g,1) = max( abs(value(details.dual.*(details.A*details.primal-details.b))));
        max_compl(g,1) =   max(max(abs(value([(pgmin-pg).*mu_g_min;(pg-pgmax).*mu_g_max;(plinemin-Bline*theta).*mu_line_min;(Bline*theta-plinemax).*mu_line_max]))));
        if  max_compl(g,1) > 10^-3
            error('Complementary slackness conditions do not hold -- improve numerical accuracy');
        end
        
        v_dist_max(g,1) = (-1)*value(obj_overall);
        MILP_exact_v_dist_max(g,1) = 1;
        options.gurobi.BestBdStop = value(obj_overall);
        
        % check that the obtained result matches!
        pg_pred_NN = Predict_NN_Output_with_ReLU_Stability(value(pd_NN).',W_input,bias,W,W_output,ReLU_layers,ReLU_stability_active,ReLU_stability_inactive);
        
        if sum(abs(pg_pred_NN-value(pg_pred))) > 10^-3
            error('Mismatch');
        end
        
    elseif diagnostics.problem == -1
        v_dist_max(g,1) = (-1)* options.gurobi.BestBdStop-10^-4;
        MILP_exact_v_dist_max(g,1) = -1;
    elseif diagnostics.problem == 3
        v_dist_max(g,1) = (-1)*diagnostics.solveroutput.result.objbound;
        MILP_exact_v_dist_max(g,1) = 0;
    end
    MILP_gap_v_dist_max(g,1) = diagnostics.solveroutput.result.mipgap;
    
    
    
    diagnostics.solvertime
    v_dist_time(g,1) = diagnostics.solvertime;
    
    % MILP_TIME
    if diagnostics.problem == 0
        % here we need to build the check which compares that for the
        % identified system loading, the solution produced by the KKTs is
        % actually the optimal solution to the DC-OPF (we use rundcopf here;
        % note that this is computationally much cheaper than solving the MILP
        % so we can do it for every one)
        % Note that this check is for debugging purposes only
        
        % set the load
        mpc_ver.bus(:,PD) = value(M_d*(pd_NN.*pd_delta + pd_min));
        
        % solve the dc-opf
        results_dcopf = rundcopf(mpc_ver,mpopt);
        if results_dcopf.success ~= 1
            error('DCOPF solve error')
        end
        % extract the active generator dispatch and compare
        % throw an error if they do not match
        if sum(abs(value(pg)-results_dcopf.gen(:,PG)./mpc.baseMVA)) >= 10
            error('KKT solution and rundcopf do not match')
        else
            fprintf('KKT solution and rundcopf do match -- continue \n')
        end
        
        % This is checking that the neural network prediction is correct
        pg_pred_NN = Predict_NN_Output_with_ReLU_Stability(value(pd_NN).',W_input,bias,W,W_output,ReLU_layers,ReLU_stability_active,ReLU_stability_inactive);
        
        if sum(abs(pg_pred_NN-value(pg_pred))) >= 10^2
            error('Neural network prediction and MILP do not match')
        else
            fprintf('Neural network prediction and MILP do match -- continue \n')
        end
        
    end
end


% compute the maximum error on the objective function value

constr_overall = [constr; constr_slack; KKTsystem];
obj_overall = (-1) * (mpc.gencost(:,6).'*([(pg_pred(1)+pg_slack_error); pg_pred(2:end)].*pg_delta-pg.*mpc.baseMVA));
options.gurobi.BestBdStop = 100000;
diagnostics = optimize(constr_overall,obj_overall,options);

% MILP_TIME
if diagnostics.problem ~= 0 && diagnostics.problem ~= 3
    % rerun the simulation with presolve set to conservative
    options.gurobi.Presolve = 1;
    diagnostics = optimize(constr_overall,obj_overall,options);
    % if issues persist abort
    if diagnostics.problem ~= 0 && diagnostics.problem ~= 3 && diagnostics.problem ~= -1
        diagnostics
        error('KKT MIL solve issue');
    end
end
if diagnostics.problem == 0
    v_opt_max(1,1)=(-1)*value(obj_overall);
    MILP_exact_v_opt_max = 1;
    
    % we need to double check that the
    % a) primal variables do not violate the derived bounds in the kkt
    % command
    if any(value(pgmin-pg)>=(M_pg_min-10^-2)) || any(value(pg-pgmax)>=(M_pg_min-10^-2)) ||any(value (plinemin-Bline*theta)>=(M_pg_min-10^-2)) ||any(value(Bline*theta-plinemax)>=(M_pg_min-10^-2))
        error('Primal bounds are binding')
    end
    max_dual(ng+1,1) = max(max(value([mu_g_min;mu_g_max;mu_line_min;mu_line_max])));
    % b) dual variables do not violate the derived bounds in the kkt
    % command
    if any(value(mu_g_min)>=(M_pg_min-10^-2)) || any(value(mu_g_max)>=(M_pg_min-10^-2)) ||any(value (mu_line_min)>=(M_pg_min-10^-2)) ||any(value(mu_line_max)>=(M_pg_min-10^-2))
        error('Dual bounds are binding')
    end
    
    % c) the complementary slackness conditions hold
    % max_compl(g,1) = max( abs(value(details.dual.*(details.A*details.primal-details.b))));
    max_compl(ng+1,1) =   max(max(abs(value([(pgmin-pg).*mu_g_min;(pg-pgmax).*mu_g_max;(plinemin-Bline*theta).*mu_line_min;(Bline*theta-plinemax).*mu_line_max]))));
    if  max_compl(ng+1,1) > 10^-4
        error('Complementary slackness conditions do not hold -- improve numerical accuracy');
    end
    
    
    % check that the obtained result matches!
    pg_pred_NN = Predict_NN_Output_with_ReLU_Stability(value(pd_NN).',W_input,bias,W,W_output,ReLU_layers,ReLU_stability_active,ReLU_stability_inactive);
    
    if sum(abs(pg_pred_NN-value(pg_pred))) > 10^-3
        error('Mismatch');
    end
    
else
    v_opt_max(1,1) = (-1)*diagnostics.solveroutput.result.objbound;
    MILP_exact_v_opt_max = 0;
end


v_opt_time(1,1) = diagnostics.solvertime;
if diagnostics.problem == 0
    
    
    % double check solution as well
    % set the load
    mpc_ver.bus(:,PD) = value(M_d*(pd_NN.*pd_delta + pd_min));
    
    % solve the dc-opf
    results_dcopf = rundcopf(mpc_ver,mpopt);
    if results_dcopf.success ~= 1
        error('DCOPF solve error')
    end
    % extract the active generator dispatch and compare
    % throw an error if they do not match
    if sum(abs(value(pg)-results_dcopf.gen(:,PG)./mpc.baseMVA)) >= 10^-1
        error('KKT solution and rundcopf do not match')
    else
        fprintf('KKT solution and rundcopf do match -- continue \n')
    end
    
    % This is checking that the neural network prediction is correct
    pg_pred_NN = Predict_NN_Output_with_ReLU_Stability(value(pd_NN).',W_input,bias,W,W_output,ReLU_layers,ReLU_stability_active,ReLU_stability_inactive);
    
    if sum(abs(pg_pred_NN-value(pg_pred))) >= 10^-5
        error('Neural network prediction and MILP do not match')
    else
        fprintf('Neural network prediction and MILP do match -- continue \n')
    end
end


% create output
v_info.v_dist_time = sum(v_dist_time);
v_info.v_dist_wc =max(v_dist_max);
[~,v_info.v_dist_ID] = max(v_dist_max);
v_info.v_opt_time = v_opt_time;

v_info.MILP_exact_v_dist_max=MILP_exact_v_dist_max;
v_info.MILP_gap_v_dist_max=MILP_gap_v_dist_max;
v_info.MILP_exact_v_opt_max=MILP_exact_v_opt_max;
v_info.max_compl=max_compl;
v_info.max_dual = max_dual;

% create reference cost
mpc_ref = mpc;
mpopt = mpoption;
mpopt.out.all =0;
results_dcopf=rundcopf(mpc_ref,mpopt);

v_info.v_opt_wc = v_opt_max./results_dcopf.f;

end




