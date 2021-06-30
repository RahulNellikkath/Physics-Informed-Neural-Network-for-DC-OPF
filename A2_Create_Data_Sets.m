% We create the datasets for neural network training
% Input/Feature: System loading
% Output: Optimal generation with MATPOWER DC-OPF
clear all;
close all;

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

cases =  {'case39_DCOPF_39_bus';'case118_DCOPF';'case162_DCOPF'};

nr_cases = size(cases,1);
Nr_samples = 10^2;

tic();

for c = 1:nr_cases
    
    mpc = eval(cases{c});
    
    nb = size(mpc.bus,1);
    ng = size(mpc.gen,1);
    nbr = size(mpc.branch,1);
    
    dataset_folder = strcat(pwd,'/Data_File/',num2str(nb),'/');
    
    % identify the loads which are non-zero
    ID_loads = find(mpc.bus(:,PD)~=0);
    
    %map from buses to loads and generator loads
    nloads = size(ID_loads,1);
    map_loadsbus=zeros(nloads,nb);
    map_l2b = zeros(nb,nloads);
    for i = 1:nloads
        map_loadsbus(i,ID_loads(i)) = 1;
        map_l2b(ID_loads(i),i) = 1;
    end
    
    pd_max = mpc.bus(ID_loads,PD);
    
    % this is the definition of the load input domain
    % Here: from 60% to 100% loading
    
    pd_min =  pd_max.*0.6;
    pd_delta = pd_max.*0.4;
    
    
    mpopt = mpoption;
    mpopt.verbose = 0;
    mpopt.out.all = 0;
    
    pg_delta = mpc.gen(1:end,PMAX)-mpc.gen(1:end,PMIN);
    
    % step size
    % Latin Hypercube Sampling
    input_lhs = lhsdesign(Nr_samples,nloads);
    Input = input_lhs;
    Output = zeros(Nr_samples,ng);
    
    feas = zeros(Nr_samples,1);
    
    for n = 1:Nr_samples
        mpc_new = mpc;
        % set the system loading
        mpc_new.bus(ID_loads,PD)=(input_lhs(n,:).').*pd_delta+pd_min;
        results_dcopf = rundcopf(mpc_new,mpopt);
        if results_dcopf.success ~= 1
            feas(n,1) = -1;
            fprintf('DCOPF is infeasible\n');
        else
            feas(n,1)= 1;
        end
        Output(n,:) = (results_dcopf.gen(:,PG)./pg_delta).';
        
    end
    PTDF=makePTDF(mpc_new.baseMVA, mpc_new.bus, mpc_new.branch, mpc.gen(1,1))';
    % save results for neural network training
    csvwrite(strcat(dataset_folder,'PTDF.csv'),PTDF);
    csvwrite(strcat(dataset_folder,'NN_input.csv'),Input);
    csvwrite(strcat(dataset_folder,'NN_output.csv'),Output);
    
    if sum(feas) ~= Nr_samples
        error('Not all DC-OPF problems are feasible -- investigate');
    end
    
end


toc();
