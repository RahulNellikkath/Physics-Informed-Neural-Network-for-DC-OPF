% This function converts the matpower/pglib opf test cases to the DC-OPF test cases 
% and applies several assumptions

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

% the first test case is from MATPOWER 
% available at https://matpower.org/
% the other are PGLIB-OPF networks
% available at https://github.com/power-grid-lib/pglib-opf
cases = {'case9';'pglib_opf_case30_ieee';'pglib_opf_case39_epri';
    'pglib_opf_case57_ieee';'pglib_opf_case118_ieee'; 'pglib_opf_case162_ieee_dtc';
    'pglib_opf_case300_ieee'};

nr_cases = size(cases,1);

% to create table on Test Case Characteristics
nr_d_ = zeros(nr_cases,1);
nr_g_ = zeros(nr_cases,1);
nr_b_ = zeros(nr_cases,1);
nr_line_ = zeros(nr_cases,1);
sum_p_d_max_ = zeros(nr_cases,1);


for c = 1:nr_cases
    mpc = eval(cases{c});
    
    % We apply several assumptions to the test cases to ensure
    % compatability
    % with the current implementation
    
    % remove external bus numbering
    mpc = ext2int(mpc);
    mpc=rmfield(mpc,'order');
    
    % setting lower generator limits to zero
    mpc.gen(:,PMIN)=0;
        
    % loosen angle limits
    mpc.branch(:,ANGMAX) = 360;
    mpc.branch(:,ANGMIN) = -360;
    
    % consider only linear costs
    mpc.gencost(:,5) = 0;
    mpc.gencost(:,7) = 0;
  
    nb = size(mpc.bus,1);
    
    % delete all generators that are synchronous condensers as they do not
    % play any role in the DC-OPF
    ID_synchr = mpc.gen(:,PMAX)==0;
    ID_bus_synchr = mpc.gen(ID_synchr,GEN_BUS);
    mpc.gen(ID_synchr,:) = [];
    mpc.gencost(ID_synchr,:) = [];
    mpc.bus(ID_bus_synchr,BUS_TYPE) = PQ;
    
    % delete shunt elements
    mpc.bus(:,GS) = 0; 
    
    % delete phase shifts
    mpc.branch(:,SHIFT) = 0;
    
    % setting slack bus to first bus
    mpc.bus(find(mpc.bus(:,2)==3),2)=2;
    mpc.bus(mpc.gen(1,1),2)=3;
    
    mpc.gen = [mpc.gen zeros(size(mpc.gen,1),11)];
    
    % save test cases
    savecase(strcat('./Test_Cases/case',num2str(nb),'_DCOPF'),mpc);
    
    nr_d_(c,1) = sum(mpc.bus(:,PD)~=0);
    nr_g_(c,1) = size(mpc.gen,1);
    nr_b_(c,1) = size(mpc.bus,1);
    nr_line_(c,1) = size(mpc.branch,1);
    sum_p_d_max_(c,1) = sum(mpc.bus(:,PD));
    
end

% output the necessary entries for the first table in the paper
% test set characteristics
nr_d_
nr_g_
nr_b_
nr_line_
sum_p_d_max_
