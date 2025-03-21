clc;
clear
close all;
%% Setting the parameter for Sparse QCQP
% The prolem is defined based on the objective function f(x) = 0.5*x^T *Q*x
% + l |x|_1. Where Q is a semi-defnite matrix. The constraints are the form
% of f(x) = 0.5*x^T *A_j*x<= d_j. Where A_j is produced similar to Q and
% d_j is a random number uniformly distributed between 0 and 1. 
tStart = tic; 
R= 6; %total runs; 
K =101; %Total iterations; 
n=100; % set the dimension of the primal
m = 10; % set the number of quadratic constraints
time_aug_conex= zeros(1,R); 
time_conex= zeros(1,R); 
optim_gap = zeros(R,K-1);
feasib_gap = zeros(R,K-1);
optim_gap_conex = zeros(R,K-1);
feasib_gap_conex = zeros(R,K-1);
feasibility = zeros(R,m,K-1);
feasibility_conex = zeros(R,m,K-1);
x_optim = zeros(n,R); % optimal solutions of problem
y_optim = zeros(m,R); %optimal solution of Dual problem
iteration_time_aug_conex = zeros(R,K-1); 
iteration_time_conex = zeros(R,K-1); 
x_last_iter = zeros(n,K-1,R);
x_bar = zeros(n,K-1,R);
iter_proj = zeros(R,K-1);
norm_y_tilde = zeros(K-1,R);
norm_s = zeros(K-1,R);
for r=1:R
    diagonal_Q = diag(rand(n,1)*n); % set the diogonal matrix for the objective function exclude 0 for strong convexity
    orthonormal_Q = orth(randn(n)); %set the orthonormal matrix for the objective function 
    Q = orthonormal_Q'* diagonal_Q * orthonormal_Q; 
    c = 20*ones(n,1);
    %c = 10^6*ones(n,1);
    l = 20+ 2*(r-1); % set the sparsity parameter 
    A = zeros(n,n,m);
    B = zeros(n,m);
    for j=1:m
    diagonal_A = diag(rand(n,1)) + eye(n); % constructing diagonal matrix for the constraint
    orthonormal_A = orth(randn(n)); % constructing orthonormal matrix for the constraint
    A(:,:,j) = orthonormal_A'*diagonal_A*orthonormal_A; %constructing the tensor containing m of n*n matrices A_j. 
    B(:,j) = rand(n,1); %contructng the matrix containing m of n dimensional vectors.  
    end
    D_x  = 10;
    norm_A = zeros(m,1);
    norms = zeros(m,1);
    for j=1:m
        norms(j) = D_x*norm(A(:,:,j))+ norm(B(:,j));
        norm_A(j) = norm(A(:,:,j)); 
    end
    d  = 2*rand(m,1); % vector of constants
    l_f  = norm(Q); %smoothneess constant of objective function
    %H_f = l*(2*sqrt(n)); %non-smoothness constant of objective function
    %H_f = l*(2*sqrt(n)); %non-smoothness constant of objective function proj
    H_f = 0; %non-smoothness constant of objective function prox
    mu_f = 0; % strong_convexity of objective function. Changes in strongly-convex setting
    M_g = norm(norms); % Lipschitz-continuity of constraints
    l_g = norm(norm_A); %smoothness of the constraint function
    H_g = 0 ; 
    sigma = 10;
    %initialization
    % Solution in primal
     cvx_begin quiet
     variable x_sol(n)
     minimize(5);
     subject to 
     norm(x_sol)<= D_x;
     for j=1:m 
         0.5*x_sol'*A(:,:,j)*x_sol + x_sol'*B(:,j)<=d(j);
     end
      x_sol>=0 ;
     cvx_end;
     %x_optim(:,r) = x_sol;
    x_init = D_x*ones(n,1)/sqrt(n);
    y_init = zeros(m,1);
     %% Find the optimal solutions in primal and dual (Set a problem that gives non-trivial answer!)
     % Solution in primal
     cvx_begin quiet
     variable x_sol(n)
     minimize(0.5*x_sol'*Q*x_sol + x_sol'*c+ l*norm(x_sol,1));
     subject to 
     norm(x_sol)<= D_x;
     for j=1:m 
         0.5*x_sol'*A(:,:,j)*x_sol + x_sol'*B(:,j)<=d(j);
     end
      %x_sol <= -.1;
     cvx_end;
     x_optim(:,r) = x_sol;
     constantB = 200;
     %% Run the algorithm 
     tstart_aug_conex=tic;
    [optim_gap(r,1:(K-1)/2), feasib_gap(r,1:(K-1)/2),feasibility(r,:,1:(K-1)/2),iteration_time_aug_conex(r,1:(K-1)/2),x_last_iter(:,1:(K-1)/2,r),iter_proj(r,1:(K-1)/2), norm_y_tilde(1:(K-1)/2,r),norm_s(1:(K-1)/2,r)]=aug_conex(n,m,(K-1)/2 +1,Q,c,l,A,B,d,l_f,H_f,H_g,mu_f,M_g,l_g,constantB,sigma,D_x,x_optim(:,r),y_optim(:,r),x_init,y_init);
    [optim_gap(r,(K-1)/2+1:K-1), feasib_gap(r,(K-1)/2+1:K-1),feasibility(r,:,(K-1)/2+1:K-1),iteration_time_aug_conex(r,(K-1)/2+1:K-1),x_last_iter(:,(K-1)/2+1:K-1,r),iter_proj(r,(K-1)/2+1:K-1), norm_y_tilde((K-1)/2+1:K-1,r),norm_s((K-1)/2+1:K-1,r)]=aug_conex(n,m,(K-1)/2 +1,Q,c,l,A,B,d,l_f,H_f,H_g,mu_f,M_g,l_g,constantB,sigma,D_x,x_optim(:,r),y_optim(:,r),x_last_iter(:,(K-1)/2,r),y_init);
    time_aug_conex (r) = toc(tstart_aug_conex);
    tstart_conex =tic;
    [optim_gap_conex(r,1:(K-1)/2), feasib_gap_conex(r,1:(K-1)/2),feasibility_conex(r,:,1:(K-1)/2),iteration_time_conex(r,1:(K-1)/2),x_bar(:,1:(K-1)/2,r)]=ConEx(n,m,(K-1)/2 +1,Q,c,l,A,B,d,l_f,H_f,H_g,mu_f,M_g,l_g,constantB,sigma,D_x,x_optim(:,r),y_optim(:,r),x_init,y_init);
    [optim_gap_conex(r,(K-1)/2+1:K-1), feasib_gap_conex(r,(K-1)/2+1:K-1),feasibility_conex(r,:,(K-1)/2+1:K-1),iteration_time_conex(r,(K-1)/2+1:K-1),x_bar(:,(K-1)/2+1:K-1,r)]=ConEx(n,m,(K-1)/2 +1,Q,c,l,A,B,d,l_f,H_f,H_g,mu_f,M_g,l_g,constantB,sigma,D_x,x_optim(:,r),y_optim(:,r),x_bar(:,(K-1)/2,r),y_init);
    time_conex (r) = toc(tstart_conex);
end
 %% pLot the resutls
avg_iteration_time_aug_conex = mean(cumsum(iteration_time_aug_conex,2),1);
avg_iteration_time_conex = mean(cumsum(iteration_time_conex,2),1);
avg_optim_gap  = mean(optim_gap,1);
avg_feasib_gap  = mean(feasib_gap,1);
avg_optim_gap_conex  = mean(optim_gap_conex,1);
avg_feasib_gap_conex  = mean(feasib_gap_conex,1);
figure('Name','Measures');
subplot(3,2,1); 
plot(avg_optim_gap,'LineWidth',2); 
hold on 
plot(avg_optim_gap_conex,'LineWidth',2);
legend('Aug-ConEx','ConEx');
title("Optimality Convergence");
xlabel('Iteration number','FontWeight','bold');
ylabel('Optimality gap','FontWeight','bold');
set(gca,'FontWeight','bold');
hold off
subplot(3,2,2); 
plot(avg_feasib_gap,'LineWidth',2);
hold on
plot(avg_feasib_gap_conex,'LineWidth',2);
legend('Aug-ConEx','ConEx');
title("Feasibility Convergence");
xlabel('Iteration number','FontWeight','bold');
ylabel('Feasibility gap','FontWeight','bold');
set(gca,'FontWeight','bold');
hold off
subplot(3,2,[3,4])
plot(avg_iteration_time_aug_conex,avg_optim_gap,'LineWidth',2);
hold on
plot(avg_iteration_time_aug_conex,avg_optim_gap_conex,'LineWidth',2);
legend('Aug-ConEx', 'ConEx')
title("Time complexity of optimality gap");
xlabel('Run time (seconds)','FontWeight','bold');
ylabel('Optimality gap','FontWeight','bold');
set(gca,'FontWeight','bold');
hold off
subplot(3,2,[5,6])
plot(avg_iteration_time_aug_conex,avg_feasib_gap,'LineWidth',2);
hold on
plot(avg_iteration_time_aug_conex,avg_feasib_gap_conex,'LineWidth',2);
legend('Aug-ConEx', 'ConEx');
title("Time complexity of feasibility gap");
xlabel('Run time (seconds)','FontWeight','bold');
ylabel('Feasibility gap','FontWeight','bold');
set(gca,'FontWeight','bold');
hold off