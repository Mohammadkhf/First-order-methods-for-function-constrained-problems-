function [optim_gap, feasib_gap,feasibility,iteration_time_conex,x_bar]=ConEx(n,m,K,Q,c,l,A,B,d,l_f,H_f,H_g,mu_f,M_g,l_g,constantB,sigma,D_x,x_optim,y_optim,x_init,y_init)
%% set the step-sizes and initialization 
D_x = 2*D_x;  %Differecne in the definition of Dx
gamma = zeros(1,K);
if mu_f == 0
    k_0 = 0;
else
    k_0 = 4*(l_f + constantB*l_g)/mu_f + 2;
end
gamma(1) = 1;
gamma(2) = (mu_f==0)*1 + (mu_f~=0)*((k_0+2)/(k_0+3));
theta=zeros(1,K);
tau=zeros(1,K);
tau(1) = (mu_f==0)*4*D_x*M_g/constantB + (mu_f~=0)*7;
theta(1) = 1;
eta = zeros(1,K);
H_star = H_g*constantB;
eta_0 = (sqrt(2*K*(H_star^2 +(H_f^2+sigma^2)))/D_x) + 12*constantB*M_g/D_x;
x = zeros(n,K);
x_bar = zeros(n,K-1);
x_bar(:,1) = x_init;
x(:,1) = x_init;
x(:,2) = x_init;
y = zeros(m,K);
y(:,1) = y_init;
y(:,2) = y(:,1);
iteration_time_conex = zeros(1,K-1);
s = zeros(m,K-1);
g = zeros(m,K);
g_x = zeros(m,K); 
ell_g = zeros(m,K);
grad_g  = zeros (n,m); 
optim_gap = zeros(K-1,1);
feasib_gap = zeros(K-1,1);
for j=1:m 
    g(j,1) =  0.5*x_init'*A(:,:,j)*x_init + x_init'*B(:,j)-d(j);
    g(j,2) = g(j,1);
    ell_g(j,2) = g(j,1); 
    ell_g(j,1) = g(j,1);
end
for k=2:K-1
    tStart_conex = tic;
    theta(k) = (mu_f == 0)* theta(1) + (mu_f~=0)*(k+k_0+1)/(k+k_0+2);
    tau(k) = (mu_f==0)*tau(1) + (mu_f~=0)*128*M_g^2/mu_f;
    gamma(k+1) = (mu_f == 0)* gamma(1) + (mu_f~=0)*(k+k_0+2);
    eta(k) = (mu_f==0)*eta_0 + (mu_f~=0)*mu_f*(k+k_0+1)/2;
    for j=1:m 
     grad_g(:,j) = A(:,:,j)*x(:,k) + B(:,j); 
    end
    s(:,k) = (1+theta(k))*(ell_g(:,k)) - theta(k)*(ell_g(:,k-1));
    y(:,k+1) = max(y(:,k) + s(:,k)/tau(k),0);
    %Grad for projection (uncomment for projection)
    %stoch_grad  = Q*x(:,k)+c+l*sign(x(:,k)) + normrnd(0,sqrt(sigma),n,1); 
    %Grad for prox oracle (uncomment for prox)
    stoch_grad  = Q*x(:,k)+c + normrnd(0,sqrt(sigma),n,1); 
eval_point = x(:,k) - (stoch_grad+ grad_g*y(:,k+1))/eta(k);
%uncomment for projection
%x(:,k+1)=  eval_point * min (1, D_x/norm(eval_point));
%uncomment for prox
unconstrained_min = max(eval_point - l/eta(k),0) -  max(-eval_point -l/eta(k),0);
 x(:,k+1)= unconstrained_min*min (1, D_x/norm(unconstrained_min));
 ell_g(:,k+1) = g_x(:,k) + grad_g'*(x(:,k+1)-x(:,k));
 for j=1:m
     g_x(j,k+1) =  0.5*x(:,k+1)'*A(:,:,j)*x(:,k+1) + x(:,k+1)'*B(:,j)-d(j);
 end
%Prox with l1 norm in l2 ball constraint
    % unconstrained_min = max(eval_point - l/eta(k),0) -  max(-eval_point -l/eta(k),0);
    % x(:,k+1) = unconstrained_min*min (1, D_x/norm(unconstrained_min)); 
 iteration_time_conex(k-1) = toc(tStart_conex);
end
%% Measures 
feasibility = zeros(m,K-1);
for k=1:K-1
    x_bar(:,k) = sum(gamma(2:k+1).*x(:,2:k+1),2)/sum(gamma(2:k+1));
    optim_gap(k) = 0.5*x_bar(:,k)'*Q*x_bar(:,k) + x_bar(:,k)'*c+ l*norm(x_bar(:,k),1)-...
        (0.5*x_optim'*Q*x_optim + x_optim'*c+ l*norm(x_optim,1));
    for j=1:m 
         feasibility(j,k) = 0.5*x_bar(:,k)'*A(:,:,j)*x_bar(:,k) + x_bar(:,k)'*B(:,j)-d(j);
    end
    feasib_gap(k) = norm(max(feasibility(:,k),0));
end









