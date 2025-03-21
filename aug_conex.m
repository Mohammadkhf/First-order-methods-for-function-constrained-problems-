function [optim_gap, feasib_gap,feasibility,iteration_time_aug_conex,x_last_iter,iter_proj,norm_y_tilde, norm_s]=aug_conex(n,m,K,Q,c,l,A,B,d,l_f,H_f,H_g,mu_f, M_g, l_g,constantB,sigma,D_x,x_optim,y_optim,x_init,y_init);
%% set the step-sizes and initialization 
rho = zeros(1,K);
rho(1) = (mu_f==0)*1 + (mu_f~=0)*mu_f/(2*M_g^2);
tau=zeros(1,K);
tau(1) = 1;
L = zeros(1,K);
iter_proj = zeros(1,K-1);
eta = zeros(1,K);
%H_star = l_g*D_x*max(norm(y_optim)+1-constantB,0);
H_star = H_g*constantB;
L(1) = 2*(l_f+constantB*l_g + rho(1)*K*M_g^2) + (K*sqrt(120*K*(H_star^2 + 8*(H_f^2+sigma^2))))/(120*D_x);
beta =zeros(1,K);
x_hat = zeros(n,K);
x = zeros(n,K-1);
x(:,1) = x_init;
y_tilde = zeros(m,K);
y_tilde(:,1) = y_init;
x_hat(:,1) = x_init;
iteration_time_aug_conex = zeros(1,K-1);
s = zeros(m,K-1);
v = zeros(m,K);
g = zeros(m,K);
g_x_hat = zeros(m,K);
ell_g = zeros(m,K);
grad_g  = zeros (n,m); 
u = zeros (m,1); 
optim_gap = zeros(K-1,1);
feasib_gap = zeros(K-1,1);
%Debugging
norm_y_tilde = zeros(K-1,1);
norm_s = zeros(K-1,1);

%T = 100; %set the interations of projection
error = 10^-10; %set the projection error.
for j=1:m 
    g(j,1) =  0.5*x_init'*A(:,:,j)*x_init + x_init'*B(:,j)-d(j);
end
v(:,1) = g(:,1);
for k=1:K-1
    tStart_auge_conex = tic;
    tau(k+1) =(mu_f==0)*2/(k+2) + (mu_f~=0)*0.5*tau(k)*(sqrt(tau(k)^2+4)-tau(k));
    rho(k+1) = (mu_f==0)*rho(1)*(2+k)+ (mu_f~=0)*rho(1)/tau(k+1)^2;
    eta(k) = (mu_f==0)*rho(1)*k/K+ (mu_f~=0)*rho(k);
    L(k)= (mu_f==0)*L(1)+(mu_f~=0)*2*(l_f+constantB*l_g + rho(k)*M_g^2);
    L(k+1)= (mu_f==0)*L(1)+(mu_f~=0)*2*(l_f+constantB*l_g + rho(k+1)*M_g^2);
    beta(k+1) = (mu_f==0)*(1-tau(k))*tau(k+1)/tau(k) + (mu_f~=0)*(1-tau(k))*tau(k)*L(k)/(tau(k)^2*L(k)+L(k+1)*tau(k+1));
    %Grad for projection (uncomment this for having l1 norm as a part of f)
    %stoch_grad  = Q*x_hat(:,k)+c+l*sign(x_hat(:,k)) + normrnd(0,sqrt(sigma),n,1); 
    %Grad for prox oracle (uncomment this for having l1 norm separate of f)
    stoch_grad  = Q*x_hat(:,k)+c + normrnd(0,sqrt(sigma),n,1); 
 for j=1:m 
     grad_g(:,j) = A(:,:,j)*x_hat(:,k) + B(:,j); 
     g_x_hat(j,k) =  0.5*x_hat(:,k)'*A(:,:,j)*x_hat(:,k) + x_hat(:,k)'*B(:,j)-d(j);
     u(j) =g_x_hat(j,k) - grad_g(:,j)'*x_hat(:,k) -(1-tau(k))*v(j,k) + y_tilde(j,k)/rho(k);
 end
w_init  = x_hat(:,k); 
%[x(:,k+1),iter_proj(k)] = proj(w_init,x_hat(:,k),L(k),rho(k),stoch_grad,grad_g,u,error,D_x,n);
[x(:,k+1),s(:,k+1),iter_proj(k)] = proj(w_init,x_hat(:,k),L(k),rho(k),stoch_grad,grad_g,u,error,D_x,n,l);
ell_g(:,k+1) = g_x_hat(:,k) + grad_g'*(x(:,k+1)-x_hat(:,k));
 for j=1:m 
     v(j,k+1) =  0.5*x(:,k+1)'*A(:,:,j)*x(:,k+1) + x(:,k+1)'*B(:,j)-d(j)-s(j,k+1);
     y_tilde(j,k+1) = y_tilde(j,k) + eta(k)*(ell_g(j,k+1)-s(j,k+1)-(1-tau(k))*(v(j,k)-s(j,k)));
 end
 x_hat(:,k+1) = x(:,k+1)+ beta(k+1)*(x(:,k+1)-x(:,k));

 iteration_time_aug_conex(k) = toc(tStart_auge_conex);
end
x_last_iter = x(:,1:K-1);
%% Measures 
feasibility = zeros(m,K-1);
for k=1:K-1
    optim_gap(k) = 0.5*x(:,k)'*Q*x(:,k) + x(:,k)'*c+ l*norm(x(:,k),1)-...
        (0.5*x_optim'*Q*x_optim + x_optim'*c+ l*norm(x_optim,1));
    for j=1:m 
         feasibility(j,k) = 0.5*x(:,k)'*A(:,:,j)*x(:,k) + x(:,k)'*B(:,j)-d(j);
    end
    feasib_gap(k) = norm(max(feasibility(:,k),0));
    norm_y_tilde(k) = norm(y_tilde(:, k));
    norm_s (k) = norm(s(:,k));
end









