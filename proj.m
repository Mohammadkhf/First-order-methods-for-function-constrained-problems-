function [fixed_point,s,iter]  =  proj(w_init,x_hat,L,rho,stoch_grad,grad_g,u,error,D_x,n,l)
iter=0;
fixed_point = w_init; % Initial guess for fixed point
previous_fixed_point = inf; % Initialize previous fixed point
while abs(fixed_point - previous_fixed_point) > error
    previous_fixed_point = fixed_point;
    
    % Evaluate the projection point
    eval_point = x_hat - (stoch_grad + rho * grad_g * max(0, u + grad_g' * fixed_point)) / L;
    
    %Projection onto l2 ball (uncomment if l1 norm is a part of f)
    %fixed_point = eval_point * min (1, D_x/norm(eval_point));
    
    %Prox with l1 norm with l2 ball constraint (uncomment if l1 norm is
    %separate from f)
    unconstrained_min = max(eval_point - l/L,0) -  max(-eval_point -l/L,0);
    fixed_point = unconstrained_min*min (1, D_x/norm(unconstrained_min));
    
    iter = iter+1;
end
s  = min(0, u + grad_g' * fixed_point);