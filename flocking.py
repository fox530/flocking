from pylab import *
from numpy import random


def phi_alpha(z):
    global r_alpha, d_alpha
    return rho_h(z/r_alpha) * phi(z-d_alpha)

def phi(z):
    global a,b,c
    return 1./2*((a+b)*(z+c)/sqrt(1+(z+c)**2) + (a-b) )

def rho_h(z):
    global h
    if (z>=0) and (z<h):
        return 1.;
    elif (z>=h) and (z<=1):
        return 1./2*(1+cos(pi*(z-h)/(1-h)))
    else:
        return 0.;

def sigma_norm(z):
    global epsilon
    return 1/epsilon*(sqrt(1+epsilon * (norm(z)**2) ) - 1)

def find_neighbor(x, iagent):
    global r, N
    neighbor=[];
    for inbor in range(N):
        pdiv = x[inbor,0:2] - x[iagent, 0:2]
        if (inbor != iagent) and norm(pdiv)<r:
            neighbor = hstack((neighbor, inbor))
    return neighbor
        


dt = 0.01; 
T = 5;
n_iter = int(T/dt);

N = 6;
n = 4;

A=array([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]]);
B=eye(4,2,-2);
Adj=array([[0,1,1],[1,0,0],[1,0,0]]);
x=zeros((N, n_iter,4))
ptgt=zeros((n_iter,2))
vtgt=ptgt.copy();

## uncertainty: theta
theta = eye(2,4,2)*0.1;

h=0.9
a=5.; b=5.; c=abs(a-b)/sqrt(4*a*b);
d=7.; r=1.2*d;
epsilon=0.1;
d_alpha = sigma_norm(d);
r_alpha = sigma_norm(r);
c1=1;c2=1;
##initialization
x[:,0,0:2]=random.uniform(-5,5,(N,2));
ptgt[0]=random.uniform(-10,10,2);
vtgt[0]= array([0.1,0.1]) + random.normal(0.5,10,2)

for k in range(1, n_iter):
    
    ## emulate target
    vtgt[k] = array([0.1,0.1])+random.normal(0,0.1,2);
    ptgt[k] = ptgt[k-1] + dt*vtgt[k-1];
    
    ## system update
    for iagent in range(N):
        # neighbor=nonzero(Adj[iagent]>0)[0];
        neighbor = find_neighbor(x[:,k-1,:], iagent);
        con_a, con_b, con_c = zeros(2), zeros(2), zeros(2)
        con_c = -c1*(x[iagent,k-1,0:2] - ptgt[k-1]) - c2*(x[iagent,k-1,2:4] - vtgt[k-1]);
        for inbr in neighbor:
            pdif = x[inbr,k-1,0:2] - x[iagent,k-1,0:2];
            vdif = x[inbr,k-1,2:4] - x[iagent,k-1,2:4];

            con_a = con_a + phi_alpha(sigma_norm(pdif) ) * pdif/sqrt(1 + epsilon*(norm(pdif)**2));
            con_b = con_b + rho_h(sigma_norm(pdif)/r_alpha )*vdif;

        u = con_a + con_b + con_c;
        theta_x = dot(theta, x[iagent,k-1]); ## uncertainty term
        xdot = dot(A, x[iagent,k-1]) + dot(B,u + theta_x);
        x[iagent,k] = x[iagent,k-1] + dt*xdot;



################################################### Plot Results
close('all')
## 2-D Trajectory
figure()
color=('r','g','b','c','m','y','w','k')

for iagent in range(N):
    plot(x[iagent,:,0],x[iagent,:,1],color[iagent])
    plot(x[iagent,::n_iter/4,0], x[iagent,::n_iter/4,1], color[iagent]+'o')

plot(ptgt[:,0], ptgt[:,1],'k:')
plot(ptgt[::n_iter/4,0], ptgt[::n_iter/4,1], 'ko')
title('2D trajectory')

## Position and Velocity of x dimension
figure()
subplot(211)
title('Time History of Position and Speed in x direction')
for iagent in range(N):
    plot(x[iagent,:,0], color[iagent])
plot(ptgt[:,0],'k:')

subplot(212)
for iagent in range(N):
    plot(x[iagent,:,2], color[iagent])
plot(vtgt[:,0],'k:')

xlabel('Time')

show()
