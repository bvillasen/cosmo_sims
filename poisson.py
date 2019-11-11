import numpy as np



N_total = 20
N_real = N_total-2
data = np.zeros( [N_total, N_total] )
# data = np.random.rand( N_total ,N_total ) * 2 -1

data[0,:] = 1
data[N_total-1,:] = 1
data[:,0] = 1
data[:,N_total-1] = 1




kmin = np.sin(np.pi/2.0/N_real)**2 + np.sin(np.pi/2.0/N_real)**2;
kmax = 2;

#Calculate max number of iterations, M
M=0;
sig = 1e-4;
kap0 =-(1 + kmin/kmax)/(1-kmin/kmax);
T0 = 1;
T1 = kap0;
res = 1/np.abs(T1);
while (res>sig):
    temp = 2.*kap0*T1-T0;
    T0 = T1;
    T1 = temp;
    res = 1./np.abs(T1);
    M+=1;

print("M = ", M);

n_iterations = M*2


for iter in range(n_iterations):
  # w = 1.5
  w = 2.0/((kmax + kmin) - (kmax-kmin) * np.cos( np.pi * ( 2 * (iter+1) - 1) / (2. * M) ) );


  data_c = data[1:N_total-1, 1:N_total-1] #center
  data_u = data[0:N_total-2, 1:N_total-1] #up 
  data_d = data[2:N_total,   1:N_total-1] #down
  data_l = data[1:N_total-1, 0:N_total-2] #left
  data_r = data[1:N_total-1,   2:N_total] #right

  data_iter = (1. - w)*data_c + w/4.*(data_l + data_r + data_u + data_d)
  data_old = data[1:N_total-1, 1:N_total-1]
  residual = np.abs( data_old  - data_iter).max()
  print( "Residual: ", residual )
  data[1:N_total-1, 1:N_total-1] = data_iter