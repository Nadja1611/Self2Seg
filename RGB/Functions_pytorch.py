


from __future__ import division
import numpy as np
import os
#import matplotlib.pyplot as plt
import torch



'''----input has to be torch tensor of shape ([channels,h,w]), output has size [channels,2,h,w]----'''
def gradient(img):
    z1 = torch.zeros_like(img)
    z2 = torch.zeros_like(img)
    g2 = img[:,:,1:]-img[:,:,:-1]
    g1 = img[:,1:,:]-img[:,:-1,:]
    z2[:,:,:-1] = g2
    z1[:,:-1,:] = g1

    return torch.stack([z1,z2], 1)



def div(grad):
    z1 = grad[:,0]
    z2 = grad[:,1]
    #dx,dy = torch.gradient(grad,dim = [-1,-2])
    dx = grad[:,0,1:-1,:]-grad[:,0,:-2,:]
    dy = grad[:,1,:,1:-1]-grad[:,1,:,:-2]

    #z1 = dx[:,0]
    #z2 = dy[:,1]
    z1[:,1:-1,:] = dx
    z2[:,:,1:-1] = dy
    z1[:,-1,:] = -grad[:,0,-2,:]
    z2[:,:,-1] = -grad[:,1,:,-2]
    return z1 + z2
    


def psi(x, mu):
    '''
    Huber function needed to compute tv_smoothed
    '''
    res = torch.abs(x)
    m = res < mu
    res[m] = x[m]**2/(2*mu) + mu/2
    return res


def huber(x,mu):
    return torch.sum(psi(x,mu), dim = [-1,-2])







#since the adjoint of the l1-norm, the resolvent operator reduces to 
# pointwise euclidean projectors onto l2-balls
def proj_l1_grad(g, Lambda):
    '''
    proximity operator of l1
    '''
    L = Lambda*torch.ones_like(g[:,0])
   # g=g/L
    #g = torch.minimum(torch.sqrt(g[:,0]**2+g[:,1]**2), L)
    n = torch.maximum(torch.sqrt(g[:,0]**2+g[:,1]**2),L)
    g[:,0] = g[:,0]/n
    g[:,1]= g[:,1]/n #g/max(alpa, |g|)
    g = Lambda*g
    return g

def proj_l1(g, beta=torch.tensor(1.0)):
    '''
    proximity operator of l1
    '''
    n = torch.maximum(torch.abs(g), torch.tensor(beta))
    g = g/n
    #res = np.concatenate((res1,res2), axis = 0)
    return g

def proj_unitintervall(g):
    '''proximity operator of indicator function'''
    g[g>1] = 1
    g[g<0] = 0
    return g


def epsilon_approx(u):
    return torch.sum(torch.sqrt(u**2 + 0.0000001))
    
def norm1(mat):
    return torch.sum(torch.abs(mat))

#skalarprdoukt
def mydot(mat1, mat2):
    return np.dot(mat1.ravel(), mat2.ravel())

mu = 0.000001


def Fid1(u,f):
   '''
   Computes M1(u) u*(f-((u,f)/|u|)**2)
   '''
   Hub = img_dot(u,f)/epsilon_approx(u)
   return u*(f-(torch.ones_like(f)*(Hub[:,None,None])))**2


# def Fid2(img1,img2):
#    '''
#    Computes M1(u)
#    '''
#    Hub = img_dot((torch.ones_like(img1)-img1),img2)/(huber(torch.ones_like(img1)-img1,mu))
#    return ((torch.ones_like(img1)-img1)*(img2-torch.ones_like(img2)*Hub[:,None,None])**2)

def Fid2(img1,img2):
   '''
   Computes M1(u)
   '''
   Hub = img_dot((torch.ones_like(img1)-img1),img2)/(epsilon_approx(torch.ones_like(img1)-img1))
   return ((torch.ones_like(img1)-img1)*(img2-torch.ones_like(img2)*Hub[:,None,None])**2)


def mymulti(a,b,h):
    '''a = M^T,b = factor2'''
    prod = np.zeros_like(h)
    for i in range(np.shape(a)[0]):
        prod1 = a[i]*b
        prod[i] = mydot(prod1,h)
    return prod    


def img_dot(img1,img2):
    '''---------scalar product of two images with multiple channels----'''
    prod = torch.sum(img1*img2, dim = [-1,-2])
    return prod

# def adjoint_der_Fid1(img1,img2,h):
#     '''Computes adjoint of derivative and applies it on h T^2*h + 2*M^T*U*T*h'''
#     div = (img_dot(img1,img2)/huber(img1,0.000001))
#     div2 = img_dot(img1,img2)/huber(img1,0.000001)**2
#     T_f = (img2 -div[:,None,None]*torch.ones_like(img2))
#     S_1 = T_f**2*h   
#     ip = img_dot(img1,img2)
#     '''reshape to column'''
#     Hub2 = huber(img1,0.000001)
#     Hub3 = huber(img1,0.000001)
#     M_T = (-img2*Hub2[:,None,None] *torch.ones_like(img2) + torch.sign(img1)*ip[:,None,None])/Hub3[:,None,None]**2
#     '''computation of U*T pointwise'''
#     factor2 = img1*T_f
#     S_2 = 2*img_dot(factor2,h)[:,None,None]*M_T
#     S_gesamt = S_1+ S_2
#     return S_gesamt
    
def adjoint_der_Fid1(img1,img2,h):
#     '''Computes adjoint of derivative and applies it on h T^2*h + 2*M^T*U*T*h'''
    div = (img_dot(img1,img2)/epsilon_approx(img1))
    div2 = img_dot(img1,img2)/epsilon_approx(img1)**2
    T_f = (-img2 +div[:,None,None]*torch.ones_like(img2))
    S_1 = T_f**2*h
    ip = img_dot(img1,img2)
    '''reshape to column'''
    Hub2 = epsilon_approx(img1)
    Hub3 = epsilon_approx(img1)
    M_T = (img2*Hub2 *torch.ones_like(img2) - (img1*ip[:,None,None]/Hub2))/Hub3**2
    '''computation of U*T pointwise'''
    factor2 = img1*T_f
    S_2 = -2*img_dot(factor2,h)[:,None,None]*M_T
    S_gesamt = S_1+ S_2
    return S_gesamt



    
def Fid1_op(img1,img2):
#     '''Computes adjoint of derivative and applies it on h T^2*h + 2*M^T*U*T*h'''
    div = (img_dot(img1,img2)/epsilon_approx(img1))
    div2 = img_dot(img1,img2)/epsilon_approx(img1)**2
    T_f = (-img2 +div[:,None,None]*torch.ones_like(img2))
    S_1 = T_f**2
    ip = img_dot(img1,img2)
    '''reshape to column'''
    Hub2 = epsilon_approx(img1)
    Hub3 = epsilon_approx(img1)
    M_T = (img2*Hub2 *torch.ones_like(img2) - (img1*ip[:,None,None]/Hub2))/Hub3**2
    '''computation of U*T pointwise'''
    factor2 = img1*T_f
    S_2 = -2*factor2[:,None,None]*M_T
    S_gesamt = S_1+ S_2
    return S_gesamt
# def adjoint_der_Fid2(img1,img2,h):
#     '''Computes adjoint of derivative'''
#     # h= np.reshape(h,(np.shape(img1.ravel())[0],1))
#     div = (img_dot((torch.ones_like(img1)-img1),img2)/huber((torch.ones_like(img1)-img1),0.000001))
#     div2 = img_dot(1-img1,img2)/huber(1-img1,0.000001)**2
#     T_f = (img2 -div[:,None,None]*torch.ones_like(img2))
#     S_1 = -T_f**2*h   
#     ip = img_dot(1-img1,img2)

#     Hub2 = huber(1-img1,0.000001)
#     Hub3 = huber(1-img1,0.000001)
#     M_T = (img2*Hub2[:,None,None]*torch.ones_like(img2) - torch.sign(1-img1)*ip[:,None,None])/Hub3[:,None,None]**2
#     factor2 =(1-img1)*T_f
#     S_2 = 2*img_dot(factor2, h)[:,None,None]*M_T
#     S_gesamt = S_1 + S_2
#     return S_gesamt



def adjoint_der_constant_bg(img1,img2,h):
    '''Computes adjoint of derivative'''
    # h= np.reshape(h,(np.shape(img1.ravel())[0],1))
    ip = img_dot(1-img1,img2)
    Hub2 = epsilon_approx(1-img1)
    Hub3 = epsilon_approx(1-img1)
    M_T = (-img2*Hub2*torch.ones_like(img2) - (-1+img1)*ip[:,None,None]/Hub2)/Hub3**2
    return M_T

def adjoint_der_Fid2(img1,img2,h):
    '''Computes adjoint of derivative'''
    # h= np.reshape(h,(np.shape(img1.ravel())[0],1))
    div = (img_dot((1-img1),img2)/epsilon_approx((1-img1)))
    div2 = img_dot(1-img1,img2)/epsilon_approx(1-img1)**2
    T_f = (-img2 +div[:,None,None]*torch.ones_like(img2))
    S_1 = -T_f**2*h   
    ip = img_dot(1-img1,img2)

    Hub2 = epsilon_approx(1-img1)
    Hub3 = epsilon_approx(1-img1)
    M_T = (-img2*Hub2*torch.ones_like(img2) - (-1+img1)*ip[:,None,None]/Hub2)/Hub3**2
    factor2 =(1-img1)*T_f
    S_2 = 2*img_dot(factor2, h)[:,None,None]*M_T
    S_gesamt = S_1 + S_2
    return S_gesamt    

def Fid2_op(img1,img2):
    '''Computes adjoint of derivative'''
    # h= np.reshape(h,(np.shape(img1.ravel())[0],1))
    div = (img_dot((torch.ones_like(img1)-img1),img2)/epsilon_approx((torch.ones_like(img1)-img1)))
    div2 = img_dot(1-img1,img2)/epsilon_approx(1-img1)**2
    T_f = (-img2 +div[:,None,None]*torch.ones_like(img2))
    S_1 = -T_f**2
    ip = img_dot(1-img1,img2) #<1-u,f>
    Hub2 = epsilon_approx(1-img1)
    Hub3 = epsilon_approx(1-img1)
    M_T = (-img2*Hub2*torch.ones_like(img2) - (-1+img1)*ip[:,None,None]/Hub2)/Hub3**2
    factor2 =(1-img1)*T_f
    S_2 = 2*factor2[:,None,None]*M_T
    S_gesamt = S_1 + S_2
    return S_gesamt


def indi(img1, eps):
    '''Computes indicator function on convex set'''
    if np.max(img1)<=1+eps and np.min(img1)>=-0.01:
        return 0
    else:
        return float('inf')
    
    
# #projection operator for nonoverlaüping channels   , das ist aber für sum |u|^2 ACHTUNG
def proj_unitball(g):
#     '''
#     proximity operator of l1
#     '''
     g = np.asarray(g)
     res = np.copy(g)
     n = np.maximum(np.sum(np.abs(g), 0), 1.0)
     res = res/n
     #res = np.concatenate((res1,res2), axis = 0)
     return res







def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0 
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex"""
    c,h,f = v.shape
    v = v.flatten(start_dim = 1)
    n = v.shape[0]  # will raise ValueError if v is not 1-D
    
    index = (torch.sum(v,0) <= s) *(torch.min(v,0)[0]>=0)

    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = torch.sort(v, descending = True, dim = 0)[0]
    cssv = torch.cumsum(u,0)
    # get the number of > 0 components of the optimal solution
    a = (u * torch.stack(v.shape[1]*[torch.arange(1, n+1).to(v.device)],1) > (cssv - s))
    rho = torch.sum(a,0) -1
    #torch.nonzero(u * torch.stack(v.shape[1]*[torch.arange(1, n+1)],1) > (cssv - s))[-1]
    arr=torch.arange(0,len(rho))
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho, arr]- s) / (rho+1)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    w[:,index] = v[:,index]
    w= torch.reshape(w,(c,h,f))
    return w


#application of function along channel axis

# def euclidean_proj_l1ball(v, s=1):
#     """ Compute the Euclidean projection on a L1-ball
#     Solves the optimisation problem (using the algorithm from [1]):
#         min_w 0.5 * || w - v ||_2^2 , s.t. || w ||_1 <= s
#     Parameters
#     ----------
#     v: (n,) numpy array,
#        n-dimensional vector to project
#     s: int, optional, default: 1,
#        radius of the L1-ball
#     Returns
#     -------
#     w: (n,) numpy array,
#        Euclidean projection of v on the L1-ball of radius s
#     Notes
#     -----
#     Solves the problem by a reduction to the positive simplex case
#     See also
#     --------
#     euclidean_proj_simplex
#     """
#     assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
#     n, = v.shape  # will raise ValueError if v is not 1-D
#     # compute the vector of absolute values
#     u = np.abs(v)
#     # check if v is already a solution
#     if u.sum() <= s:
#         # L1-norm is <= s
#         return v
#     # v is not already a solution: optimum lies on the boundary (norm == s)
#     # project *u* on the simplex
#     w = euclidean_proj_simplex(u, s=s)
#     # compute the solution to the original problem on v
#     w *= np.sign(v)
#     return w