import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize
import warnings

# Convex Profile Inversion implementation of 
# Steven J. Ostro and Robert Connelly, *Convex Profiles from Asteroid Lightcruves*, 1983 

def v_func(n,phi):
    warnings.simplefilter('ignore', category=RuntimeWarning)
    l_n = len(n) if type(n)==np.ndarray else 1
    l_phi = len(phi) if type(phi)==np.ndarray else 1
    N = np.tile(n, (l_phi,1))
    PHI = np.tile(phi, (l_n,1)).T
    PSI = np.pi - PHI
    V = np.zeros((l_phi,l_n), dtype=complex)
    np.putmask(V, np.abs(N)==1, N*0.25*(1 + 2.j*PSI - np.exp(-n*2.j*PHI)))
    np.putmask(V, (np.abs(N)!=1)&(PHI>=0), 0.5*(np.exp(1.j*(N-1)*PSI)/(N-1) \
                                              - np.exp(1.j*(N+1)*PSI)/(N+1) \
                                            - 2/(N**2-1)))
    np.putmask(V, (np.abs(N)!=1)&(PHI<0), 0.5*(np.exp(-1.j*(N+1)*PHI)/(N+1) \
                                             - np.exp(-1.j*(N-1)*PHI)/(N-1) \
                                             + 2*(-1)**(N+1)/(N**2-1)))    
    if l_n*l_phi==1:
        return V[0,0]
    if l_n==1:
        return V[:,0]
    if l_phi==1:
        return V[0]
    return V

def radius2xy(radius, phase=None):
    if phase is None: phase = np.linspace(0,2*np.pi,len(radius))
    x_integrand = -radius*np.sin(phase)
    y_integrand = radius*np.cos(phase)
    x = cumulative_trapezoid(x_integrand, x=phase, initial=0)
    y = cumulative_trapezoid(y_integrand, x=phase, initial=0)
    return x, y

def plot_from_radius(radius, phase=None, ax=None, color=None):
    if phase is None: phase = np.linspace(0,2*np.pi,len(radius))
    x, y = radius2xy(radius, phase)
    if ax is None: fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(x,y, color=color)
    ax.set_aspect('equal')
    ax.set_axis_off()
    if ax is None: plt.show()

def cut_fft(fft, harmonics):
    return np.fft.ifftshift(np.fft.fftshift(fft)[len(fft)//2-harmonics:len(fft)//2+harmonics+1])

def sum_fft(fft, x):
    n = np.fft.fftfreq(len(fft), d=1/len(fft))
    return np.real(np.sum(np.tile(fft,(len(x),1)).T * np.exp(1.j*np.tile(x,(len(n),1))*np.tile(n,(len(x),1)).T), axis=0))

def radius2light_curve(radius, sun_angle):
    # sun_angle in rad
    r_fft = np.fft.fft(radius)/len(radius)
    n = np.fft.fftfreq(len(r_fft), d=1/len(r_fft))
    v = np.array([v_func(i, sun_angle) for i in n])
    I_fft = r_fft*v
    phase = np.linspace(0,2*np.pi*(1-1/len(radius)),len(radius))
    I = sum_fft(I_fft, phase)
    return phase, I/np.mean(I)

def light_curve2radius(light_curve, sun_angle, harmonics=None):
    I_fft = np.fft.fft(light_curve)
    if harmonics is not None: I_fft = cut_fft(I_fft, min(harmonics, (len(I_fft)-1)//2))
    n = np.fft.fftfreq(len(I_fft), d=1/len(I_fft))
    v = np.array([v_func(i, sun_angle) for i in n])
    r_fft = I_fft/v
    # Closed shape
    r_fft[1] = 0
    r_fft[-1] = 0
    phase = np.linspace(0,2*np.pi,len(I_fft))
    r = sum_fft(r_fft, phase)
    return phase, r/np.mean(r)

def real2complex(z):
    return z[:len(z)//2] + 1j * z[len(z)//2:]

def complex2real(z):
    return np.concatenate((np.real(z), np.imag(z)))

def light_curve2radius_cons(light_curve, sun_angle, harmonics=None, normalize=True, return_res=False):
    I_fft = np.fft.fft(light_curve)
    if harmonics is not None: I_fft = cut_fft(I_fft, min(harmonics, (len(I_fft)-1)//2))
    n = np.fft.fftfreq(len(I_fft), d=1/len(I_fft))
    v = np.array([v_func(i, sun_angle) for i in n])
    r_fft = I_fft/v
    phase = np.linspace(0,2*np.pi,len(I_fft))
    cons = [{'type':'eq', 'fun':lambda x: x[1]},
            {'type':'eq', 'fun':lambda x: x[1+len(x)//2]},
            {'type':'eq', 'fun':lambda x: x[-1]},
            {'type':'eq', 'fun':lambda x: x[-1+len(x)//2]},
            {'type':'ineq', 'fun':lambda x: np.min(sum_fft(real2complex(x), phase))}]
    r_fft_real = complex2real(r_fft)
    res = minimize(lambda x: np.linalg.norm(x-r_fft_real), r_fft_real, method='SLSQP', tol=1e-10, constraints=cons)
    r = sum_fft(real2complex(res.x), phase)
    if normalize: r = r/np.mean(r)
    if return_res: return phase, r, res
    return phase, r


def main():
    a = np.array([1+2.j, 3+4.j, 5+6.j])
    a_real = complex2real(a)
    print(a_real)
    print(a_real[len(a_real)//2])

if __name__=='__main__':
    main()