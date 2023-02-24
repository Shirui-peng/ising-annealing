import numpy as np

def init_X(Nx,Ny):

    '''Create a Nx x Ny lattice with random spin configuration'''
    
    X0 = np.random.choice([1, -1], size=(Nx, Ny))
    return X0


def deltaJ(xi, Sxj, cij, theta, bi):

    '''Energy difference for a spin flip'''
    
    return 2 * xi * (theta * bi + cij * Sxj)


def metropolis(c, theta, b, T, niter=500000):

    '''Metropolis algorithm for simulating the Gibbs law.'''
    Nx,Ny = b.shape
    N = Nx*Ny
    X = np.sign(b)#init_X(Nx,Ny)
    energy = 0
    energies = []
    ie = []
    spins = []
    spin = np.sum(X)
    isp = []
    
    for n in range(niter):

        i = np.random.randint(N)
        y = np.mod(i, Ny)
        x = i // Ny

        # Periodic Boundary Condition
        Sxj = X[(x - 1) % Nx, y] + X[(x + 1) % Nx, y] + \
              X[x, (y - 1) % Ny] + X[x, (y + 1) % Ny]

        dJ = deltaJ(X[x, y], Sxj, c, theta, b[x,y])

        if dJ < 0 or np.random.random() < np.exp(-dJ/T):
            X[x, y] = -X[x, y]
            energy += dJ
            energies.append(energy)
            ie.append(n)
            spin += 2*X[x, y]

        if n % 10000 == 0:
            spins.append(spin)
            isp.append(n)
        
    return X, energies, spins, ie, isp