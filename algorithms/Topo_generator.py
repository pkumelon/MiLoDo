import torch


def generate_metropolis_weights(topo_A):
    A = topo_A
    n = A.shape[0]
    W = torch.zeros_like(A)
    for i in range(n):
        for j in range(n):
            if A[i, j] > 0:
                W[i, j] = 1 / (1 + max(torch.sum(A[i]), torch.sum(A[j])))

    for i in range(n):
        W[i, i] = 1 - torch.sum(W[i]) + W[i, i]
        
    # print(W)    
    return W

def generate_fully_connected_topology(n):

    # Create a matrix filled with ones, but subtract the identity matrix to exclude self-connections
    A = torch.ones((n, n)) 
    
    # Normalize the matrix so that the sum of weights for each node (excluding self-connections) is 1
    # W = A / (n - 1)
    return A


def generate_ring_topology(n):
    A = torch.zeros((n, n))
    for i in range(n):
        A[i, i] = 1
        A[i, (i - 1) % n] = 1
        A[i, (i + 1) % n] = 1
    return A

def generate_ring_W(n):
    A = generate_ring_topology(n)
    W = A / 3
    return W