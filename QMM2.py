import numpy as np

def generate_lambda_matrix(n):
    #Generate the lambda matrix with the correct dimensions based on n
    return np.random.uniform(-1, 1, (3*(n-1)+3*n, 3*(n-1)+3*n))

def construct_hamiltonian(n, theta, lambda_matrix):
    # Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    #Initialize the Hamiltonian for n qubits
    H = np.zeros((2**n, 2**n), dtype=np.complex128)
    
    #Interaction term
    for i in range(n):
        for P, sigma in zip([sigma_x, sigma_y, sigma_z], [sigma_x, sigma_y, sigma_z]):
            for j in range(n):
                H_ij = np.kron(np.eye(2**i, dtype=np.complex128), np.kron(sigma, np.eye(2**(n-1-i), dtype=np.complex128)))
                H += theta * lambda_matrix[i*n + j] * H_ij  #Interaction term adjusted by theta

    #Measurement in the ground state
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    ground_state = eigenvectors[:, np.argmin(eigenvalues)]
    return ground_state

def calculate_qfi(ground_state):
    #To be continued
    return np.var(ground_state)  # Simplified example, will be replaced with actual QFI calculation

def main():
    n = int(input("Enter the number of qubits (N): "))  # User input for number of qubits
    theta = 0.5  #Example value for theta, the parameter of interest, by defult we set it 1/2
    lambda_matrix = generate_lambda_matrix(n)
    ground_state = construct_hamiltonian(n, theta, lambda_matrix)
    qfi = calculate_qfi(ground_state)
    print(f"Quantum Fisher Information: {qfi}")


main()
