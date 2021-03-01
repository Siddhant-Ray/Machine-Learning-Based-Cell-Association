import pandas as pd
import numpy as np
import csv

#HMM based learning for cell association


# The forward algorithm
def forward(V, a, b, initial_distribution):
    alpha = np.zeros((V.shape[0], a.shape[0]))
    alpha[0, :] = initial_distribution * b[:, V[0]]

    for t in range(1, V.shape[0]):
        for j in range(a.shape[0]):
            # Matrix Computation Steps
            #                  ((1x2) . (1x2))      *     (1)
            #                        (1)            *     (1)
            alpha[t, j] = alpha[t - 1].dot(a[:, j]) * b[j, V[t]]

    return alpha

# The backward algorithm
def backward(V, a, b):
    beta = np.zeros((V.shape[0], a.shape[0]))

    # setting beta(T) = 1
    beta[V.shape[0] - 1] = np.ones((a.shape[0]))

    # Loop in backward way from T-1 to
    # looping will be from T-2 to 0
    for t in range(V.shape[0] - 2, -1, -1):
        for j in range(a.shape[0]):
            beta[t, j] = (beta[t + 1] * b[:, V[t + 1]]).dot(a[j, :])

    return beta

# HMM learning algorithm
def baum_welch(V, a, b, initial_distribution, n_iter=100):
    M = a.shape[0]
    T = len(V)

    for n in range(n_iter):
        alpha = forward(V, a, b, initial_distribution)
        beta = backward(V, a, b)

        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            denominator = np.dot(np.dot(alpha[t, :].T, a) * b[:, V[t + 1]].T, beta[t + 1, :])
            for i in range(M):
                numerator = alpha[t, i] * a[i, :] * b[:, V[t + 1]].T * beta[t + 1, :].T
                xi[i, :, t] = numerator / denominator

        gamma = np.sum(xi, axis=1)
        a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))

        # Add additional T'th element in gamma
        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))

        K = b.shape[1]
        denominator = np.sum(gamma, axis=1)
        for l in range(K):
            b[:, l] = np.sum(gamma[:, V == l], axis=1)

        b = np.divide(b, denominator.reshape((-1, 1)))

    return a,b

number_of_devices=[150,250,350,450,550,600]

P_list=[]

for number in number_of_devices:
    name='data_python_%s.csv'%number
    data = pd.read_csv(name)

    V = data['Observational states'].values

    # Transition Probabilities
    a = np.ones((2, 2))
    a = a / np.sum(a, axis=1)

    # Emission Probabilities
    b = np.array(((1, 3, 5), (2, 4, 6)))
    b = b / np.sum(b, axis=1).reshape((-1, 1))

    # Equal Probabilities for the initial distribution
    initial_distribution = np.array((0.5, 0.5))

    alpha = forward(V, a, b, initial_distribution)
    #print(alpha)
    Po=sum(alpha)
    #print(Po)
    Po_givenlambda=sum(Po)
    #print(Po_givenlambda)

    Pth= 0.15 #Value given as per data studied

    print(baum_welch(V, a, b, initial_distribution, n_iter=100))

    P_list.append(Po_givenlambda)

print(P_list)

P_available=[]

for i in range(0,len(P_list)):
    P_available.append(1- P_list[i])

print(P_available)

# HMM decoding algorithm
def viterbi(V, a, b, initial_distribution):
    T = V.shape[0]
    M = a.shape[0]

    omega = np.zeros((T, M))
    omega[0, :] = np.log(initial_distribution * b[:, V[0]])

    prev = np.zeros((T - 1, M))

    for t in range(1, T):
        alpha = forward(V, a, b, initial_distribution)
        beta = backward(V, a, b)
        for j in range(M):
            probability = omega[t - 1] + np.log(a[:, j]) + np.log(b[j, V[t]])

            #Most probable state given previous state at time t(1)
            prev[t - 1, j] = np.argmax(probability)

            #Probability of the most probable state(2)
            omega[t, j] = np.max(probability)

    # Path Array
    S = np.zeros(T)
    # Find the most probable last hidden state
    last_state = np.argmax(omega[T - 1, :])

    S[0] = last_state

    backtrack_index = 1
    for i in range(T - 2, -1, -1):
        S[backtrack_index] = prev[i, int(last_state)]
        last_state = prev[i, int(last_state)]
        backtrack_index += 1

    S = np.flip(S, axis=0)

    # Convert numeric values to actual hidden states
    result = []
    for s in S:
        if s == 0:
            result.append("Cell A is selected")
        else:
            result.append("Cell B is selected")

    return result

number_of_devices=[150,250,350,450,550,600]

count_cellA=0
count_cellB=0
res_cellA=[]
res_cellB=[]

for number in number_of_devices:
    name='data_python_%s.csv'%number
    data = pd.read_csv(name)

    V = data['Observational states'].values

    # Transition Probabilities
    a = np.ones((2, 2))
    a = a / np.sum(a, axis=1)

    # Emission Probabilities
    b = np.array(((1, 3, 5), (2, 4, 6)))
    b = b / np.sum(b, axis=1).reshape((-1, 1))

    # Equal Probabilities for the initial distribution
    initial_distribution = np.array((0.5, 0.5))

    alpha = forward(V, a, b, initial_distribution)
    file_name='output_%s.txt'%number
    f = open(file_name, "w")
    output=viterbi(V, a, b, initial_distribution)
    for i in output:
        if i=="Cell A is selected":
            count_cellA+=1
        else:
            count_cellB+=1
        f.write(i+"\n")
    f.close()
    res_cellA.append(count_cellA)
    res_cellB.append(count_cellB)
    count_cellA=0
    count_cellB=0


print(res_cellA)
print(res_cellB)

import matplotlib.pyplot as plt
plt.plot(number_of_devices, P_available, marker='X', linestyle='--', color='green')
plt.title("Channel Availability for varying MTC device number")
plt.grid(True)
plt.axis([150,600, 0.675, 0.875])
plt.legend(['HMM based cell selection'], loc='upper right')
plt.xlabel("Number of MTC devices")
plt.ylabel("Channel Availability")
plt.show()

plt.plot(number_of_devices, res_cellA, 'r--', number_of_devices, res_cellB, 'b' )
plt.title("Frequency of selected cell for varying MTC device number")
plt.grid(True)
plt.axis([150,600, 30, 420])
plt.legend(['Cell A','Cell B'],loc='upper left')
plt.xlabel("Number of MTC devices")
plt.ylabel("Number of times cell selected")
plt.show()