import numpy as np
import matplotlib.pyplot as plt

def plot_decision_surface(W_ji, W_kj):
    x_range = np.linspace(-2, 2, 100)
    y_range = np.linspace(-2, 2, 100)
    xx, yy = np.meshgrid(x_range, y_range)
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    predictions = []

    for point in grid_points:
        x_m = np.array([1, point[0], point[1]])
        
        net_j = np.dot(W_ji, x_m)
        y_j = np.tanh(net_j)
        y_aug = np.insert(y_j, 0, 1)
        
        net_k = np.dot(W_kj, y_aug)
        y_k = np.tanh(net_k)
        
        predictions.append(y_k[0])

    zz = np.array(predictions).reshape(xx.shape)

    plt.figure(figsize=(8, 6))

    contour = plt.contourf(xx, yy, zz, levels=50, cmap='RdBu', alpha=0.8)
    plt.colorbar(contour, label='Network Output (y_k)')

    xor_inputs = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    for i in range(4):
        plt.scatter(xor_inputs[i, 0], xor_inputs[i, 1], 
                    c='white')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True)
    plt.show()

def forward(x):
    y = np.zeros(len(x))
    for m in range(len(x)):
        x_m = x[m]

        net_j = np.dot(W_ji, x_m)
        y_j = f(net_j)
        y_aug = np.insert(y_j, 0, 1) 
        
        net_k = np.dot(W_kj, y_aug)
        y_k = f(net_k)
        y[m] = y_k[0]
    return y

learningrate = 0.1
threshhold = 0.001

xd = np.array([
    [1, -1, -1],
    [1, -1,  1],
    [1,  1, -1],
    [1,  1,  1]
])

t = np.array([-1, 1, 1, -1])

f = lambda x: np.tanh(x)
df = lambda x: 1-(f(x))**2

W_ji = np.array([
    [0.1, 0.5, 0.2],
    [0.1, 0.5, 0.2]
    ])

W_kj = np.array([[0.3, 0.2, 0.1]])

MSE = 0 
history = []
epochs = 0

while True:
    deltaW_ji = np.zeros_like(W_ji)
    deltaW_kj = np.zeros_like(W_kj)
    J_w = 0
    
    for m in range(len(xd)):
        x_m = xd[m]
        t_k = t[m]
        
        net_j = np.dot(W_ji, x_m)
        y_j = f(net_j)
        y_aug = np.insert(y_j, 0, 1) 
        
        net_k = np.dot(W_kj, y_aug)
        y_k = f(net_k)
        
        e = t_k - y_k
        J_w += 0.5 * (e ** 2)
        
        delta_k = df(net_k) * e
        delta_j = df(net_j) * (W_kj[0, 1:] * delta_k)
        
        deltaW_ji += learningrate * np.outer(delta_j, x_m)

        deltaW_kj += learningrate * delta_k * y_aug
        
        
    W_kj += deltaW_kj
    W_ji += deltaW_ji
    
    MSE = J_w / len(xd)
    history.append(MSE[0])
    epochs += 1

    if MSE < threshhold:
        break

print(forward(x=xd))
print(t)
print(f'Number of Epochs = {epochs}')

plt.plot(history, linewidth=2)
plt.grid(True)
plt.xlabel('# of Epochs')
plt.ylabel('MSE')
plt.show()

plot_decision_surface(W_ji, W_kj)