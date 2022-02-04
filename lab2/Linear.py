import numpy as np
import matplotlib.pyplot as plt

class Linear:
    def __init__(self,nin,nout):
        self.W = np.random.uniform(-1, 1, (nout,nin))
        self.b = np.random.uniform(-1, 1, (1,nout))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.d = nin
        
    def forward(self, x):
        self.x = x
        return np.dot(x, self.W.T) + self.b

    def loss(self, out, y):
        return np.sum((y-out)**2)
    
    def backward(self, z, y):
        self.dW = self.x* (y-z)
        self.db = y-z
    
    def update(self,lr=0.01):
        self.W += lr*self.dW
        self.b += lr*self.db

    def fit(self, X, Y = [], ep = 50, loss=0.01):
        for i in range(ep):
            ls = 0.
            for j in range(0, len(X)-self.d-1):
                x = X[j: j+self.d] 
                z = self.forward(x)
                if len(Y):
                    self.backward(z,Y[j+self.d])
                    self.update()
                    ls+= self.loss(z,Y[j+self.d])
                else:
                    self.backward(z,X[j+self.d])
                    self.update()
                    ls+= self.loss(z,X[j+self.d])
            print('epoch: {}\t  loss:{}\n'.format(i, ls/len(X)))
            if(ls/len(X) <= loss):
                break
        self.lastX = X[-self.d:]
        print('W = {}\nb = {}'.format(self.W, self.b))

    def predict(self, n=10):
        preds = np.array([])
        for i in range(n):
            z = self.forward(self.lastX)
            preds = np.append(preds,z)
            self.lastX = np.append(self.lastX[1:],z)
        return preds

    def predict3(self,X):
        preds = np.array([])
        for j in range(0, len(X)-self.d-1):
            x = X[j: j+self.d] 
            z = self.forward(x)
            preds = np.append(preds,z)
        return preds
##### PARAMS
deep = 5
step = 0.025
step2 = 0.02
n_preds = 10

##### TASK1

X1 = np.arange(0, 4.5, step)
Y1 = np.cos(-2*X1**2+7*X1) - 0.5*np.cos(X1)
net = Linear(deep,1)
net.fit(Y1, loss = 0.001)
nextY1 = net.predict(1)
apro = net.predict3(Y1)

##### TASK2

net2 = Linear(deep,1)
net2.fit(Y1, loss = 0.001)
nextY2 = net.predict(n_preds)


##### TASK3

X2 = np.arange(0,4.,step2)
Y2 = np.sin(X2**2)
Y3 = np.sin(X2**2 + np.pi/2)/3
net3 = Linear(deep,1)
net3.fit(Y2, Y3, loss = 0.001)
predY3 = net3.predict3(Y2)


##### PLOTS

plt.plot(X1,Y1)
plt.plot(X1[deep+1:], apro, '.')
plt.plot(4.5+step, nextY1, '.')
plt.show()


plt.plot(X1,Y1, '-')
nextX1 = np.arange(4.5, 4.5+step*n_preds, step)
plt.plot(np.append(X1[-1],nextX1),np.append(Y1[-1],nextY2), '-')
plt.plot(np.append(X1[-1],nextX1),np.append(Y1[-1],nextY2), '.')
plt.show()


plt.plot(X2[deep+1:], predY3)
plt.plot(X2,Y3)
plt.show()