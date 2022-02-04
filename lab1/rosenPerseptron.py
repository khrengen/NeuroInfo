import numpy as np
import matplotlib.pyplot as plt

class Perseptron:
    def __init__(self,nin):
        self.W = np.random.uniform(-1, 1, nin)
        self.b = np.random.uniform(-1, 1)
        self.err = 0.
        
    def forward(self, x):
        return np.dot(x, self.W) + self.b
    
    def update(self,x,lr=0.1):
        self.W += lr * self.err * x
        self.b += lr * self.err

    def fit(self, X, Y):
        flag = True
        count = 0
        ep = 0
        while flag and ep < 50:
            for x,y in zip(X,Y):
                self.err = y - (0 if self.forward(x) < 0 else 1)
               
                if (self.err == 0):
                    count+=1
                    if count == len(X):
                        flag = False
                        break
                else:
                    self.update(x)
                    count = 0
            ep+=1


    def get_param(self):
        return self.W, self.b

    def predict(self, X):
       return [0 if self.forward(x) < 0 else 1 for x in X]

class Net2:
    def __init__(self, nin):
        self.per1 = Perseptron(nin)
        self.per2 = Perseptron(nin)

    def fit(self, X, y1, y2):
        self.per1.fit(X, y1)
        self.per2.fit(X, y2)

    def get_param(self):
        return [self.per1.get_param(), self.per2.get_param()]

    def predict(self, X):
        return [self.per1.predict(X), self.per2.predict(X)]

def accuracy(pred, Y):
    return (pred == Y).mean() 

##### TASK1
X = np.array([[-2.5, -4.8],[4.2,1.5], [4.3,-3.4], [4.2,2.9], [0.7,-0.6], [-2.5,2.5]])
Y = np.array([0,1,0,1,0,0])

net = Perseptron(len(X[0]))
net.fit(X,Y)
W,b = net.get_param()

##### PLOTS
colors = ['r' if l else 'b' for l in Y]
plt.scatter(X[:, 0], X[:, 1], c=colors)
x = np.array([-5., 5.])
y = np.array([(W[0]*i + b)/-W[1] for i in x])
plt.plot(x,y)
plt.xlim(-5, 5), plt.ylim(-5, 5)
plt.show()


##### TASK2
X = np.array([[-3.,4.], [1.7,-0.4], [4.1,-4], [2.4,2.3], [0.6,-3.2], [0.9,-2.1], [-3.7,-2.9], [3.9,-4.3]])
y1 = np.array([1,1,0,1,0,0,0,0])
y2 = np.array([0,1,1,1,1,1,0,1])

net = Net2(len(X[0]))
net.fit(X, y1, y2)
[W1, b1], [W2, b2] = net.get_param()


### PLOTS
colors = ['r' if (cl1 and cl2) else 'g' if (cl1 and not cl2) else 'y' if (not cl1 and cl2) else 'b' for cl1, cl2 in zip(y1, y2)]
plt.scatter(X[:, 0], X[:, 1], c=colors)
x = np.array([-5., 5.])
y = np.array([(W1[0]*i + b1)/-W1[1] for i in x])
plt.plot(x,y)
y = np.array([(W2[0]*i + b2)/-W2[1] for i in x])
plt.plot(x,y)
plt.xlim(-5, 5), plt.ylim(-5, 5)
plt.show()