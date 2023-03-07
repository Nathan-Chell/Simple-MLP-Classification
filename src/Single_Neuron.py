

from sklearn.linear_model import Perceptron

#Input data, all possible combinations of 2 bits
def get_data():
    
    return [[0,0],[0,1],[1,0],[1,1]]

#XOR function is not linearly separable
#Thus we cannot use a single perceptron to approximate it
#We'd have to use multiple perceptrons
def XOR(data):
    
    return list(map(lambda x: 1 if x[0] != x[1] else 0, data))

#AND function is linearly separable
#Thus we can use a single perceptron to approximate it
def AND(data):
    
    return list(map(lambda x: 1 if x[0] == 1 and x[1] == 1 else 0, data))

#Again NAND is also linearly separable
def NAND(data):
    
    return list(map(lambda x: 1 if x[0] == 0 and x[1] == 0 else 0, data))


def train_neuron(X, y):
    
    model = Perceptron()
    model.fit(X, y)
    
    print(model.score(X, y))
    print(model.predict(X))

def main():
    
    X = get_data()
    #y = XOR(X)
    #y = AND(X)
    y = NAND(X)
    
    train_neuron(X, y)
    

if __name__ == "__main__":
    
    main()
    