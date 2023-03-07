
from sklearn.neural_network import MLPClassifier


#All possible combinations of 2 bits
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

def train_MLP(X, y):
    
    activations = ['logistic', 'relu','identity','tanh']
    
    for activation in activations:
        model = MLPClassifier(activation=activation)
        model.fit(X, y)
        
        print("Activation: {}".format(activation))
        print(model.score(X, y))
        print(model.predict(X))

def main():
    
    data = get_data()
    y = XOR(data)
    
    train_MLP(data, y)
    
if __name__ == "__main__":
    
    main()