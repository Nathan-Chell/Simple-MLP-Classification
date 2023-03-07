from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier

from random import randint

def get_data():
    #Get the iris dataset
    
    data = datasets.load_iris()
    x = data.data
    y = data.target
    
    return x, y

def Split_data(X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=randint(0, 1000))
    
    return X_train, X_test, y_train, y_test
    
def train_MLP(X_train, y_train, X_test, y_test, layers=(100,)):
    
    
    model = MLPClassifier(hidden_layer_sizes=layers, max_iter=2000, solver='adam')
    model.fit(X_train, y_train)
    
    #print("Hidden layer size: {}".format(layers))
    #print("Test accuracy: {}".format(model.score(X_test, y_test)))
    #print("Confusion matrix: {}".format(confusion_matrix(y_test, model.predict(X_test))))
    
    return model.score(X_test, y_test)

def main():
    
    X, y = get_data()
    X_train, X_test, y_train, y_test = Split_data(X, y)
    
    iter = 3
    total_accuracy = 0
    layers = (10,)
    
    for i in range(iter):
        total_accuracy += train_MLP(X_train, y_train, X_test, y_test, layers)
        
    print("Average accuracy: {} for layer size: {}".format(total_accuracy/iter, layers))
    
if __name__ == "__main__":
    
    main()