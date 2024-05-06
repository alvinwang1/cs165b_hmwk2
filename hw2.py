import time
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score



class Binary_Classifier(object):

    def __init__(self, train_data, train_target):
        """
        Data is pre-loaded and passed to __init__ directly. train_data is the train feature matrix
        and train_target is the train label vector. You can store both
        training features and target vector as instance variables
        """
        self.x = train_data.values
        self.y = train_target.values
        self.weights_loss = None
        self.weights_accuracy = None
        self.best_error = 100000
        self.best_accuracy = 0
        self.svm_classifier = None

    def logistic_training(self, alpha, lam, nepoch, epsilon):
        """
        Training process of logistic regression. 
        alpha: learning rate
        lam: regularization strength
        nepoch: number of epochs to train.
        eposilon: early stop condition.
        
        The use of these parameters is the same as that in program #1.
        You implementation must include 5-fold cross validation,
        Hint: You can store the weight as an instance variable,
        so other functions can directly use them
        """
        self.x, self.y = self.shuffle_data(self.x, self.y)
        k = 5
        fold_size = len(self.x) // k
        for i in range(k):
            start = i * fold_size
            end = (i+1) * fold_size
            test_X = self.x[start:end]
            test_Y = self.y[start:end]
            test_X = np.concatenate((test_X, np.ones((test_X.shape[0], 1))), axis=1)
            train_X = np.concatenate([self.x[:start], self.x[end:]], axis=0)
            train_Y = np.concatenate([self.y[:start], self.y[end:]], axis=0)

            best_weights = self.SGD_Solve(alpha, lam, nepoch, epsilon, train_X, train_Y)
            prediction = test_X @ best_weights
            error = test_Y - prediction
            loss = np.mean(error**2)       
            loss = loss / prediction.shape[0]
            if(self.best_error > loss):
                #print(loss)
                self.best_error = loss
                self.weights_loss = best_weights
            
            total_correct = 0
            for j in range(prediction.shape[0]):
                predict = None
                if(prediction[j] >= 0.5):
                    predict = 1
                else:
                    predict = 0
                if(predict == test_Y[j]):
                    total_correct += 1
            if(total_correct > self.best_accuracy):
                #print(total_correct)
                self.best_accuracy = total_correct
                self.weights_accuracy = best_weights
            

    def SGD_Solve(self, alpha, lam, nepoch, epsilon, train_X, train_Y):
        best_weights = None
        best_error = 100000
        alpha_array = np.geomspace(alpha[0], alpha[1], num=5)
        lam_array = np.geomspace(lam[0], lam[1], num=5)
        x_augmented = np.concatenate((train_X, np.ones((train_X.shape[0], 1))), axis=1)
        for a in alpha_array:
            for l in lam_array:
                w = np.random.randn(x_augmented.shape[1])
                loss = np.inf
                break_counter = 0
                for i in range(nepoch):
                    x_augmented, train_Y = self.shuffle_data(x_augmented, train_Y)
                    prediction = x_augmented @ w
                    error = train_Y - prediction
                    prev = loss
                    loss = np.mean(error**2)
                    if loss > prev:
                        break_counter += 1
                    else: 
                        break_counter = 0

                    if(break_counter > 20):
                        break
                    
                    if loss < epsilon:
                        #print(f"Training stopped at epoch {i+1} because error < epsilon.")
                        break

                    total_error = 0.0
                    for j in range(x_augmented.shape[0]):
                        temp_data = x_augmented[j]
                        temp_ground_truth = train_Y[j]
                        prediction = temp_data @ w
                        error = prediction - temp_ground_truth
                        total_error += error ** 2
                        w_gradient = (temp_data.T * error) + l * w
                        """
                        grad_norm = np.linalg.norm(w_gradient, ord=1)
                        clip_threshold = 100  # Adjust as needed
                        if grad_norm > clip_threshold:
                            w_gradient = w_gradient * clip_threshold / grad_norm
                        """
                        w -= a * (w_gradient)                    
                    loss = total_error / x_augmented.shape[0]

                    
                    print(f"Epoch {i+1}/{nepoch}, MSE: {loss}, alpha: {a}, lambda: {l}")

                    if(total_error < best_error):
                        #print(total_error, self.best_error)
                        best_error = total_error
                        best_weights = w
                #print(f"weights: {self.weights}")
                #print(f"bias:{self.bias}")
        return best_weights
    
    def shuffle_data(self, x, y):
        """Function to shuffle data"""
        indices = np.arange(len(x))
        np.random.shuffle(indices)
        return x[indices], y[indices]

    def logistic_testing(self, testX):
        test_X = np.concatenate((testX, np.ones((testX.shape[0], 1))), axis=1)
        prediction = test_X @ self.weights_loss
        result = (prediction >= 0.5).astype(int)
        return np.array([result]).T

    def svm_training(self, gamma, C):
        """
        Training process of the support vector machine. 
        gamma, C: grid search parameters

        As softmargin SVM can handle nonlinear boundaries and outliers much better than logistic regression,
        we do not perform 3-fold validation here (just one training run with 90-10 training-validation split).
        Furthmore, you are allowed to use SVM's built-in grid search method.
        
        This function will be a "wrapper" around sklearn.svm.SVC with all other parameters take the default values.
        Please consult sklearn.svm.SVC documents to see how to use its "fit" and "predict" functions. 
        https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        """
        self.svm_classifier = SVC(kernel='linear', C = 0.1, gamma = 'scale')
        self.svm_classifier.fit(self.x, self.y)

    def svm_testing(self, testX):
        y_pred = self.svm_classifier.predict(testX)
        return np.array([y_pred]).T


# Dataset preparation: Dataset is divided into 90% and 10%
# 90% for you to perform n-fold cross validation and 10% for autograder to validate your performance.
################## PLEASE DO NOT MODIFY ANYTHING! ##################
dataset = load_breast_cancer(as_frame=True)
train_data = dataset['data'].sample(frac=0.9, random_state=0) # random state is a seed value
train_target = dataset['target'].sample(frac=0.9, random_state=0) # random state is a seed value
test_data = dataset['data'].drop(train_data.index)
test_target = dataset['target'].drop(train_target.index)


# Model training: You are allowed to change the last two inputs for model.logistic_training
################## PLEASE DO NOT MODIFY ANYTHING ELSE! ##################
model = Binary_Classifier(train_data, train_target)

# Logistic Regression
logistic_start = time.time()
model.logistic_training([1e-8, 1e-7], [1500,5000], 500, 1e-5)
logistic_end = time.time()
# SVM
svm_start = time.time()
model.svm_training([1e-9, 1000], [0.01, 1e10])
svm_end = time.time()

#print(model.svm_testing(test_data))
#print(model.logistic_testing(test_data))