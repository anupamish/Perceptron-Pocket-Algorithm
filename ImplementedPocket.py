import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)
data = np.genfromtxt("classification.txt", delimiter=",")
learningRate = 0.01
data = np.delete(data, [3], axis=1)  # deletinig the 4th column
Y = data[:, -1]
X_train = np.delete(data, [3], axis=1)
oneVector = np.ones((X_train.shape[0], 1))
X_train = np.concatenate((oneVector, X_train), axis=1)
# xa = X_train[0].reshape(-1,X_train.shape[1])
print (X_train)
print(Y)
plotData = []
weights = np.random.rand(4, 1)
misClassifications = 1
minMisclassifications = 10000
iteration = 0
while (misClassifications != 0 and (iteration<7000)):
    iteration += 1
    misClassifications = 0
    for i in range(0, len(X_train)):
        currentX = X_train[i].reshape(-1, X_train.shape[1])
        currentY = Y[i]
        wTx = np.dot(currentX, weights)[0][0]
        if currentY == 1 and wTx < 0:
            misClassifications += 1
            weights = weights + learningRate * np.transpose(currentX)
        elif currentY == -1 and wTx > 0:
            misClassifications += 1
            weights = weights - learningRate * np.transpose(currentX)
    plotData.append(misClassifications)
    if misClassifications<minMisclassifications:
        minMisclassifications = misClassifications
    # if iteration%1==0:
    print("Iteration {}, Misclassifications {}".format(iteration, misClassifications))
print ("Minimum Misclassifications : ",minMisclassifications)
print(weights.transpose())
print ("Best Case Accuracy of Pocket Learning Algorithm is: ",(((X_train.shape[0]-minMisclassifications)/X_train.shape[0])*100),"%")
# print("Number of misclassifications: ", misClassifications)
plt.plot(np.arange(0,7000),plotData)
plt.xlabel("Number of Iterations")
plt.ylabel("Number of Misclassifications")
plt.show()
