import numpy as np
np.random.seed(0)
data = np.genfromtxt("classification.txt",delimiter=",")
learningRate = 0.0001
data = np.delete(data,[4],axis=1)# deletinig the 5th column
Y = data[:,-1]
X_train = np.delete(data,[3],axis=1)
oneVector = np.ones((X_train.shape[0],1))
X_train = np.concatenate((oneVector,X_train),axis=1)
# xa = X_train[0].reshape(-1,X_train.shape[1])
weights = np.random.rand(4,1)
misClassifications=1
iteration = 0
while(misClassifications!=0):
    iteration+=1
    misClassifications=0
    for i in range(0,len(X_train)):
        currentX = X_train[i].reshape(-1,X_train.shape[1])
        currentY = Y[i]
        wTx = np.dot(currentX, weights)[0][0]
        if currentY==1 and wTx<0:
            misClassifications+=1
            weights = weights + learningRate * np.transpose(currentX)
        elif currentY==-1 and wTx>0:
            misClassifications+=1
            weights = weights - learningRate * np.transpose(currentX)

    # if iteration%1==0:
    print ("Iteration {}, Misclassifications {}".format(iteration,misClassifications))
print("\n")
print ("Weights")
print(weights.transpose())
print("Number of misclassifications: ",misClassifications)
