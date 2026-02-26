import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.stats import norm

#load iris data set
iris = datasets.load_iris()

Aseplen = iris.data[:50, 0]
Apetlen = iris.data[:50, 3]
setA = np.column_stack((Aseplen,Apetlen))

Bseplen = iris.data[50:100, 0]
Bpetlen = iris.data[50:100, 3]
setB = np.column_stack((Bseplen,Bpetlen))

Cseplen = iris.data[100:150, 0]
Cpetlen = iris.data[100:150, 3]
setC = np.column_stack((Cseplen,Cpetlen))

# test and train split
A70p = np.column_stack((Aseplen,Apetlen))[:35] # 70%
A30p = np.column_stack((Aseplen,Apetlen))[35:50] # 30%

B70p = np.column_stack((Bseplen,Bpetlen))[:35] # 70% 
B30p = np.column_stack((Bseplen,Bpetlen))[35:50] # 30% 

C70p = np.column_stack((Cseplen,Cpetlen))[:35] # 70% 
C30p = np.column_stack((Cseplen,Cpetlen))[35:50] # 30% 


def normalize(array):
    x = array.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j] = x[i,j]*(-1)
    return x




# task 1 and 2
trainAB = np.vstack((A30p, normalize(B30p))) # 30% train
testAB = np.vstack((A70p, B70p)) # 70% test

train3 = np.vstack((A70p, normalize(B70p)))  # 70% train
test3 = np.vstack((A30p, B30p)) # 30% test

trainBC = np.vstack((B30p, normalize(C30p))) # 30% train
testBC = np.vstack((B70p, C70p)) # 70% test

train4 = np.vstack((A30p, normalize(B30p)))  # 70% train
test4 = np.vstack((A70p, B70p)) # 30% test

tolerance = 0.0001
learningrate = 0.01 # n(k) = 0.1

def updateA(avector, ymiss):
    newa = avector.copy()
    for i in range(len(ymiss)):
        newa[i] = newa[i] + learningrate*ymiss[i]
    return newa


def learn(testingSet, a):
    newa = a.copy()
    k = 0
    norm = 0
    while(True):
        ymiss = [0, 0, 0]
        olda = newa.copy()
        for i in range(testingSet.shape[0]):
            if i < testingSet.shape[0]/2:
                norm = 1
            else:
                norm = -1
            if (newa[0]*norm + newa[1]*testingSet[i,0] + newa[2]*testingSet[i,1]) < 0:
                ymiss[0] += norm
                ymiss[1] += testingSet[i,0]
                ymiss[2] += testingSet[i,1]
           
        newa = updateA(newa, ymiss=ymiss)

        k += 1
        if (abs(olda[0]-newa[0])+abs(olda[1]-newa[1])+abs(olda[2]-newa[2])) < tolerance:
            break
        if k > 300:
            break
    print(f"number of iterations {k}")
    return newa            

def plot(testingSet, a, task):

    half = int(testingSet.shape[0] / 2)
    class1_x = testingSet[:half, 0]
    class1_y = testingSet[:half, 1] 
    class2_x = testingSet[half:, 0]
    class2_y = testingSet[half:, 1]

    plt.figure(figsize=(8, 6))
    plt.scatter(class1_x, class1_y, color='blue', label='Class 1', edgecolors='k')
    plt.scatter(class2_x, class2_y, color='red', label='Class 2', edgecolors='k')

    x = np.array([0, np.max(testingSet[:, 0]) + 0.5])

    line_y = -(a[0] + a[1] * x) / a[2]

    plt.plot(x, line_y, color='green', linewidth=2, label='Decision Boundary')

    plt.title("Perceptron Decision Boundary on Test Data")
    plt.xlabel("Sepal Length (Feature 1)")
    plt.ylabel("Petal Width (Feature 2)")
    plt.legend()
    plt.grid(True)
    
    plt.show()

def test(testingSet, a):
    w = a.copy()
    real = 0
    test = 0
    misfire = 0
    for i in range(testingSet.shape[0]):
        if (w[0] + w[1]*testingSet[i,0] + w[2]*testingSet[i,1]) > 0:
            #print("test says: class 1")
            test = 0
        else:
            #print("class says: class 2")
            test=1

        if i < testingSet.shape[0]/2:
            #print("real is: class 1")
            real=0
        else: 
            #print("real is: class 2")
            real=1

        if test != real:
            misfire += 1
    accuracy = (1-(misfire/testingSet.shape[0])) *100
    print(f"total misfires = {misfire} \naccuracy = {accuracy}")

    #plot it
    plot(testingSet=testingSet, a=a)


a = [1, 1, 1] # randomly initialize a
print(f"With value of a = [{a[0]}, {a[1]}, {a[2]}]")

print("Task 2: Data Set AB - Testing with 70%")
test(testAB,a=learn(trainAB, a=a))

print("\nTask 3: Data Set AB - Testing with 30%")
test(test3,a=learn(train3, a=a))

print("\nTask 4.1: Data Set BC - Testing with 70%")
test(testBC,a=learn(trainBC, a=a))

print("\nTask 4.2: Data Set BC - Testing with 30%")
test(test4,a=learn(train4, a=a))


a = [2, 3, 5] # randomly initialize a
print(f"\nWith value of a = [{a[0]}, {a[1]}, {a[2]}]")

print("Task 2: Data Set AB - Testing with 70%")
test(testAB,a=learn(trainAB, a=a))

print("\nTask 3: Data Set AB - Testing with 30%")
test(test3,a=learn(train3, a=a))

print("\nTask 4.1: Data Set BC - Testing with 70%")
test(testBC,a=learn(trainBC, a=a))

print("\nTask 4.2: Data Set BC - Testing with 30%")
test(test4,a=learn(train4, a=a))


a = [0, 34, 2] # randomly initialize a
print(f"\nWith value of a = [{a[0]}, {a[1]}, {a[2]}]")

print("Task 2: Data Set AB - Testing with 70%")
test(testAB,a=learn(trainAB, a=a))

print("\nTask 3: Data Set AB - Testing with 30%")
test(test3,a=learn(train3, a=a))

print("\nTask 4.1: Data Set BC - Testing with 70%")
test(testBC,a=learn(trainBC, a=a))

print("\nTask 4.2: Data Set BC - Testing with 30%")
test(test4,a=learn(train4, a=a))


