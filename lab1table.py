import numpy as np
from sklearn import datasets
from scipy.stats import norm

# loading iris data set
iris = datasets.load_iris()

setosa_indices = range(0, 50)
versicolor_indices = range(50, 100)

def table(feature_idx, feature_name, test_values):
    print(f"\n--- Classification Table for {feature_name} (x={feature_idx}) ---")
    
    # turn data into 1d
    X = iris.data[:, feature_idx]
    X0 = X[setosa_indices]     
    X1 = X[versicolor_indices] 
    
    # calculate mu and sigma
    mu0, std0 = np.mean(X0), np.std(X0)
    mu1, std1 = np.mean(X1), np.std(X1)
    
    
    p0 = 0.5
    p1 = 0.5
    
    
    print(f"{'Value':<8} | {'P(w0|x)':<12} | {'P(w1|x)':<12} | {'g(x)':<10} | {'Prediction'}")
    print("-" * 65)
    
    for x in test_values:
        # the distribution
        lik0 = norm.pdf(x, mu0, std0)
        lik1 = norm.pdf(x, mu1, std1)
        
        # normalization
        evidence = (lik0 * p0) + (lik1 * p1)
        
        #posteriors
        if evidence < 1e-9:
            #decisions which is more likely
            post0 = 1.0 if lik0 > lik1 else 0.0
            post1 = 1.0 if lik1 > lik0 else 0.0
        else:
            post0 = (lik0 * p0) / evidence
            post1 = (lik1 * p1) / evidence
            
        #the discrimninant
        #add err so log(0) isnt a problem
        err = 1e-15
        g_val = np.log((post0 + err) / (post1 + err))
        
        #decision
        pred = "setosa (w0)" if g_val > 0 else "versicolor (w1)"
        
        print(f"{x:<8} | {post0:.5f}      | {post1:.5f}      | {g_val:+.4f}    | {pred}")


test_values = [3.3, 4.4, 1.2, 5.0, 5.7, 6.3, 1.5]

#part 6: x0 sepal length
table(0, "Sepal Length", test_values)

#part 7 x1 sepal width
table(1, "Sepal Width", test_values)