import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.stats import norm

class GaussianClassifier1D:
    def __init__(self):
        self.mu0, self.std0 = None, None
        self.mu1, self.std1 = None, None
        self.p0, self.p1 = None, None
    
    def fit(self, X0, X1):
        """Designs the classifier by estimating parameters."""
        self.mu0, self.std0 = np.mean(X0), np.std(X0)
        self.mu1, self.std1 = np.mean(X1), np.std(X1)
        
        N0, N1 = len(X0), len(X1)
        self.p0 = N0 / (N0 + N1)
        self.p1 = N1 / (N0 + N1)

    def get_posteriors_and_g(self, x):
        lik0 = norm.pdf(x, self.mu0, self.std0)
        lik1 = norm.pdf(x, self.mu1, self.std1)
        
        px = (lik0 * self.p0) + (lik1 * self.p1)
        
        post0 = (lik0 * self.p0) / px
        post1 = (lik1 * self.p1) / px
        
        discr = np.log((post0 + 1e-10) / (post1 + 1e-10))
        
        return post0, post1, discr

    def find_threshold(self, risk_ratio=1.0):
       
        var0, var1 = self.std0**2, self.std1**2
        
        a = 1/(2*var0) - 1/(2*var1)
        b = self.mu1/var1 - self.mu0/var0
        c = self.mu0**2/(2*var0) - self.mu1**2/(2*var1) - np.log(self.p1/self.p0 * risk_ratio * (self.std0/self.std1))
        
        roots = np.roots([a, b, c])

        for r in roots:
            if min(self.mu0, self.mu1) <= r <= max(self.mu0, self.mu1):
                return r
        return roots[0]

#load iris data set
iris = datasets.load_iris()

test_values = [3.3, 4.4, 1.2, 5.0, 5.7, 6.3, 1.5]

def analyze_feature(feature_index, feature_name):
   
    print(f"Analysis: {feature_name}")
    
    setosa_data = iris.data[:50, feature_index]
    versicolor_data = iris.data[50:100, feature_index]
    
    clf = GaussianClassifier1D()
    clf.fit(setosa_data, versicolor_data)
    
    u1 = clf.find_threshold(risk_ratio=1.0)
    print(f"Optimal Threshold u1 (Min Error): {u1:.4f}")
    
    penalties = [1, 2, 5, 10]
    u2_vals = []
    print("\nThresholds u2 with different penalties (Cost_Misclassifying_w1):")
    for p in penalties:
        t = clf.find_threshold(risk_ratio=p)
        u2_vals.append(t)
        print(f"  Penalty {p}x: {t:.4f}")

    #graph    
    X = iris.data[:, feature_index]
    x_axis = np.linspace(X.min()-1, X.max()+1, 1000)
    plt.figure(figsize=(10, 4))
    plt.plot(x_axis, norm.pdf(x_axis, clf.mu0, clf.std0)*clf.p0, 'b', label='Class 0 (Setosa)')
    plt.plot(x_axis, norm.pdf(x_axis, clf.mu1, clf.std1)*clf.p1, 'r', label='Class 1 (Versicolor)')
    plt.axvline(u1, color='k', linestyle='--', label=f'u1 (Min Error): {u1:.2f}')
    plt.axvline(u2_vals[-1], color='g', linestyle=':', label=f'u2 (10x Penalty): {u2_vals[-1]:.2f}')
    plt.title(f'Distributions and Thresholds for {feature_name}')
    plt.legend()
    plt.show()

    #table print
    print(f"\nClassification Table for {feature_name}:")
    print(f"{'Value (x)':<10} | {'P(w0|x)':<12} | {'P(w1|x)':<12} | {'g(x)':<10} | {'decision'}")
    print("-" * 65)
    
    for val in test_values:
        p0, p1, g = clf.get_posteriors_and_g(val)
        decision = "w0-Setosa" if g > 0 else "w1-Versicolor"
        print(f"{val:<10} | {p0:.5f}      | {p1:.5f}      | {g:+.4f}    | {decision}")

analyze_feature(0, "Sepal Length")

analyze_feature(1, "Sepal Width")
#sepal width overlap makes it perform worse than length