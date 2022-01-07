import matplotlib.pyplot as plt
import numpy as np
from DetectionTheory import gaussian_cdf

if __name__ == "__main__":

   # Plot example ROC curves illustrating good and bad detectors
    
    plt.close("all")
    
    gamma_range = np.linspace(-10,10,1000)
    sigma_range = [0.5, 0.9]
    labels = ["Good detector", "Better detector"]

    for i, sigma in enumerate(sigma_range):
        points = []
        
        for gamma in gamma_range:
            PD  = gaussian_cdf(gamma, mu = 0, sigma = sigma)
            PFA = gaussian_cdf(gamma, mu = 1, sigma = sigma)
            points.append([PD, PFA])
        
        points = np.array(points)
        auc = np.trapz(points[:, 0], points[:, 1])
        plt.plot(points[:, 1], points[:, 0], linewidth = 2, label = labels[i])

    plt.plot([0, 1], [0, 1], '-', linewidth = 2, label = "Bad detector")
    plt.legend(loc = 4)
    plt.xlabel('Probability of False Alarm $P_{FA}$')
    plt.ylabel('Probability of Detection $P_{D}$')

    #plt.show()
    #plt.title("ROC Curve")
    plt.savefig("../images/roc_difficulty.pdf", bbox_inches = "tight")
