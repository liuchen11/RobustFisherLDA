import util
from FisherLDA import mainFisherLDAtest
import json
import numpy as np
import matplotlib.pyplot as plt




if __name__ == "__main__":
    dataset = ['ionosphere', 'sonar']  # choose the dataset
    alphas = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

    fisherRightRates = np.empty(shape=(0, len(alphas)))
    for i in range(10):
        rightRates = []
        for alpha in alphas:
            rightRate = mainFisherLDAtest(dataset[0], alpha)
            rightRates.append(rightRate)
        rightRates = np.array(rightRates)
        rightRates = rightRates.reshape(1, len(alphas))
        fisherRightRates = np.concatenate((fisherRightRates, rightRates), axis=0)
        
    fisherRightRates_mean = np.mean(fisherRightRates, axis=0)
    fisherRightRates_std = np.std(fisherRightRates, axis=0)

    plt.fill_between(alphas, fisherRightRates_mean - fisherRightRates_std,
                     fisherRightRates_mean + fisherRightRates_std, alpha=0.1,
                     color="r")
    plt.fill_between(alphas, fisherRightRates_mean - fisherRightRates_std,
                     fisherRightRates_mean + fisherRightRates_std, alpha=0.1, color="g")
    plt.plot(alphas, fisherRightRates_mean, 'o-', color="r",
             label="Training Precision")
    plt.plot(alphas, fisherRightRates_mean, '*-', color="g",
             label="Cross-validation Precision")

    plt.legend(loc="best")

    plt.xlabel('Max Depth(log2(all features) to be considered) ')
    plt.ylabel('Precision')
    plt.title('Validation Curve with Decision Tree on the parameter of Max Depth')
    plt.grid(True)
    plt.show()

    # save the right rates into a json file
    #with open('data.txt', 'w') as outfile:
    #    json.dump(fisherLDA_rightRates, outfile)

