import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

#i must to say this algorithm is bad. failed to classify some cumber data
#actually ,we should use the vote MeanShift algorithm.the following is truely bad


X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8 ],
              [8, 8],
              [1, 0.6],
              [9,11],
              [1,3],
              [8,9],
              [0,3],
              [5,4],
              [6,4],
                ])

##plt.scatter(X[:,0], X[:,1], s=150)
##plt.show()

colors = 10*["g","r","c","b","k"]

class Mean_Shift:
    def __init__(self,radius =4):
        self.radius = radius
    def fit(self,data):
        centroids = {}

        for i in range(len(data)): #initial
            centroids[i] = data[i]

        while True:
            new_centroids = []
            classification = {}
            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]
                classification[i] = []
                for featureset in data:
                    if np.linalg.norm(featureset- centroid)< self.radius:
                        in_bandwidth.append(featureset)
                        classification[i].append(featureset)
                new_centroid = np.average(in_bandwidth,axis=0)
                new_centroids.append(tuple(new_centroid))

            uniques = sorted(list(set(new_centroids)))

            prev_centroids = dict(centroids)

            centroids = {}

            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])

            optimized = True

            for i in centroids:
                #when the centroids not changed ,the algrothim is done
                #if else,do the optimize again
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False
                if not optimized:
                    break

            if optimized:
                break
        self.centroids = centroids
        self.classification = classification

    def predict(self,data):
        pass

clf  = Mean_Shift()
clf.fit(X)

centroids = clf.centroids
classification = clf.classification
for i in classification:
    for featureset in classification[i]:
        plt.scatter(featureset[0],featureset[1],s=150,color = colors[i])

for centroid in centroids:
    plt.scatter(centroids[centroid][0],centroids[centroid][1],color = 'k',marker = 'x',s =150)

plt.show()


                








                
            
