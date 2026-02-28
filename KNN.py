from collections import Counter

class KNN:

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    # in KNN's there is explicitly no training, only prediction which is O(n)

    def distance(self, a, b):
      sum = 0
      for i in range(len(a)):
        sum += (a[i] - b[i])**2
        return sum**0.5

    def predict(self, x_test, K=3):

        distances = []

        for i in range(len(self.X_train)):
            d = self.distance(x_test, self.X_train[i])
            distances.append((d, self.y_train[i]))

        distances.sort(key=lambda x: x[0])
        neighbors = distances[:K]

        labels = [label for _, label in neighbors]
        return Counter(labels).most_common(1)[0][0]
