from sklearn.neighbors import NearestNeighbors

arr_train = [[1,2], [-1,2,], [3,4], [100,200], [50, -50], [-10, -5], [5, 10], [1, 1], [5, 5], [4, 4]]
neigh = NearestNeighbors(n_neighbors=3)
neigh.fit(arr_train)
arr_distance, arr_index = neigh.kneighbors([[3, 3], [100, 100]])
print(arr_distance)
print(arr_index)
