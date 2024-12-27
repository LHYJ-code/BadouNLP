import numpy as np
import random
import sys


class KMeansClusterer:
    def __init__(self, ndarray, cluster_num):
        self.ndarray = ndarray
        self.cluster_num = cluster_num
        self.points = self.__pick_start_point(ndarray, cluster_num)

    def cluster(self):
        result = []
        for i in range(self.cluster_num):
            result.append([])
        for item in self.ndarray:
            distance_min = sys.maxsize
            index = -1
            for i in range(len(self.points)):
                distance = self.__distance(item, self.points[i])
                if distance < distance_min:
                    distance_min = distance
                    index = i
            result[index] = result[index] + [item.tolist()]
        new_center = []
        for item in result:
            if item:  # 防止传入空列表
                centroid = self.__center(np.array(item))
                new_center.append(centroid.tolist())
        # 中心点未改变，说明达到稳态，结束递归
        if np.array_equal(np.array(self.points), np.array(new_center)):
            sum_dis = []
            for i in range(len(self.points)):
                dis = 0
                for j in range(len(result[i])):
                    dis += self.__distance(result[i][j], self.points[i])
                sum_dis.append(dis / len(result[i]))
            sorted_index = np.argsort(sum_dis)
            sorted_result = []
            sorted_centers = []
            for ind in sorted_index:
                sorted_result.append(result[ind])
                sorted_centers.append(self.points[ind].tolist())
            total_sum = sum(sum_dis)
            return sorted_result, np.array(sorted_centers), total_sum
        self.points = np.array(new_center)
        return self.cluster()

    def __distance(self, p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))

    def __center(self, list):
        return np.mean(list, axis=0)

    def __pick_start_point(self, ndarray, cluster_num):
        if cluster_num < 0 or cluster_num > ndarray.shape[0]:
            raise Exception("簇数设置有误")
        indexes = random.sample(np.arange(0, ndarray.shape[0], step=1).tolist(), cluster_num)
        points = []
        for index in indexes:
            points.append(ndarray[index].tolist())
        return np.array(points)


x = np.random.rand(100, 8)
kmeans = KMeansClusterer(x, 10)
result, centers, distances = kmeans.cluster()
print(result)
print(centers)
print(distances)
