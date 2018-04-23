def euclid_distance(pt1, pt2): return ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5


x, y = (1, 3), (2, 5)
euclid_dist = euclid_distance(x, y)
print(euclid_dist)
