import numpy as np
# a = [(5,1), (10, 5), (13, 3), (14, 4)]
# l1, l2 = np.array(list(zip(*a)))
# print(l1 - l2)
# bits = np.log2(3)
# print(bits)
# a = 11
# print(('{0:0' + str(20) + 'b}').format(a))
#
# list3 = np.random.randint(0, 6, size=(5, 7, 4))
# list2 = np.random.randint(0, 2, size=(7, 4)).astype(np.bool)
# print(list3[0])
# print(list3[1])
# list3[0][list2] = list3[[1], list2]
# print(list3[0])
vector_a = np.array([1, 1])
vector_b = np.array([2, -1])
print(np.dot(vector_a, vector_b))
print((np.linalg.norm(vector_a) * np.linalg.norm(vector_b)))
print(np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b)))
