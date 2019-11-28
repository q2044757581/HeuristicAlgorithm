# import numpy as np
# from copy import deepcopy
# capacity = 10
# w = [2, 3, 4, 5]  # 体积
# v = [2, 3, 6, 3]  # 价值
# f = np.zeros([len(w), capacity+1])
# plan = []
# temp_plan = []
# for j in range(capacity+1):
#     if j >= w[0]:
#         f[0][j] = v[0]
#         temp_plan.append([0])
#     else:
#         temp_plan.append([])
# plan.append(deepcopy(temp_plan))
# for i in range(1, len(w)):
#     temp_plan = []
#     for j in range(capacity + 1):
#         if j >= w[i]:
#             if f[i - 1][j] > f[i - 1][j - w[i]] + v[i]:
#                 f[i][j] = f[i - 1][j]
#                 temp_plan.append(deepcopy(plan[i-1][j]))
#             else:
#                 f[i][j] = f[i - 1][j - w[i]] + v[i]
#                 temp_plan.append(deepcopy(plan[i - 1][j - w[i]]) + [i])
#         else:
#             f[i][j] = f[i - 1][j]
#             temp_plan.append(deepcopy(plan[i - 1][j]))
#     plan.append(temp_plan)
# print("最大价值：", f[len(w) - 1][capacity])
# print("最佳方案：", plan[len(w) - 1][capacity])
import random
for i in range(10):
    print(random.randint(0, 10))