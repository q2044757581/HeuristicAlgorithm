import matplotlib.pyplot as plt
import numpy as np
import random
import time
x = np.array([i/100 for i in range(500)])
y = np.sin(10*x)*x + np.cos(2*x)*x


class PSO:
    def __init__(self, n, scope, k):
        self.number = n
        self.scope = scope
        self.k = k
        self.c1 = 1
        self.c2 = 1
        gap = (scope[1] - scope[0]) / n
        # 初始化鸟群位置
        self.birds = []
        for i in range(n):
            self.birds.append(scope[0] + i * gap)
        # 初始最优解为开始的鸟群位置
        self.each_best_solution = self.birds.copy()
        self.each_best_score = []
        self.v = [0 for i in range(n)]
        self.globe_best_solution = scope[0] - 1
        self.globe_best_score = 0
        self.w = 0.5  # 惯性

    def get_height(self, x):
        return np.sin(10*x)*x + np.cos(2*x)*x

    def run(self):
        # 一开始先对整个鸟群的位置进行评估， 得到全局最优解
        for j in range(self.k):
            if j == 0:
                # 评估
                for i in range(self.number):
                    temp_score = self.get_height(self.birds[i])
                    self.each_best_score.append(temp_score)
                    if self.globe_best_solution == self.scope[0] - 1:
                        self.globe_best_solution = self.birds[i]
                        self.globe_best_score = temp_score
                    elif temp_score > self.globe_best_score:
                        self.globe_best_solution = self.birds[i]
                        self.globe_best_score = temp_score
            else:
                # 评估
                for i in range(self.number):
                    temp_score = self.get_height(self.birds[i])
                    if temp_score > self.each_best_score[i]:
                        self.each_best_score[i] = temp_score
                        self.each_best_solution[i] = self.birds[i]
                    if temp_score > self.globe_best_score:
                        self.globe_best_solution = self.birds[i]
                        self.globe_best_score = temp_score
            # 状态转移
            for i in range(self.number):
                self.v[i] = self.v[i] * self.w \
                                + self.c1 * random.random() * (self.each_best_solution[i] - self.birds[i]) + \
                                self.c2 * random.random() * (self.globe_best_solution - self.birds[i])
                self.birds[i] = self.birds[i] + self.v[i]
                if self.birds[i] > 5:
                    self.birds[i] = 5
                elif self.birds[i] < 0:
                    self.birds[i] = 0
            if j == self.k - 1:
                # plt.cla()
                # plt.plot(x, y)
                # for i in range(self.number):
                #     plt.scatter(self.birds[i], self.get_height(self.birds[i]))
                # plt.show()
                print(self.birds)
                # print(self.each_best_solution)
                # print(self.each_best_score)
                # print(self.globe_best_solution)
                print(self.globe_best_score)


t = PSO(10, [0, 5], 30)
time1 = time.time()
t.run()
print(time.time() - time1)







