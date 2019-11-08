"""
我们希望得到初始的数据nums，每个变量的边界bound， 运算需要的函数func，DNA的大小（或碱基对的个数）DNA_SIZE,
染色体交叉的概率cross_rate，基因变异的概率mutation
其中得到的nums为二维N * M列表，N为进化种群的总数，M为DNA个数（变量的多少）。
bound为M * 2的二维列表，形如：[(0, 5), (-11.2, +134), ...]。
func为方法(function)对象，既可以用def定义出的，也可以是lambda表达式定义的匿名函数。
DNA_SIZE可以指定大小，为None时会自动指派。
cross_rate和mutation也有各自的默认值
"""
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import numpy as np

class GA:
    def __init__(self, nums, bound, func, DNA_SIZE=None, cross_rate=0.8, mutation=0.003):
        """
        :param nums: 初始数据
        :param bound: 每个变量的值域边界
        :param func: 目标函数
        :param DNA_SIZE: 基因长度
        :param cross_rate: 染色体交叉的概率
        :param mutation: 基因变异的概率
        编码： encoded_num = (num - var_min) / (vax_max - var_min) * (2 ^ DNA_SIZE)
        解码： decoded_num = (num / 2 ^ DNA_SIZE) * (var_max - var_min) + var_min
        """
        #     nums: m * n  n is nums_of x, y, z, ...,and m is population's quantity
        #     bound:n * 2  [(min, nax), (min, max), (min, max),...]
        #     DNA_SIZE is binary bit size, None is auto
        nums = np.array(nums)
        bound = np.array(bound)
        self.bound = bound
        # 判断初始数据的维度和bound的维度相不相符
        if nums.shape[1] != bound.shape[0]:
            raise Exception(f'范围的数量与变量的数量不一致, 您有{nums.shape[1]}个变量，却有{bound.shape[0]}个范围')
        # 判断每个基因的值在不在给定的值域内
        for var in nums:
            for index, var_curr in enumerate(var):
                if var_curr < bound[index][0] or var_curr > bound[index][1]:
                    raise Exception(f'{var_curr}不在取值范围内')
        # 判断取值范围是否合理
        for min_bound, max_bound in bound:
            if max_bound < min_bound:
                raise Exception(f'抱歉，({min_bound}, {max_bound})不是合格的取值范围')

        # 所有变量的最小值和最大值
        # var_len为所有变量的取值范围大小
        # bit为每个变量按整数编码最小的二进制位数
        min_nums, max_nums = np.array(list(zip(*bound)))
        self.var_len = var_len = max_nums - min_nums
        # 先取log2 在 取上限证书
        bits = np.ceil(np.log2(var_len + 1))
        # 把最大的bit值作为DNA_SIZE
        if DNA_SIZE == None:
            DNA_SIZE = int(np.max(bits))
        # 如果有定义则取定义值
        self.DNA_SIZE = DNA_SIZE

        # POP_SIZE为进化的种群数
        self.POP_SIZE = len(nums)
        POP = np.zeros((*nums.shape, DNA_SIZE))
        for i in range(nums.shape[0]):
            for j in range(nums.shape[1]):
                # 编码方式：
                # encoded_num = (num - var_min) / (vax_max - var_min) * (2 ^ DNA_SIZE)
                num = int(round((nums[i, j] - bound[j][0]) * ((2 ** DNA_SIZE) / var_len[j])))
                # 用python自带的格式化转化为前面空0的二进制字符串，然后拆分成列表
                POP[i, j] = [int(k) for k in ('{0:0' + str(DNA_SIZE) + 'b}').format(num)]
        self.POP = POP
        # 用于后面重置（reset）
        self.copy_POP = POP.copy()
        self.cross_rate = cross_rate
        self.mutation = mutation
        self.func = func

    # 将编码后的DNA翻译回来（解码）
    def translateDNA(self):
        W_vector = np.array([2 ** i for i in range(self.DNA_SIZE)]).reshape((self.DNA_SIZE, 1))[::-1]
        binary_vector = self.POP.dot(W_vector).reshape(self.POP.shape[0:2])
        for i in range(binary_vector.shape[0]):
            for j in range(binary_vector.shape[1]):
                binary_vector[i, j] /= ((2 ** self.DNA_SIZE) / self.var_len[j])
                binary_vector[i, j] += self.bound[j][0]
        return binary_vector

    # 得到适应度
    def get_fitness(self, non_negative=False):
        result = self.func(*np.array(list(zip(*self.translateDNA()))))
        # 如果要求非负， 则全部都减去最小值
        if non_negative:
            min_fit = np.min(result, axis=0)
            result -= min_fit
        return result

    # 自然选择
    def select(self):
        fitness = self.get_fitness(non_negative=True)
        # 轮盘赌选择
        self.POP = self.POP[np.random.choice(np.arange(self.POP.shape[0]), size=self.POP.shape[0], replace=True,
                                             p=fitness / np.sum(fitness))]

    # 染色体交叉
    def crossover(self):
        for people in self.POP:
            if np.random.rand() < self.cross_rate:
                i_ = np.random.randint(0, self.POP.shape[0], size=1)
                # 2维数组， 第一维是基因长度， 第二维是每个基因的编码长度， 值为0，1
                cross_points = np.random.randint(0, 2, size=(len(self.var_len), self.DNA_SIZE)).astype(np.bool)
                # 把选中的个体的基因值替换给当前的个体
                people[cross_points] = self.POP[i_, cross_points]

    # 基因变异
    def mutate(self):
        for people in self.POP:
            for var in people:
                for point in range(self.DNA_SIZE):
                    if np.random.rand() < self.mutation:
                        var[point] = 1 if var[point] == 0 else 1

    # 进化
    def evolution(self):
        self.select()
        self.crossover()
        self.mutate()

# 重置
    def reset(self):
        self.POP = self.copy_POP.copy()

# 打印当前状态日志
    def log(self):
        pass

# 一维变量作图
    def plot_in_jupyter_1d(self, iter_time=50):
        for _ in range(iter_time):
            self.evolution()
            if _ == iter_time - 1:
                plt.cla()
                x = np.linspace(*self.bound[0], self.var_len[0] * 50)
                plt.plot(x, self.func(x))
                x = self.translateDNA().reshape(self.POP_SIZE)
                plt.scatter(x, self.func(x), s=200, lw=0, c='red', alpha=0.5)
                plt.show()


func = lambda x:np.sin(10*x)*x + np.cos(2*x)*x
ga = GA([[np.random.rand()*5] for _ in range(100)], [(0, 5)], DNA_SIZE=11, func=func)
ga.plot_in_jupyter_1d()