import cupy as cp
import numpy as np
from scipy.special import gamma
from cupyx.scipy.ndimage import gaussian_filter
from dataclasses import dataclass
from typing import Any, List


class NewModel:
    find_indx = {1: 'left', 2: 'right', 3: 'up', 4: 'down'}  # 用于求fit_direction时的方向匹配字典

    def __init__(self,
                 image_height=450, image_width=450,
                 gaussian_kernel_sigma=1.0,
                 alpha_lamina=1, tau_lamina=3,
                 emd_tau=1, emd_order=10, emd_x0=2,
                 stmd_tau=[1, 3, 5, 7, 9], stmd_order=10,
                 eta_on=-1.0, eta_off=-1.0,
                 alpha_feedback=4, wide_feedback=10,
                 alpha_restrain_ON_OFF=4, wide_restrain_ON_OFF=10,
                 lamda_DT=0.00015  # 0.0001
                 ):

        # Retina层高斯核的sigma参数
        self.sigma = gaussian_kernel_sigma

        # Lamina层循环队列与权重核
        self.tau_lamina = tau_lamina
        self.list_lamina_len = int(tau_lamina * 4)
        self.list_lamina = CircularList(self.list_lamina_len)
        self.kernel_lamina = self.create_lamina_kernel(alpha=alpha_lamina, wide=int(self.tau_lamina * 3))  # 使用2*tau长度平滑

        # tau参数初始化(tau1: EMD延迟tau; tau2: 目标匹配tau)
        self.emd_tau = emd_tau
        self.stmd_tau = stmd_tau

        # EMD方向匹配的参数x0
        self.x0 = emd_x0

        # feedback抑制参数
        self.eta_on = eta_on
        self.eta_off = eta_off

        # 求fit_tau与fit_direction的截断阈值
        self.lamda_DT = lamda_DT

        # Feedback中的时间平滑
        self.C_LO_kernel = self.create_lamina_kernel(alpha=alpha_feedback, wide=wide_feedback)
        self.C_LO_list = {index: CircularList(wide_feedback) for index in range(4)}

        # EMD匹配的gamma核与循环列表数据
        self.gamma_kernel_emd = self.create_gamma_kernel(order=emd_order, tau=emd_tau, wide=int(emd_tau + 3))
        self.l1_list = CircularList(int(emd_tau + 3))
        self.l2_list = CircularList(max(emd_tau + 3, max(stmd_tau) + 3))  # 负部用于EMD和ESTMD匹配,所以tau参数取其中最大值

        # 目标匹配的gamma核
        self.gamma_kernel_stmd = {}
        for tau in stmd_tau:
            self.gamma_kernel_stmd[tau] = self.create_gamma_kernel(order=stmd_order, tau=tau, wide=int(tau + 3))

        # 目标匹配中ON和OFF抑制后的平滑
        self.restrain_ON = {tau: CircularList(wide_restrain_ON_OFF) for tau in stmd_tau}
        self.restrain_OFF = {tau: CircularList(wide_restrain_ON_OFF) for tau in stmd_tau}
        self.restrain_kernel = self.create_lamina_kernel(alpha=alpha_restrain_ON_OFF, wide=wide_restrain_ON_OFF)

        # 计算背景光流的卷积核
        self.W_lo2Lptc = self.create_kernel_LO_LPTC(img_height=image_height, img_width=image_width)
        self.W_lptc2stmd = self.create_kernel_LPTC_STMD(img_height=image_height, img_width=image_width)

        # 目标方向匹配的T4数据与目标匹配的feedback数据
        self.t4_list = {direction: CircularList(int(max(stmd_tau) + 3)) for direction in
                        ('left', 'right', 'up', 'down')}  # 列表长度应与最大tau相一致
        self.feedback_list = CircularList(int(max(stmd_tau) + 3))

    @staticmethod
    def create_lamina_kernel(alpha=3.0, wide=None):
        """这里wide取tau的3倍，也就是向前回溯的长度
        """

        if wide is None: raise Exception("Invalid wide!", wide)

        # Compute the values of the T vector
        timeList = np.arange(wide)
        TKernel = 1 / np.exp(timeList / alpha)

        # Normalize the T vector
        TKernel /= np.sum(TKernel)
        TKernel[TKernel < 1e-4] = 0
        TKernel /= np.sum(TKernel)

        return TKernel

    @staticmethod
    def create_gamma_kernel(order=100, tau=25, wide=None):
        """
        Generates a discretized Gamma vector.

        Parameters:
        - order: The order of the Gamma function.
        - tau: The time constant of the Gamma function.
        - wide: The length of the vector.

        Returns:
        - gammaKernel: The generated Gamma vector.
        """
        if wide is None: wide = int(np.ceil(3 * tau))

        # Ensure wide is at least 2
        if wide <= 1: wide = 2

        # Compute the values of the Gamma vector
        timeList = np.arange(wide)
        gammaKernel = (
                (order * timeList / tau) ** order *
                np.exp(-order * timeList / tau) /
                (gamma(order) * tau)
        )

        # Normalize the Gamma vector
        gammaKernel /= np.sum(gammaKernel)
        gammaKernel[gammaKernel < 1e-4] = 0
        gammaKernel /= np.sum(gammaKernel)

        return gammaKernel

    @staticmethod
    def create_kernel_LPTC_STMD(img_height: int, img_width: int) -> dict:
        """ 生成速度向量核

        Args:
            img_height: 图像矩阵的行数
            img_width: 图像矩阵的列数

        Returns:
           返回 W_lptc2stmd 核
        """

        W_lptc2stmd = {direction: [None] * 4 for direction in
                       ['forward', 'backward', 'left', 'right', 'up', 'down']}

        # 图像中心点
        i0 = cp.array(img_height // 2)
        j0 = cp.array(img_width // 2)

        # matrixX: 行的下标矩阵, matrixY: 列的下标矩阵
        matrixY, matrixX = cp.meshgrid(cp.arange(img_width), cp.arange(img_height))
        # 提前计算sqrt
        matrixSqrt = cp.sqrt(((matrixX - i0) ** 2 + (matrixY - j0) ** 2))
        matrixSqrt[i0, j0] = 1  # 把分母为0的情况避免掉

        # forward
        W_lptc2stmd['forward'][0] = (matrixX - i0) / matrixSqrt
        W_lptc2stmd['forward'][1] = -W_lptc2stmd['forward'][0]
        W_lptc2stmd['forward'][2] = (matrixY - j0) / matrixSqrt
        W_lptc2stmd['forward'][3] = -W_lptc2stmd['forward'][2]

        # backward
        W_lptc2stmd['backward'][0] = (i0 - matrixX) / matrixSqrt
        W_lptc2stmd['backward'][1] = -W_lptc2stmd['backward'][0]
        W_lptc2stmd['backward'][2] = (j0 - matrixY) / matrixSqrt
        W_lptc2stmd['backward'][3] = -W_lptc2stmd['backward'][2]

        W_lptc2stmd['left'] = cp.array([0, 0, -1, 1])
        W_lptc2stmd['right'] = cp.array([0, 0, 1, -1])
        W_lptc2stmd['up'] = cp.array([-1, 1, 0, 0])
        W_lptc2stmd['down'] = cp.array([1, -1, 0, 0])

        return W_lptc2stmd

    @staticmethod
    def create_kernel_LO_LPTC(img_height: int, img_width: int) -> dict:
        """ 生成自运动核

        Args:
            img_height: 图像矩阵的行数
            img_width: 图像矩阵的列数

        Returns:
           返回 W_lo2Lptc核
        """

        W_lo2Lptc = {direction: [None] * 4 for direction in ['forward', 'backward', 'left', 'right', 'up', 'down']}

        # 图像中心点
        i0 = cp.array(img_height // 2)
        j0 = cp.array(img_width // 2)

        # matrixX: 行的下标矩阵, matrixY: 列的下标矩阵
        matrixY, matrixX = cp.meshgrid(cp.arange(img_width), cp.arange(img_height))
        # 计算sqrt
        matrixSqrt = cp.sqrt(((matrixX - i0) ** 2 + (matrixY - j0) ** 2))
        matrixSqrt[i0, j0] = 1  # 把分母为0的情况避免掉

        # forward
        W_lo2Lptc['forward'][0] = (i0 - matrixX) / matrixSqrt
        W_lo2Lptc['forward'][1] = -W_lo2Lptc['forward'][0]
        W_lo2Lptc['forward'][2] = (j0 - matrixY) / matrixSqrt
        W_lo2Lptc['forward'][3] = -W_lo2Lptc['forward'][2]

        # backward
        W_lo2Lptc['backward'][0] = (matrixX - i0) / matrixSqrt
        W_lo2Lptc['backward'][1] = -W_lo2Lptc['backward'][0]
        W_lo2Lptc['backward'][2] = (matrixY - j0) / matrixSqrt
        W_lo2Lptc['backward'][3] = -W_lo2Lptc['backward'][2]

        W_lo2Lptc['left'] = cp.array([-1, 1, 0, 0])
        W_lo2Lptc['right'] = cp.array([1, -1, 0, 0])
        W_lo2Lptc['up'] = cp.array([0, 0, -1, 1])
        W_lo2Lptc['down'] = cp.array([0, 0, 1, -1])

        return W_lo2Lptc

    def lamina_process(self, laminaIpt):

        self.list_lamina.record_next(laminaIpt)

        opt_matrix_1 = self.compute_temporal_conv(self.list_lamina,
                                                  self.kernel_lamina,
                                                  self.list_lamina.pointer)
        opt_matrix_2 = self.compute_temporal_conv(self.list_lamina,
                                                  self.kernel_lamina,
                                                  (self.list_lamina.pointer - self.tau_lamina) % self.list_lamina_len)

        if opt_matrix_2 is not None:
            temporalDiff = opt_matrix_1 - opt_matrix_2
        else:
            temporalDiff = np.zeros_like(opt_matrix_1)

        l1Signal = np.maximum(temporalDiff, 0)  # ON
        l2Signal = np.maximum(-temporalDiff, 0)  # OFF

        # 存储正负部，正部L1、负部L2
        self.l1_list.record_next(l1Signal)
        self.l2_list.record_next(l2Signal)

        return l1Signal, l2Signal

    @staticmethod
    def shift_matrix(matrix, x0, direction, result):
        """把矩阵matrix往direction方向平移x0个像素位置
        """

        if direction == 'up':
            result[:-x0] = matrix[x0:]
        elif direction == 'down':
            result[x0:] = matrix[:-x0]
        elif direction == 'left':
            result[:, :-x0] = matrix[:, x0:]
        elif direction == 'right':
            result[:, x0:] = matrix[:, :-x0]
        else:
            raise ValueError("Invalid direction: choose from 'up', 'down', 'left', 'right'")

    def lobula_process(self, tm1Opt: cp.array,
                       tm2Opt: cp.array,
                       mi1Opt: cp.array,
                       tm3Opt: cp.array,
                       x0: int = 1):
        """lobula层的处理, 对应公式(5)~(8)

        Args:
            tm1Opt: TM1
            tm2Opt: TM2
            tm3Opt: TM3
            mi1Opt: Mi1
            x0: 像素偏置值

        Returns:
            做EMD匹配，输出d的四个背景响应方向的顺序为：右左下上
            输出结果为一个list列表类型
        """

        # 用于保存结果
        result = cp.zeros_like(tm1Opt)
        T4_matrix = []
        T5_matrix = []

        # 图像往右左下上四个方向平移的offset参数值
        for d in ('right', 'left', 'down', 'up'):
            self.shift_matrix(tm2Opt, x0, d, result)
            T4_matrix.append(tm1Opt * result)

            self.shift_matrix(tm3Opt, x0, d, result)
            T5_matrix.append(mi1Opt * result)

        res_T4_matrix = []
        res_T5_matrix = []

        res_T4_matrix.append(T4_matrix[0] - T4_matrix[1])
        res_T4_matrix.append(T4_matrix[1] - T4_matrix[0])
        res_T4_matrix.append(T4_matrix[2] - T4_matrix[3])
        res_T4_matrix.append(T4_matrix[3] - T4_matrix[2])

        res_T5_matrix.append(T5_matrix[0] - T5_matrix[1])
        res_T5_matrix.append(T5_matrix[1] - T5_matrix[0])
        res_T5_matrix.append(T5_matrix[2] - T5_matrix[3])
        res_T5_matrix.append(T5_matrix[3] - T5_matrix[2])

        # 存储t4
        for index, direction in enumerate(('right', 'left', 'down', 'up')):
            self.t4_list[direction].record_next(res_T4_matrix[index])

        return res_T4_matrix, res_T5_matrix

    @staticmethod
    def lobula_plate_lptc_process(t4Opt, t5Opt, kernelLo2Lptc):
        """实现公式(9)

        Args:
            t4Opt: list类型, 存储负部的四个背景运动方向矩阵
            t5Opt: list类型, 存储正部的四个背景运动方向矩阵
            kernelLo2Lptc: 字典类型, 键的值与下面motion_status变量的键的值一致

        Returns:
            返回字典类型, 代表六种运动模式的分量
        """

        motion_status = {direction: cp.zeros_like(t4Opt[0]) for direction in
                         ['forward', 'backward', 'left', 'right', 'up', 'down']}

        for key in ('forward', 'backward'):
            for i in range(4):
                motion_status[key] += (t4Opt[3 - i] + t5Opt[3 - i]) * kernelLo2Lptc[key][i]

        for key in ('left', 'right', 'up', 'down'):
            for i in range(4):
                motion_status[key] += (t4Opt[i] + t5Opt[i]) * kernelLo2Lptc[key][i]

        return motion_status

    def self_motion_feedback(self, lptcOpt, icputMatrix, W_lptc2stmd):
        """ 实现公式(14)、(19)
            Args:
                lptcOpt: 字典类型，公式(9)中的输出
                icputMatrix: 原始灰度图
                W_lptc2stmd: 速度向量矩阵
            Returns:
                C_feedback: 自运动光流
            """

        # 返回沿图像垂直方向和水平方向的梯度
        grad_matrix = cp.gradient(icputMatrix)

        # C_LO : [grad_x, -grad_x, grad_y, -grad_y]
        C_LO = [None] * 4
        # C_LO[0] = grad_matrix[0]
        # C_LO[1] = -grad_matrix[0]
        # C_LO[2] = grad_matrix[1]
        # C_LO[3] = -grad_matrix[1]

        self.C_LO_list[0].record_next(grad_matrix[0])
        self.C_LO_list[1].record_next(-grad_matrix[0])
        self.C_LO_list[2].record_next(grad_matrix[1])
        self.C_LO_list[3].record_next(-grad_matrix[1])

        for i in range(4):
            C_LO[i] = NewModel.compute_temporal_conv(self.C_LO_list[i], self.C_LO_kernel, self.C_LO_list[i].pointer)

        # 计算自运动光流
        C_feedback = cp.zeros_like(icputMatrix)
        for v in lptcOpt.keys():
            for w in range(4):
                C_feedback += lptcOpt[v] * C_LO[w] * W_lptc2stmd[v][w]

        # 存储当前的计算值
        self.feedback_list.record_next(C_feedback)

        return C_feedback

    def lobula_plate_stmd_process(self, t5Opt, C_feedback, feedbackList, t4OptList, Mi1, listL2Opt, gammaTau,
                                  eta_on=-0.1,
                                  eta_off=-0.1):
        """ 实现公式(20)、(21)

            Args:
                t5Opt: list类型，存储负部的右左下上四个背景运动方向矩阵
                C_feedback: cp.array类型，表示背景光流
                feedbackList: 存储C_feedback的循环列表
                t4OptList: 存储四个基本的背景方向响应矩阵的循环列表
                Mi1: 正部
                listL2Opt: 存储负部的循环列表
                gammaTau: 字典类型，键为不同tau的gamma核
                eta_on: 调节背景光流参数
                eta_off: 调节背景光流参数

            Returns:
                输出C_STMD
            """

        C_STMD_Output = {}  # (20)式输出
        for tau in gammaTau.keys():
            self.restrain_ON[tau].record_next(cp.maximum(Mi1 - eta_on * C_feedback, 0))

            self.restrain_OFF[tau].record_next(
                cp.maximum(NewModel.compute_temporal_conv(listL2Opt, gammaTau[tau], listL2Opt.pointer) +
                           eta_off * NewModel.compute_temporal_conv(feedbackList, gammaTau[tau], feedbackList.pointer),
                           0))

            C_STMD_Output[tau] = NewModel.compute_temporal_conv(self.restrain_ON[tau],
                                                                self.restrain_kernel,
                                                                self.restrain_ON[
                                                                    tau].pointer) * NewModel.compute_temporal_conv(
                self.restrain_OFF[tau], self.restrain_kernel, self.restrain_OFF[tau].pointer)

        # C_STMD_Output = {}  # (20)式输出
        # for tau in gammaTau.keys():
        #     C_STMD_Output[tau] = \
        #         cp.maximum(Mi1 - eta_on * C_feedback, 0) * \
        #         cp.maximum(compute_circularlist_conv(listL2Opt, gammaTau[tau]) +
        #                    eta_off * compute_circularlist_conv(feedbackList, gammaTau[tau]), 0)

        STMD_Output = {}  # (21)式输出
        # 计算小目标运动光流
        for tau in gammaTau.keys():
            STMD_Output[tau] = {}

            for index, direction in enumerate(('right', 'left', 'down', 'up')):
                STMD_Output[tau][direction] = C_STMD_Output[tau] * (
                        t5Opt[index] + NewModel.compute_temporal_conv(t4OptList[direction], gammaTau[tau],
                                                                      t4OptList[direction].pointer))

        # stmd_max = cp.max(STMD_Output[tau][direction])
        # stmd_min = cp.min(STMD_Output[tau][direction])
        # if stmd_max != stmd_min:
        #     STMD_Output[tau][direction] -= stmd_min
        #     STMD_Output[tau][direction] /= (stmd_max - stmd_min)

        return STMD_Output, C_STMD_Output, Mi1 - eta_on * C_feedback

    def VNC_TSDN(self, C_STMD_Output, tauList, lamda_DT):
        """实现公式(22)~(25)
            Args:
                C_STMD_Output: 双重字典
                tauList: 预定义tau列表
                lamda_DT: list类型，存储正部的右左下上四个背景运动方向矩阵

            Returns:
                1. C_TSDN
                2. fit_tau: 最优tau
                3. x0,j0: 目标位置
        """

        position = {}  # 保存每个tau下不同方向的点集
        position_union = {}  # 保存每个tau下四个方向的点集和
        C_TSDN = {}  # 公式(25)

        for tau in tauList:
            C_TSDN[tau] = {}
            position[tau] = {}
            position_union[tau] = cp.zeros((0, 2), dtype=cp.uint16)

            for direction in ['left', 'right', 'up', 'down']:
                position[tau][direction] = cp.argwhere(C_STMD_Output[tau][direction] > lamda_DT)

                if len(position[tau][direction]):
                    C_TSDN[tau][direction] = cp.mean(C_STMD_Output[tau][direction] > lamda_DT)
                    position_union[tau] = cp.vstack([position_union[tau], position[tau][direction]])
                else:
                    # 固定tau和方向下找不到点集, 就设置对应TSDN为0
                    C_TSDN[tau][direction] = 0.

            # 最后求每个tau下四个方向点集的并集
            if len(position_union[tau]) > 1:
                position_union[tau] = self.union_rows(position_union[tau])

        # 如果每个tau下点集都为0或者1,说明没有检测到结果
        if all(len(position_union[tau]) <= 1 for tau in tauList):
            return C_TSDN, -1.0, -1.0, -1.0, -1.0, -1.0

        # 如果fit_tau下每个方向点集都为0或者1,说明没有检测到结果
        fit_tau = self.compute_fit_tau(position_union, tauList)
        if all(len(position[fit_tau][direction]) <= 1 for direction in ['left', 'right', 'up', 'down']):
            return C_TSDN, -1.0, -1.0, -1.0, -1.0, -1.0

        # 方向与数值的对应：{1: 'left', 2: 'right', 3: 'up', 4: 'down'}
        fit_direction, x0, j0 = self.compute_fit_direction(position[fit_tau], ['left', 'right', 'up', 'down'])
        return C_TSDN, position, fit_tau, fit_direction, x0, j0

    @staticmethod
    def union_rows(mat):
        """求二维矩阵按行的并集

        Args:
            mat: 输入矩阵 (CuPy数组).

        Returns:
            包含并集行的CuPy数组.
        """

        _, unique_indices = cp.unique(mat[:, 0], return_index=True)
        return mat[unique_indices]

    @staticmethod
    def compute_density(positions):
        """计算某个点集中的方差与均值
        """

        # 计算质心点 (x0, y0)
        x0, y0 = cp.mean(positions, axis=0)

        # 计算每个点到质心点的欧式距离
        distances = cp.sqrt((positions[:, 0] - x0) ** 2 + (positions[:, 1] - y0) ** 2)

        # 计算欧式距离和的均值
        avg_of_distances = cp.sum(distances) / distances.shape[0]

        return avg_of_distances, cp.round(x0), cp.round(y0)

    @staticmethod
    def compute_fit_direction(position, iteration_list=None):
        """返回迭代列表中最优的迭代值与对应的均值点(x0,j0)

            Args:
                position: 对应的位置点集矩阵
                iteration_list: 迭代列表


            Returns:
                min_row[1]: 最优的迭代值
                min_row[2]、min_row[3]: 均值点
        """

        choose_matrix = cp.zeros((0, 4))  # 保存迭代值对应的方差、迭代值、均值点(x0,j0)

        i = cp.array(1)
        # 求不同tau或者方向下对应点集的方差与均值点
        for data in iteration_list:
            # 点集的数量一定要大于等于1
            if len(position[data]) > 1:
                dis, x0, j0 = NewModel.compute_density(position[data])
                choose_matrix = cp.vstack([choose_matrix, [dis, i, x0, j0]])
            i += 1

        # 按第一列数据排序矩阵
        sorted_matrix = choose_matrix[choose_matrix[:, 0].argsort()]

        # 找到数值最小的那一行
        min_row = sorted_matrix[0]

        return int(min_row[1]), min_row[2], min_row[3]

    @staticmethod
    def compute_fit_tau(position, iteration_list=None):
        """返回迭代列表中最优的迭代值与对应的均值点(x0,j0)

            Args:
                position: 对应的位置点集矩阵
                iteration_list: 迭代列表


            Returns:
                min_row[1]: 最优的迭代值
                min_row[2]、min_row[3]: 均值点
        """

        choose_matrix = cp.zeros((0, 4))  # 保存迭代值对应的方差、迭代值、均值点(x0,j0)

        # 求不同tau或者方向下对应点集的方差与均值点
        for data in iteration_list:
            # 点集的数量一定要大于1
            if len(position[data]) > 1:
                dis, x0, j0 = NewModel.compute_density(position[data])
                choose_matrix = cp.vstack([choose_matrix, [dis, cp.array(data), x0, j0]])

        # 按第一列数据排序矩阵
        sorted_matrix = choose_matrix[choose_matrix[:, 0].argsort()]

        # 找到数值最小的那一行
        min_row = sorted_matrix[0]

        return int(min_row[1])

    @staticmethod
    def direction_of_target_motion(C_TSDN, fit_tau):
        """实现公式(26)

            Args:
                C_TSDN: 双重字典,(25)式输出
                fit_tau: 最优的tau值

            Returns:
                返回C_TSDN_Dir
        """

        # 生成从0到7*cp.pi/4的8个均匀分布的角度
        angles = cp.linspace(0, 7 * cp.pi / 4, 8)

        # 计算正弦值和余弦值
        sin_values = cp.sin(angles)
        cos_values = cp.cos(angles)

        # 将正弦值和余弦值组合成一个二维矩阵
        C_Dp = cp.row_stack((sin_values, cos_values))

        # 四个基本方向
        C_alpha_d = cp.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
        C_beta_d = cp.dot(C_alpha_d, C_Dp)

        vector = cp.empty(4)
        for index, direction in enumerate(['left', 'right', 'up', 'down']):
            vector[index] = C_TSDN[fit_tau][direction]

        return cp.dot(vector, C_beta_d)

    @staticmethod
    def velocity_of_target_motion(C_TSDN, tauList=None):
        """实现公式(27)

            Args:
                C_TSDN: 双重字典,(25)式输出
                tauList: 预定义tau列表

            Returns:
                返回C_TSDN_Vel
        """

        C_TSDN_Vel = {}

        for tau in tauList:
            value = 0.
            for direction in ['left', 'right', 'up', 'down']:
                value += C_TSDN[tau][direction]

            C_TSDN_Vel[tau] = value

        return C_TSDN_Vel

    @staticmethod
    def direction_of_target_position(P_prey, x0, j0):
        """实现公式(28)

            Args:
                P_prey: 目标位置
                x0、j0:图像中心点
                x0: 行下标
                j0: 列下标

            Returns:
                返回C_TSDN_Pos
        """

        # 生成从0到7*cp.pi/4的8个均匀分布的角度
        angles = cp.linspace(0, 7 * cp.pi / 4, 8)

        # 计算正弦值和余弦值
        sin_values = cp.sin(angles)
        cos_values = cp.cos(angles)

        # 将正弦值和余弦值组合成一个二维矩阵
        C_Dp = cp.row_stack((sin_values, cos_values))

        vector = cp.array([P_prey[1] - j0, P_prey[0] - x0])

        return cp.dot(vector, C_Dp)

    @staticmethod
    def compute_temporal_conv(iptCell, kernel, pointer=None):
        """
        Computes temporal convolution.

        Parameters:
        - iptCell: A list of arrays where each element has the same dimension.
        - kernel: A vector representing the convolution kernel.
        - headPointer: Head pointer of the input cell array (optional).

        Returns:
        - optMatrix: The result of the temporal convolution.
        """

        # Default value for headPointer
        if pointer is None:
            pointer = len(iptCell) - 1

        # Initialize output matrix
        if iptCell[pointer] is None:
            return None

        # Ensure kernel is a vector
        kernel = np.squeeze(kernel)
        if not np.ndim(kernel) == 1:
            raise ValueError('The kernel must be a vector.')

        # Determine the lengths of input cell array and kernel
        k1 = len(iptCell)
        k2 = len(kernel)
        length = min(k1, k2)

        optMatrix = np.zeros_like(iptCell[pointer])
        # Perform temporal convolution
        for t in range(length):
            j = (pointer - t) % k1
            if np.abs(kernel[t]) > 1e-16 and iptCell[j] is not None:
                optMatrix += iptCell[j] * kernel[t]

        return optMatrix

    def run(self, gray_img, my_list=None):
        """ 主函数
        """

        '''1.Retina'''
        retinaOpt = gaussian_filter(gray_img, self.sigma)

        '''2.Lamina'''
        l1, l2 = self.lamina_process(retinaOpt)  # mil, tm1

        '''3.Medulla'''
        # ME ON/OFF Channel
        mi1, tm3 = l1, self.compute_temporal_conv(self.l1_list, self.gamma_kernel_emd, self.l1_list.pointer)
        tm1, tm2 = l2, self.compute_temporal_conv(self.l2_list, self.gamma_kernel_emd, self.l2_list.pointer)

        '''4.Lobula'''
        # LO
        # 做EMD匹配，t4和t5输出的四个背景响应方向的顺序为：右左下上
        t4, t5 = self.lobula_process(tm1, tm2, mi1, tm3, self.x0)

        # LOP LPTC
        lptc = self.lobula_plate_lptc_process(t4, t5, self.W_lo2Lptc)

        # self_motion_feedback
        feedback = self.self_motion_feedback(lptc, gray_img, self.W_lptc2stmd)

        # LOP STMD
        stmd, *_ = self.lobula_plate_stmd_process(t5, feedback, self.feedback_list, self.t4_list, mi1, self.l2_list,
                                                 self.gamma_kernel_stmd, self.eta_on, self.eta_off)

        # for tau in self.stmd_tau:
        #     my_list[tau] = stmd[tau]['right'].get()

        # VNC TSDN
        target_pos = cp.array([0., 0.])
        C_TSDN, position_matrix, fit_tau, fit_direction, target_pos[0], target_pos[1] = self.VNC_TSDN(stmd,
                                                                                                      self.stmd_tau,
                                                                                                      self.lamda_DT)

        my_list[0] = target_pos.get()
        if target_pos[0] != -1.0 or target_pos[1] != -1.0:
            # 拿到目标的位置, fit_tau和fit_direction下的stmd响应, 这个stmd响应下的点集
            my_list[0] = target_pos.get()
            my_list[1] = stmd[fit_tau][NewModel.find_indx[fit_direction]].get()
            my_list[2] = position_matrix[fit_tau][NewModel.find_indx[fit_direction]].get()

        return

        # 计算目标运动方向
        # C_TSDN_Dir = direction_of_target_motion(C_TSDN, fit_tau)

        # 计算目标运动速度
        # C_TSDN_Vel = velocity_of_target_motion(C_TSDN, tau2)

        # 计算目标位置方向
        # C_TSDN_Pos = direction_of_target_position(target_pos, grayImg.shape[0] // 2, grayImg.shape[1] // 2)

    def run2(self, gray_img, my_list=None):
        """ 主函数
        """

        '''1.Retina'''
        retinaOpt = gaussian_filter(gray_img, self.sigma)

        my_list[0] = retinaOpt  # save

        '''2.Lamina'''
        l1, l2 = self.lamina_process(retinaOpt)  # mil, tm1

        my_list[1] = l1  # save
        my_list[2] = l2  # save

        '''3.Medulla'''
        # ME ON/OFF Channel
        mi1, tm3 = l1, self.compute_temporal_conv(self.l1_list, self.gamma_kernel_emd, self.l1_list.pointer)
        tm1, tm2 = l2, self.compute_temporal_conv(self.l2_list, self.gamma_kernel_emd, self.l2_list.pointer)

        my_list[3] = tm3  # save
        my_list[4] = tm2  # save

        '''4.Lobula'''
        # LO
        # 做EMD匹配，t4和t5输出的四个背景响应方向的顺序为：右左下上
        t4, t5 = self.lobula_process(tm1, tm2, mi1, tm3, self.x0)

        my_list[5] = t4  # save
        my_list[6] = t5  # save

        # LOP LPTC
        lptc = self.lobula_plate_lptc_process(t4, t5, self.W_lo2Lptc)

        my_list[7] = lptc  # save

        # self_motion_feedback
        feedback = self.self_motion_feedback(lptc, gray_img, self.W_lptc2stmd)

        my_list[8] = -feedback  # save

        # LOP STMD
        stmd, *_ = self.lobula_plate_stmd_process(t5, feedback, self.feedback_list, self.t4_list, mi1, self.l2_list,
                                                 self.gamma_kernel_stmd, self.eta_on, self.eta_off)

        # VNC TSDN
        target_pos = cp.array([0., 0.])
        C_TSDN, position_matrix, fit_tau, fit_direction, target_pos[0], target_pos[1] = self.VNC_TSDN(stmd,
                                                                                                      self.stmd_tau,
                                                                                                      self.lamda_DT)

        my_list[9] = target_pos  # save

        # 没找到fit_tau和fit_direction
        if target_pos[0] == -1.0:
            my_list[10] = -1
        else:
            my_list[10] = stmd[fit_tau][NewModel.find_indx[fit_direction]]

        return

        # 计算目标运动方向
        # C_TSDN_Dir = direction_of_target_motion(C_TSDN, fit_tau)

        # 计算目标运动速度
        # C_TSDN_Vel = velocity_of_target_motion(C_TSDN, tau2)

        # 计算目标位置方向
        # C_TSDN_Pos = direction_of_target_position(target_pos, grayImg.shape[0] // 2, grayImg.shape[1] // 2)

    def run3(self, gray_img, my_list=None):
        """ 主函数
        """

        '''1.Retina'''
        retinaOpt = gaussian_filter(gray_img, self.sigma)

        '''2.Lamina'''
        l1, l2 = self.lamina_process(retinaOpt)  # mil, tm1

        '''3.Medulla'''
        # ME ON/OFF Channel
        mi1, tm3 = l1, self.compute_temporal_conv(self.l1_list, self.gamma_kernel_emd, self.l1_list.pointer)
        tm1, tm2 = l2, self.compute_temporal_conv(self.l2_list, self.gamma_kernel_emd, self.l2_list.pointer)

        '''4.Lobula'''
        # LO
        # 做EMD匹配，t4和t5输出的四个背景响应方向的顺序为：右左下上
        t4, t5 = self.lobula_process(tm1, tm2, mi1, tm3, self.x0)

        # LOP LPTC
        lptc = self.lobula_plate_lptc_process(t4, t5, self.W_lo2Lptc)

        # self_motion_feedback
        feedback = self.self_motion_feedback(lptc, gray_img, self.W_lptc2stmd)

        # LOP STMD
        stmd, f, t = self.lobula_plate_stmd_process(t5, feedback, self.feedback_list, self.t4_list, mi1, self.l2_list,
                                              self.gamma_kernel_stmd, self.eta_on, self.eta_off)

        for tau in self.stmd_tau:
            my_list[tau] = f[tau].get()

    def run4(self, gray_img, my_list=None):
        """ 主函数
        """

        '''1.Retina'''
        retinaOpt = gaussian_filter(gray_img, self.sigma)

        '''2.Lamina'''
        l1, l2 = self.lamina_process(retinaOpt)  # mil, tm1

        '''3.Medulla'''
        # ME ON/OFF Channel
        mi1, tm3 = l1, self.compute_temporal_conv(self.l1_list, self.gamma_kernel_emd, self.l1_list.pointer)
        tm1, tm2 = l2, self.compute_temporal_conv(self.l2_list, self.gamma_kernel_emd, self.l2_list.pointer)

        '''4.Lobula'''
        # LO
        # 做EMD匹配，t4和t5输出的四个背景响应方向的顺序为：右左下上
        t4, t5 = self.lobula_process(tm1, tm2, mi1, tm3, self.x0)

        # LOP LPTC
        lptc = self.lobula_plate_lptc_process(t4, t5, self.W_lo2Lptc)

        # self_motion_feedback
        feedback = self.self_motion_feedback(lptc, gray_img, self.W_lptc2stmd)

        # LOP STMD
        stmd, dir_stmd, st = self.lobula_plate_stmd_process(t5, feedback, self.feedback_list, self.t4_list, mi1, self.l2_list,
                                              self.gamma_kernel_stmd, self.eta_on, self.eta_off)

        for tau in self.stmd_tau:
            my_list[tau] = stmd[tau]['right'].get()

        # VNC TSDN
        target_pos = cp.array([-1.0, -1.0])
        C_TSDN, position_matrix, fit_tau, fit_direction, target_pos[0], target_pos[1] = self.VNC_TSDN(stmd,
                                                                                                      self.stmd_tau, self.lamda_DT)

        if target_pos[0] != -1.0:
            print(target_pos, fit_tau, NewModel.find_indx[fit_direction])

    def run5(self, gray_img, my_list=None):
        """ 主函数
        """

        '''1.Retina'''
        retinaOpt = gaussian_filter(gray_img, self.sigma)

        '''2.Lamina'''
        l1, l2 = self.lamina_process(retinaOpt)  # mil, tm1

        '''3.Medulla'''
        # ME ON/OFF Channel
        mi1, tm3 = l1, self.compute_temporal_conv(self.l1_list, self.gamma_kernel_emd, self.l1_list.pointer)
        tm1, tm2 = l2, self.compute_temporal_conv(self.l2_list, self.gamma_kernel_emd, self.l2_list.pointer)

        '''4.Lobula'''
        # LO
        # 做EMD匹配，t4和t5输出的四个背景响应方向的顺序为：右左下上
        t4, t5 = self.lobula_process(tm1, tm2, mi1, tm3, self.x0)

        # LOP LPTC
        lptc = self.lobula_plate_lptc_process(t4, t5, self.W_lo2Lptc)

        # self_motion_feedback
        feedback = self.self_motion_feedback(lptc, gray_img, self.W_lptc2stmd)

        # LOP STMD
        stmd, st, f = self.lobula_plate_stmd_process(t5, feedback, self.feedback_list, self.t4_list, mi1, self.l2_list,
                                              self.gamma_kernel_stmd, self.eta_on, self.eta_off)

        my_list[0] = mi1.get()
        my_list[1] = f.get()
        my_list[2] = self.eta_on*feedback.get()

    def run6(self, gray_img, my_list=None):
        """ 主函数
        """

        '''1.Retina'''
        retinaOpt = gaussian_filter(gray_img, self.sigma)

        '''2.Lamina'''
        l1, l2 = self.lamina_process(retinaOpt)  # mil, tm1

        '''3.Medulla'''
        # ME ON/OFF Channel
        mi1, tm3 = l1, self.compute_temporal_conv(self.l1_list, self.gamma_kernel_emd, self.l1_list.pointer)
        tm1, tm2 = l2, self.compute_temporal_conv(self.l2_list, self.gamma_kernel_emd, self.l2_list.pointer)

        '''4.Lobula'''
        # LO
        # 做EMD匹配，t4和t5输出的四个背景响应方向的顺序为：右左下上
        t4, t5 = self.lobula_process(tm1, tm2, mi1, tm3, self.x0)

        # LOP LPTC
        lptc = self.lobula_plate_lptc_process(t4, t5, self.W_lo2Lptc)

        # self_motion_feedback
        feedback = self.self_motion_feedback(lptc, gray_img, self.W_lptc2stmd)

        # LOP STMD
        stmd, *_ = self.lobula_plate_stmd_process(t5, feedback, self.feedback_list, self.t4_list, mi1, self.l2_list,
                                                 self.gamma_kernel_stmd, self.eta_on, self.eta_off)

        # for tau in self.stmd_tau:
        #     my_list[tau] = stmd[tau]['right'].get()

        # VNC TSDN
        target_pos = cp.array([0., 0.])
        C_TSDN, position_matrix, fit_tau, fit_direction, target_pos[0], target_pos[1] = self.VNC_TSDN(stmd,
                                                                                                      self.stmd_tau,
                                                                                                      self.lamda_DT)

        my_list[0] = target_pos.get()
        if target_pos[0] != -1.0 or target_pos[1] != -1.0:
            # 拿到目标的位置, fit_tau和fit_direction下的stmd响应, 这个stmd响应下的点集
            my_list[0] = target_pos.get()
            my_list[1] = stmd[fit_tau][NewModel.find_indx[fit_direction]].get()
            my_list[2] = position_matrix[fit_tau][NewModel.find_indx[fit_direction]].get()

        return

@dataclass
class CircularList(list):
    """
    CircularList represents a circular buffer for storing input matrices.
    """
    initLen: int = 0  # Default length of the circular buffer
    pointer: int = -1  # Pointer to current position in the circular buffer

    def __post_init__(self) -> List:
        """
        Post-initialization method to initialize the circular buffer with empty list.
        """
        if self.initLen:
            self.extend([None] * self.initLen)

    def reset(self) -> None:
        """
        Method to reset the circular buffer to a new length.
        """
        self.clear()  # clear List
        self.__post_init__()  # Reinitialize CircularList object
        self.pointer = -1  # Reset the pointer to initial position

    def move_pointer(self) -> None:
        """
        Method to move the circular buffer pointer to the next position.
        """
        self.pointer = (self.pointer + 1) % self.initLen

    def cover(self, iptMatrix: Any) -> None:
        """
        Method to cover the current position of the circular buffer with an input matrix.

        Parameters:
        - iptMatrix: Input matrix to cover the current position.
        """
        self[self.pointer] = iptMatrix

    def record_next(self, iptMatrix: Any) -> None:
        """
        Method to record an input matrix in the circular buffer, after moving the pointer to the next position.

        Parameters:
        - iptMatrix: Input matrix to be recorded.
        """
        self.move_pointer()
        self.cover(iptMatrix)
