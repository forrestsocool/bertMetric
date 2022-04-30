import paddle
from paddle import nn
import paddle.nn.functional as F
import math

class ArcMarginProduct(paddle.nn.Layer):
    def __init__(self, feature_dim, class_dim, s=64.0, m=0.40, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.weight = paddle.create_parameter(shape=[feature_dim, class_dim], dtype="float32",
                                              default_initializer=nn.initializer.KaimingUniform())
        self.class_dim = class_dim
        self.s = s
        self.m = m

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input: paddle.Tensor, label):
        cosine = paddle.nn.functional.linear(paddle.nn.functional.normalize(input), paddle.nn.functional.normalize(self.weight))
        sine = paddle.sqrt(paddle.clip(1.0 - paddle.pow(cosine, 2), min=0, max=1))
        # cos(x+m) = cos(x)*cos(m)-sin(x)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            # 输出为正的加角M增加难度, 否则原样输出：保证了theta加m仍然在0-Pi的单调区间
            phi = paddle.where(cosine > 0, phi, cosine)
        else:
            # 一阶泰勒展开： cos(theta + m) = cos(theta) - m * sin(theta) >= cos(theta) - m * sin(math.pi - m)
            phi = paddle.where(cosine > self.th, phi, cosine - self.mm)

        # one_hot = paddle.zeros_like(cosine)
        # paddle.scatter_(one_hot, label.reshape(-1, 1), 1, overwrite=True)

        one_hot = paddle.nn.functional.one_hot(label, self.class_dim)
        one_hot = paddle.squeeze(one_hot, axis=1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

    def forward_test(self, input: paddle.Tensor):
        cosine = paddle.nn.functional.linear(paddle.nn.functional.normalize(input), paddle.nn.functional.normalize(self.weight))
        return cosine