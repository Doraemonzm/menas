import torch.nn as nn
import math





def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v



def scale_filters(filters, multiplier, base=8):
  """Scale `filters` by `factor`and round to the nearest multiple of `base`.
  Args:
    filters: Positive integer. The original filter size.
    multiplier: Positive float. The factor by which to scale the filters.
    base: Positive integer. The number of filters will be rounded to a multiple
        of this value.
  Returns:
    Positive integer, the scaled filter size.
  """
  round_half_up = int(filters * multiplier / base + 0.5)
  result = int(round_half_up * base)
  return max(result, base)



class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, cfgs, num_classes=1000):
        super(MobileNetV3, self).__init__()

        # setting of inverted residual blocks
        block_filters_multipliers = cfgs['width-mul']
        expansion_multipliers=cfgs['e']
        SE=cfgs['se']
        block_depths=cfgs['d']
        strides= cfgs['s']
        kernel_sizes=cfgs['ks']

        base_filters=[16, 16, 32, 64, 128, 256, 512, 1280]

        hs = [0, 0, 1, 1, 1]


        # building first layer
        input_channel = base_filters[0]
        self.stem = nn.Sequential(conv_3x3_bn(3, input_channel, 2))
        # building inverted residual blocks


        layer_count=0
        for block_id in range(5):


            block_filter=scale_filters(base_filters[block_id+1], block_filters_multipliers[block_id], base=8)
            # print('block_filter:', block_filter)
            block_stride= strides[block_id]
            block_depth= block_depths[block_id]
            block_kernel=kernel_sizes[layer_count: layer_count+block_depth]
            block_exp=expansion_multipliers[layer_count:layer_count+block_depth]
            block_se=SE[layer_count:layer_count+block_depth]
            block_hs=hs[block_id]

            block = nn.ModuleList([])



            for layer_id in range(block_depth):
                output_channel=block_filter
                exp_size =  scale_filters(input_channel, block_exp[layer_id], base=8)
                kernel_size= block_kernel[layer_id]
                if block_stride==2 and layer_id==0:
                    block.append(InvertedResidual(input_channel, exp_size, output_channel, kernel_size, block_stride, block_se[layer_id], block_hs))
                else:
                    block.append(
                        InvertedResidual(input_channel, exp_size, block_filter, kernel_size, 1, block_se[layer_id],
                              block_hs))
                input_channel=output_channel
                layer_count+=1
            setattr(self, f"block{block_id + 1}", block)

        # self.features = nn.Sequential(*network)
        # building last several layers
        # head:
        self.head = conv_1x1_bn(input_channel, exp_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
        self.classifier = nn.Sequential(
            nn.Linear(exp_size, base_filters[-1]),
            h_swish(),
            nn.Dropout(0.2),
            nn.Linear(base_filters[-1], num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        for i in range(5):
            block = getattr(self, f"block{i + 1}")
            for blk in block:
                x = blk(x)
        x = self.head(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()




