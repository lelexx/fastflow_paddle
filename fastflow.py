# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from turtle import forward
import paddle
import paddle.nn as nn
from resnet18 import resnet18
import paddle.nn.functional as F
import numpy as np
import os, sys

def subnet_conv_func(kernel_size, hidden_ratio, input_chw):

    def subnet_conv(in_channels, out_channels):
        hidden_channels = int(in_channels * hidden_ratio)
        return nn.Sequential(
            nn.Conv2D(in_channels, hidden_channels, kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv2D(hidden_channels, out_channels, kernel_size, padding="same"),
        )

    return subnet_conv

class MySequenceINN(nn.Layer):
    '''
    Normalizing Flows
    '''
    def __init__(self, *dims: int, force_tuple_output=False):
        super().__init__()

        self.shapes = [tuple(dims)]
        self.conditions = []
        self.module_list = nn.LayerList()

        self.force_tuple_output = force_tuple_output


    def append(self, module_class, cond=None, cond_shape=None, **kwargs):
        dims_in = [self.shapes[-1]]
        self.conditions.append(cond)

        if cond is not None:
            kwargs['dims_c'] = [cond_shape]

        module = module_class(dims_in, **kwargs)
        self.module_list.append(module)
        ouput_dims = module.output_dims(dims_in)
        assert len(ouput_dims) == 1, "Module has more than one output"
        self.shapes.append(ouput_dims[0])

    def __getitem__(self, item):
        return self.module_list.__getitem__(item)

    def __len__(self):
        return self.module_list.__len__()

    def __iter__(self):
        return self.module_list.__iter__()

    def forward(self, x_or_z, c = None,
                rev = False, jac = True) :

        iterator = range(len(self.module_list))
        log_det_jac = 0

        for i in iterator:
            x_or_z, j = self.module_list[i](x_or_z, jac=jac, rev=rev)
            log_det_jac = j + log_det_jac

        return x_or_z, log_det_jac
class MyAllInOneBlock(nn.Layer):
    def __init__(self, dims_in, dims_c=[],
                 subnet_constructor= None,
                 affine_clamping = 2.,
                 gin_block= False,
                 global_affine_init = 1.,
                 global_affine_type= 'SOFTPLUS',
                 permute_soft= False,
                 learned_householder_permutation = 0,
                 reverse_permutation = False):
        super().__init__()

        channels = dims_in[0][0]
        # rank of the tensors means 1d, 2d, 3d tensor etc.
        self.input_rank = len(dims_in[0]) - 1
        # tuple containing all dims except for batch-dim (used at various points)
        self.sum_dims = tuple(range(1, 2 + self.input_rank))

        if len(dims_c) == 0:
            self.conditional = False
            self.condition_channels = 0
        else:
            assert tuple(dims_c[0][1:]) == tuple(dims_in[0][1:]), \
                F"Dimensions of input and condition don't agree: {dims_c} vs {dims_in}."
            self.conditional = True
            self.condition_channels = sum(dc[0] for dc in dims_c)

        split_len1 = channels - channels // 2
        split_len2 = channels // 2
        self.splits = [split_len1, split_len2]

        try:
            self.permute_function = {0: F.linear,
                                     1: F.conv1d,
                                     2: F.conv2d,
                                     3: F.conv3d}[self.input_rank]
        except KeyError:
            raise ValueError(f"Data is {1 + self.input_rank}D. Must be 1D-4D.")

        self.in_channels         = channels
        self.clamp               = affine_clamping
        self.GIN                 = gin_block
        self.reverse_pre_permute = reverse_permutation
        self.householder         = learned_householder_permutation

        if permute_soft and channels > 512:
            warnings.warn(("Soft permutation will take a very long time to initialize "
                           f"with {channels} feature channels. Consider using hard permutation instead."))

        # global_scale is used as the initial value for the global affine scale
        # (pre-activation). It is computed such that
        # global_scale_activation(global_scale) = global_affine_init
        # the 'magic numbers' (specifically for sigmoid) scale the activation to
        # a sensible range.
        
        if global_affine_type == 'SIGMOID':
            global_scale = 2. - np.log(10. / global_affine_init - 1.)
            self.global_scale_activation = (lambda a: 10 * paddle.sigmoid(a - 2.))
        elif global_affine_type == 'SOFTPLUS':
            global_scale = 2. * np.log(np.exp(0.5 * 10. * global_affine_init) - 1)
            self.softplus = nn.Softplus(beta=0.5)
            self.global_scale_activation = (lambda a: 0.1 * self.softplus(a))
        elif global_affine_type == 'EXP':
            global_scale = np.log(global_affine_init)
            self.global_scale_activation = (lambda a: paddle.exp(a))
        else:
            raise ValueError('Global affine activation must be "SIGMOID", "SOFTPLUS" or "EXP"')

        self.global_scale = paddle.create_parameter( shape=[1, self.in_channels, *([1] * self.input_rank)], dtype='float32') 
        self.global_offset = paddle.create_parameter( shape=[1, self.in_channels, *([1] * self.input_rank)], dtype='float32')


        self.global_scale.set_value(np.ones(self.global_scale.shape, dtype = np.float32))
        self.global_offset.set_value(np.zeros(self.global_offset.shape, dtype = np.float32))
        self.global_scale.stop_gradient = False
        self.global_offset.stop_gradient = False
        
        w = np.zeros((channels, channels), dtype = np.float32) ####
        for i, j in enumerate(np.random.permutation(channels)): ###
            w[i, j] = 1. ###

        self.w_perm =paddle.create_parameter( shape=[channels, channels, *([1] * self.input_rank)], dtype='float32')
        self.w_perm_inv =paddle.create_parameter( shape=[channels, channels, *([1] * self.input_rank)], dtype='float32')
        self.w_perm.stop_gradient = True
        self.w_perm_inv.stop_gradient = True
        self.w_perm.set_value(w.reshape(self.w_perm.shape))
        self.w_perm_inv.set_value(w.T.reshape(self.w_perm_inv.shape))
            

        if subnet_constructor is None:
            raise ValueError("Please supply a callable subnet_constructor"
                             "function or object (see docstring)")
        self.AFFINE_2 = True
        self.AFFINE_1 = True

        if self.AFFINE_1:
            self.subnet = subnet_constructor(self.splits[0] + self.condition_channels, 2 * self.splits[1])
        if self.AFFINE_2:
            self.subnet_2 = subnet_constructor(self.splits[0] + self.condition_channels, 2 * self.splits[1])
        self.last_jac = None



    def _permute(self, x, rev=False):
        '''Performs the permutation and scaling after the coupling operation.
        Returns transformed outputs and the LogJacDet of the scaling operation.'''
        if self.GIN:
            scale = 1.
            perm_log_jac = 0.
        else:
            scale = self.global_scale ####
            perm_log_jac = paddle.sum(paddle.log(scale)) #####

        if rev:
            return ((self.permute_function(x, self.w_perm_inv) - self.global_offset) / scale,
                    perm_log_jac)
        else:

            return (self.permute_function(x * scale + self.global_offset, self.w_perm), #####
                     perm_log_jac)

    def _affine(self, x, a, rev=False):
        '''Given the passive half, and the pre-activation outputs of the
        coupling subnetwork, perform the affine coupling operation.
        Returns both the transformed inputs and the LogJacDet.'''
        a *= 0.1 #####
        ch = x.shape[1] ######

        sub_jac = self.clamp * paddle.tanh(a[:, :ch]) ######
        #sub_jac = self.clamp * self.f_clamp(a[:, :ch]) ######

        if self.GIN:
            sub_jac -= paddle.mean(sub_jac, dim=self.sum_dims, keepdim=True)
        
        if not rev:
            return (x * paddle.exp(sub_jac) + a[:, ch:],   #############
                    paddle.sum(sub_jac, axis=self.sum_dims))
        else:
            return ((x - a[:, ch:]) * paddle.exp(-sub_jac),
                    -paddle.sum(sub_jac, dim=self.sum_dims))

    def forward(self, x, c=[], rev=False, jac=True):

        '''See base class docstring'''
        if self.householder:
            self.w_perm = self._construct_householder_permutation()
            if rev or self.reverse_pre_permute:
                self.w_perm_inv = self.w_perm.transpose(0, 1).contiguous()
   
        
        if rev:
            x, global_scaling_jac = self._permute(x[0], rev=True)
            x = (x,)
        elif self.reverse_pre_permute:
            x = (self._pre_permute(x[0], rev=False),)
            
        
        x1, x2 = paddle.split(x, self.splits, axis=1) #######
        if not rev:
            j2 = 0
            if self.AFFINE_1:
                a1 = self.subnet(x1)  #####
                x2, j2 = self._affine(x2, a1) ###### change x2
            
            j1 = 0
            if self.AFFINE_2:
                a2 = self.subnet(x2)
                x1, j1 = self._affine(x1, a2) ###### change x1
            
        else:
            a1 = self.subnet(x1)
            x2, j2 = self._affine(x2, a1, rev=True)
            

        log_jac_det = j2 + j1####
        
        x_out = paddle.concat((x1, x2), 1) ######

        if not rev:
            x_out, global_scaling_jac = self._permute(x_out, rev=False) ######
        elif self.reverse_pre_permute:
            x_out = self._pre_permute(x_out, rev=True)

        # add the global scaling Jacobian to the total.
        # trick to get the total number of non-channel dimensions:
        # number of elements of the first channel of the first batch member
        n_pixels = x_out[0, :1].numel() # w * h


        log_jac_det += (-1)**rev * n_pixels * global_scaling_jac

        return x_out, log_jac_det

    def output_dims(self, input_dims):
        return input_dims


def nf_fast_flow(input_chw, conv3x3_only, hidden_ratio, flow_steps, clamp=2.0):
    ### create fastflow module
    nodes = MySequenceINN(*input_chw)
    step = 2
    for i in range(flow_steps):
        if i % step == (step - 1) and not conv3x3_only: ### conv 1x1
            kernel_size = 1
        else:                                           ### conv 3x3
            kernel_size = 3
        nodes.append(
            MyAllInOneBlock,
            subnet_constructor=subnet_conv_func(kernel_size, hidden_ratio, input_chw),
            affine_clamping=clamp,
            permute_soft=False,
        )

    return nodes

    
class FastFlow(nn.Layer):
    '''
    FastFLow model
    '''
    def __init__(self,
        flow_steps = 8,
        input_size = 256,
        conv3x3_only=True,
        hidden_ratio=1.0,
        use_norm = True,
        momentum = 0.95,
        channels = [64, 128, 256],
        scales = [4, 8, 16],
        clamp = 2.0,
        ):
        super().__init__()
        #### Moudle1: Encoder - resnet18
        self.feature_extractor = resnet18(pretrained=True)          
        for param in self.feature_extractor.parameters():
            param.stop_gradient = True

        self.input_size = input_size
        #### Moudle2: Norm - BatchNorm
        self.Norm = use_norm
        if self.Norm:
            self.norms = nn.LayerList()
            for in_channels, scale in zip(channels, scales):                
                self.norms.append(
                    nn.BatchNorm2D(
                        in_channels, momentum=momentum
                    )
                )
        #### Moudle3: 2D Normalizing Flows - fastflow
        self.nf_flows = nn.LayerList()
        for in_channels, scale in zip(channels, scales):
            self.nf_flows.append(
                nf_fast_flow(
                    [in_channels, int(input_size / scale), int(input_size / scale)],
                    conv3x3_only=conv3x3_only,
                    hidden_ratio=hidden_ratio,
                    flow_steps=flow_steps,
                    clamp=clamp,
                )
            )

    def forward(self, x):
        
        ##step 1: encode feature
        self.feature_extractor.eval()
        features = self.feature_extractor(x)
        
        ## step 2: norm features
        if self.Norm:
            features = [self.norms[i](feature) for i, feature in enumerate(features)]
        
        
        loss = 0
        outputs = []
        ## step3: fastflow
        for i, feature in enumerate(features):
            output, log_jac_dets = self.nf_flows[i](feature)
            ### loss = -logP(y) = -(logP(z) + jac)
            loss +=  (paddle.mean(
                            0.5 * paddle.sum(output**2, axis=(1, 2, 3)) -log_jac_dets
                        ) / (output.shape[1] * output.shape[2] * output.shape[3]))

            
            outputs.append(output)
        ### step4 :post process
        ret = {"loss": loss}
        if not self.training:
            anomaly_map_list = []
            for output in outputs:
                log_prob = -paddle.mean(output**2, axis=1, keepdim=True) * 0.5  ###logP(z)
                prob = paddle.exp(log_prob) ###P(z)
                ### get the final probability map and upsample it to the input image resolution using bilinear interpolation.
                a_map = F.interpolate(
                    1-prob,
                    size=[self.input_size, self.input_size],
                    mode="bilinear",
                    align_corners=True,
                )

                anomaly_map_list.append(a_map)

            anomaly_map_list = paddle.stack(anomaly_map_list, axis=-1)
            anomaly_map = paddle.mean(anomaly_map_list, axis=-1)

            ret["anomaly_map"] = anomaly_map
        return ret
