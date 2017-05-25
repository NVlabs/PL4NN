# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import scipy.signal
import caffe

class L2(caffe.Layer):
    "A loss layer that calculates Mean-Squared-Error loss"
    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.diff[...] = bottom[0].data - bottom[1].data
        top[0].data[...] = np.sum(self.diff**2) / bottom[0].count / 2.

    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff / bottom[i].count

class L1(caffe.Layer):
    "A loss layer that calculates Mean-Absolute-Error loss"
    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.diff[...] = bottom[0].data - bottom[1].data
        top[0].data[...] = np.sum(np.abs(self.diff)) / bottom[0].count

    def backward(self, top, propagate_down, bottom):

	    # get the sign
        diff_sign = np.sign(self.diff)

        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * diff_sign / bottom[i].count

class SSIM(caffe.Layer):
    "A loss layer that calculates (1-SSIM) loss. Assuming bottom[0] is output data and bottom[1] is label, meaning no back-propagation to bottom[1]."

    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.C1 = params.get('C1', 0.01) ** 2
        self.C2 = params.get('C2', 0.03) ** 2
        self.sigma = params.get('sigma', 5.)

        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

        if (bottom[0].width%2) != 1 or (bottom[1].width%2) != 1 :
            raise Exception("Odd patch size preferred")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # loss output is scalar
        top[0].reshape(1)

		# initialize the gaussian filter based on the bottom size
        width = bottom[0].width
        self.w = np.exp(-1.*np.arange(-(width/2), width/2+1)**2/(2*self.sigma**2))
        self.w = np.outer(self.w, self.w.reshape((width, 1)))	# extend to 2D
        self.w = self.w/np.sum(self.w)							# normailization
        self.w = np.reshape(self.w, (1, 1, width, width)) 		# reshape to 4D
        self.w = np.tile(self.w, (bottom[0].num, 3, 1, 1))

    def forward(self, bottom, top):
        self.mux = np.sum(self.w * bottom[0].data, axis=(2,3), keepdims=True)
        self.muy = np.sum(self.w * bottom[1].data, axis=(2,3), keepdims=True)
        self.sigmax2 = np.sum(self.w * bottom[0].data ** 2, axis=(2,3), keepdims=True) - self.mux **2
        self.sigmay2 = np.sum(self.w * bottom[1].data ** 2, axis=(2,3), keepdims=True) - self.muy **2
        self.sigmaxy = np.sum(self.w * bottom[0].data * bottom[1].data, axis=(2,3), keepdims=True) - self.mux * self.muy
        self.l = (2 * self.mux * self.muy + self.C1)/(self.mux ** 2 + self.muy **2 + self.C1)
        self.cs = (2 * self.sigmaxy + self.C2)/(self.sigmax2 + self.sigmay2 + self.C2)

        top[0].data[...] = 1 - np.sum(self.l * self.cs)/(bottom[0].channels * bottom[0].num)

    def backward(self, top, propagate_down, bottom):
        self.dl = 2 * self.w * (self.muy - self.mux * self.l) / (self.mux**2 + self.muy**2 + self.C1)
        self.dcs = 2 / (self.sigmax2 + self.sigmay2 + self.C2) * self.w * ((bottom[1].data - self.muy) - self.cs * (bottom[0].data - self.mux))

        bottom[0].diff[...] = -(self.dl * self.cs + self.l * self.dcs)/(bottom[0].channels * bottom[0].num)	    # negative sign due to -dSSIM
        bottom[1].diff[...] = 0

class MSSSIM(caffe.Layer):
    "A loss layer that calculates (1-MSSSIM) loss. Assuming bottom[0] is output data and bottom[1] is label, meaning no back-propagation to bottom[1]."

    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.C1 = params.get('C1', 0.01) ** 2
        self.C2 = params.get('C2', 0.03) ** 2
        self.sigma = params.get('sigma', (0.5, 1., 2., 4., 8.))

        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

        if (bottom[0].width%2) != 1 or (bottom[1].width%2) != 1 :
            raise Exception("Odd patch size preferred")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # loss output is scalar
        top[0].reshape(1)

		# initialize the size to 5D
        num_scale = len(self.sigma)
        width = bottom[0].width
        self.w = np.empty((num_scale, bottom[0].num, 3, width, width))
        self.mux = np.empty((num_scale, bottom[0].num, 3, 1, 1))
        self.muy = np.empty((num_scale, bottom[0].num, 3, 1, 1))
        self.sigmax2 = np.empty((num_scale, bottom[0].num, 3, 1, 1))
        self.sigmay2 = np.empty((num_scale, bottom[0].num, 3, 1, 1))
        self.sigmaxy = np.empty((num_scale, bottom[0].num, 3, 1, 1))
        self.l = np.empty((num_scale, bottom[0].num, 3, 1, 1))
        self.cs = np.empty((num_scale, bottom[0].num, 3, 1, 1))

		# initialize the gaussian filters based on the bottom size
        for i in range(num_scale):
            weights = np.exp(-1.*np.arange(-(width/2), width/2+1)**2/(2*self.sigma[i]**2))
            weights = np.outer(weights, weights.reshape((width, 1)))	# extend to 2D
            weights = weights/np.sum(weights)							# normailization
            weights = np.reshape(weights, (1, 1, width, width)) 		# reshape to 4D
            weights = np.tile(weights, (bottom[0].num, 3, 1, 1))
            self.w[i,:,:,:,:] = weights

    def forward(self, bottom, top):

		# tile the bottom blob to 5D
		self.bottom0data = np.tile(bottom[0].data, (len(self.sigma), 1, 1, 1, 1))
		self.bottom1data = np.tile(bottom[1].data, (len(self.sigma), 1, 1, 1, 1))

		self.mux = np.sum(self.w * self.bottom0data, axis=(3, 4), keepdims=True)
		self.muy = np.sum(self.w * self.bottom1data, axis=(3, 4), keepdims=True)
		self.sigmax2 = np.sum(self.w * self.bottom0data ** 2, axis=(3, 4), keepdims=True) - self.mux **2
		self.sigmay2 = np.sum(self.w * self.bottom1data ** 2, axis=(3, 4), keepdims=True) - self.muy **2
		self.sigmaxy = np.sum(self.w * self.bottom0data * self.bottom1data, axis=(3, 4), keepdims=True) - self.mux * self.muy
		self.l = (2 * self.mux * self.muy + self.C1)/(self.mux ** 2 + self.muy **2 + self.C1)
		self.cs = (2 * self.sigmaxy + self.C2)/(self.sigmax2 + self.sigmay2 + self.C2)

		self.Pcs = np.prod(self.cs, axis=0)
		top[0].data[...] = 1 - np.sum(self.l[-1, :, :, :, :] * self.Pcs)/(bottom[0].channels*bottom[0].num)

    def backward(self, top, propagate_down, bottom):
        self.dl = 2 * self.w * (self.muy - self.mux * self.l) / (self.mux**2 + self.muy**2 + self.C1)
        self.dcs = 2 / (self.sigmax2 + self.sigmay2 + self.C2) * self.w * ((self.bottom1data - self.muy) - self.cs * (self.bottom0data - self.mux))

        dMSSSIM = self.dl[-1, :, :, :, :]
        for i in range(len(self.sigma)):
            dMSSSIM += self.dcs[i, :, :, :, :] / self.cs[i, :, :, :, :] * self.l[-1, :, :, :, :]
        dMSSSIM *= self.Pcs

        bottom[0].diff[...] = -dMSSSIM/(bottom[0].channels*bottom[0].num)	# negative sign due to -dSSIM
        bottom[1].diff[...] = 0

class MSSSIML1(caffe.Layer):
    "A loss layer that calculates alpha*(1-MSSSIM)+(1-alpha)*L1 loss. Assuming bottom[0] is output data and bottom[1] is label, meaning no back-propagation to bottom[1]."

    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.C1 = params.get('C1', 0.01) ** 2
        self.C2 = params.get('C2', 0.03) ** 2
        self.sigma = params.get('sigma', (0.5, 1., 2., 4., 8.))
        self.alpha = params.get('alpha', 0.025)

        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

        if (bottom[0].width%2) != 1 or (bottom[1].width%2) != 1 :
            raise Exception("Odd patch size preferred")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # loss output is scalar
        top[0].reshape(1)

		# initialize the size to 5D
        num_scale = len(self.sigma)
        self.width = bottom[0].width
        self.channels = bottom[0].channels
        self.batch = bottom[0].num

        self.w = np.empty((num_scale, self.batch, self.channels, self.width, self.width))
        self.mux = np.empty((num_scale, self.batch, self.channels, 1, 1))
        self.muy = np.empty((num_scale, self.batch, self.channels, 1, 1))
        self.sigmax2 = np.empty((num_scale, self.batch, self.channels, 1, 1))
        self.sigmay2 = np.empty((num_scale, self.batch, self.channels, 1, 1))
        self.sigmaxy = np.empty((num_scale, self.batch, self.channels, 1, 1))
        self.l = np.empty((num_scale, self.batch, self.channels, 1, 1))
        self.cs = np.empty((num_scale, self.batch, self.channels, 1, 1))

		# initialize the gaussian filters based on the bottom size
        for i in range(num_scale):
            gaussian = np.exp(-1.*np.arange(-(self.width/2), self.width/2+1)**2/(2*self.sigma[i]**2))
            gaussian = np.outer(gaussian, gaussian.reshape((self.width, 1)))	# extend to 2D
            gaussian = gaussian/np.sum(gaussian)								# normailization
            gaussian = np.reshape(gaussian, (1, 1, self.width, self.width)) 	# reshape to 4D
            gaussian = np.tile(gaussian, (self.batch, self.channels, 1, 1))
            self.w[i,:,:,:,:] = gaussian

    def forward(self, bottom, top):

		# tile the bottom blob to 5D
		self.bottom0data = np.tile(bottom[0].data, (len(self.sigma), 1, 1, 1, 1))
		self.bottom1data = np.tile(bottom[1].data, (len(self.sigma), 1, 1, 1, 1))

		self.mux = np.sum(self.w * self.bottom0data, axis=(3, 4), keepdims=True)
		self.muy = np.sum(self.w * self.bottom1data, axis=(3, 4), keepdims=True)
		self.sigmax2 = np.sum(self.w * self.bottom0data ** 2, axis=(3, 4), keepdims=True) - self.mux **2
		self.sigmay2 = np.sum(self.w * self.bottom1data ** 2, axis=(3, 4), keepdims=True) - self.muy **2
		self.sigmaxy = np.sum(self.w * self.bottom0data * self.bottom1data, axis=(3, 4), keepdims=True) - self.mux * self.muy
		self.l = (2 * self.mux * self.muy + self.C1)/(self.mux ** 2 + self.muy **2 + self.C1)
		self.cs = (2 * self.sigmaxy + self.C2)/(self.sigmax2 + self.sigmay2 + self.C2)
		self.Pcs = np.prod(self.cs, axis=0)

		loss_MSSSIM = 1 - np.sum(self.l[-1, :, :, :, :] * self.Pcs)/(self.batch * self.channels)
		self.diff = bottom[0].data - bottom[1].data
		loss_L1 = np.sum(np.abs(self.diff) * self.w[-1, :, :, :, :]) / (self.batch * self.channels)  # L1 loss weighted by Gaussian

		top[0].data[...] = self.alpha * loss_MSSSIM + (1-self.alpha) * loss_L1

    def backward(self, top, propagate_down, bottom):
        self.dl = 2 * self.w * (self.muy - self.mux * self.l) / (self.mux**2 + self.muy**2 + self.C1)
        self.dcs = 2 / (self.sigmax2 + self.sigmay2 + self.C2) * self.w * ((self.bottom1data - self.muy) - self.cs * (self.bottom0data - self.mux))

        dMSSSIM = self.dl[-1, :, :, :, :]
        for i in range(len(self.sigma)):
            dMSSSIM += self.dcs[i, :, :, :, :] / self.cs[i, :, :, :, :] * self.l[-1, :, :, :, :]
        dMSSSIM *= self.Pcs

        diff_L1 = np.sign(self.diff) * self.w[-1, :, :, :, :] / (self.batch * self.channels)		# L1 gradient weighted by Gaussian
        diff_MSSSIM = -dMSSSIM/(self.batch * self.channels)

        bottom[0].diff[...] = self.alpha * diff_MSSSIM + (1-self.alpha) * diff_L1
        bottom[1].diff[...] = 0

class MSSSIML2(caffe.Layer):
    "A loss layer that calculates alpha*(1-MSSSIM)+(1-alpha)*L2 loss. Assuming bottom[0] is output data and bottom[1] is label, meaning no back-propagation to bottom[1]."

    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.C1 = params.get('C1', 0.01) ** 2
        self.C2 = params.get('C2', 0.03) ** 2
        self.sigma = params.get('sigma', (0.5, 1., 2., 4., 8.))
        self.alpha = params.get('alpha', 0.1)

        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

        if (bottom[0].width%2) != 1 or (bottom[1].width%2) != 1 :
            raise Exception("Odd patch size preferred")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # loss output is scalar
        top[0].reshape(1)

        # initialize the size to 5D
        num_scale = len(self.sigma)
        self.width = bottom[0].width
        self.channels = bottom[0].channels
        self.batch = bottom[0].num

        self.w = np.empty((num_scale, self.batch, self.channels, self.width, self.width))
        self.mux = np.empty((num_scale, self.batch, self.channels, 1, 1))
        self.muy = np.empty((num_scale, self.batch, self.channels, 1, 1))
        self.sigmax2 = np.empty((num_scale, self.batch, self.channels, 1, 1))
        self.sigmay2 = np.empty((num_scale, self.batch, self.channels, 1, 1))
        self.sigmaxy = np.empty((num_scale, self.batch, self.channels, 1, 1))
        self.l = np.empty((num_scale, self.batch, self.channels, 1, 1))
        self.cs = np.empty((num_scale, self.batch, self.channels, 1, 1))

		# initialize the gaussian filters based on the bottom size
        for i in range(num_scale):
            gaussian = np.exp(-1.*np.arange(-(self.width/2), self.width/2+1)**2/(2*self.sigma[i]**2))
            gaussian = np.outer(gaussian, gaussian.reshape((self.width, 1)))	# extend to 2D
            gaussian = gaussian/np.sum(gaussian)								# normailization
            gaussian = np.reshape(gaussian, (1, 1, self.width, self.width)) 	# reshape to 4D
            gaussian = np.tile(gaussian, (self.batch, self.channels, 1, 1))
            self.w[i,:,:,:,:] = gaussian

    def forward(self, bottom, top):

		# tile the bottom blob to 5D
		self.bottom0data = np.tile(bottom[0].data, (len(self.sigma), 1, 1, 1, 1))
		self.bottom1data = np.tile(bottom[1].data, (len(self.sigma), 1, 1, 1, 1))

		self.mux = np.sum(self.w * self.bottom0data, axis=(3, 4), keepdims=True)
		self.muy = np.sum(self.w * self.bottom1data, axis=(3, 4), keepdims=True)
		self.sigmax2 = np.sum(self.w * self.bottom0data ** 2, axis=(3, 4), keepdims=True) - self.mux **2
		self.sigmay2 = np.sum(self.w * self.bottom1data ** 2, axis=(3, 4), keepdims=True) - self.muy **2
		self.sigmaxy = np.sum(self.w * self.bottom0data * self.bottom1data, axis=(3, 4), keepdims=True) - self.mux * self.muy
		self.l = (2 * self.mux * self.muy + self.C1)/(self.mux ** 2 + self.muy **2 + self.C1)
		self.cs = (2 * self.sigmaxy + self.C2)/(self.sigmax2 + self.sigmay2 + self.C2)
		self.Pcs = np.prod(self.cs, axis=0)

		loss_MSSSIM = 1 - np.sum(self.l[-1, :, :, :, :] * self.Pcs)/(self.batch * self.channels)
		self.diff = bottom[0].data - bottom[1].data
		loss_L2 = np.sum((self.diff**2) * self.w[-1, :, :, :, :]) / (self.batch * self.channels) / 2.  # L2 loss weighted by Gaussian

		top[0].data[...] = self.alpha * loss_MSSSIM + (1-self.alpha) * loss_L2

    def backward(self, top, propagate_down, bottom):
        self.dl = 2 * self.w * (self.muy - self.mux * self.l) / (self.mux**2 + self.muy**2 + self.C1)
        self.dcs = 2 / (self.sigmax2 + self.sigmay2 + self.C2) * self.w * ((self.bottom1data - self.muy) - self.cs * (self.bottom0data - self.mux))

        dMSSSIM = self.dl[-1, :, :, :, :]
        for i in range(len(self.sigma)):
            dMSSSIM += self.dcs[i, :, :, :, :] / self.cs[i, :, :, :, :] * self.l[-1, :, :, :, :]
        dMSSSIM *= self.Pcs

        diff_L2 = self.diff * self.w[-1, :, :, :, :] / (self.batch * self.channels)		# L2 gradient weighted by Gaussian
        diff_MSSSIM = -dMSSSIM/(self.batch * self.channels)

        bottom[0].diff[...] = self.alpha * diff_MSSSIM + (1-self.alpha) * diff_L2
        bottom[1].diff[...] = 0
