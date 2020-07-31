'''

Authors: Shubham Ugare.

Copyright:
Copyright (c) 2018 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

'''

import os, sys
import onnxruntime
import onnx
from onnx import helper, numpy_helper
import unittest
from onnx import TensorProto
import numpy as np
import subprocess
from datetime import date
import time
import hashlib
from onnx import helper

# import smain
import seedot.main as main
import seedot.config as config
import seedot.compiler.ONNX.common as common

class TestNode(unittest.TestCase):

	def _get_rnd_float32(self, low=-1.0, high=1.0, shape=None, get_np_array=False):
		output = np.random.uniform(low, high, shape)
		if get_np_array:
			return output
		cnt = 1
		for val in shape: cnt*=val
		if shape == None:
			return np.float32(output)
		else:
			return output.astype(np.float32).reshape(cnt).tolist()

	def get_list_prod(self, list):
		prod = 1
		for val in list: prod*=val
		return prod
		

	# First read the ONNX file
	def get_onnx_output(self, model_path, input, intermediate_node=None):
		model = onnx.load(model_path)
		sess = onnxruntime.InferenceSession(model_path) 

		x = input
		x = x.astype(np.float32)

		input_name = model.graph.input[0].name

		if (intermediate_node != None):
			intermediate_layer_value_info = helper.ValueInfoProto()
			intermediate_layer_value_info.name = sys.argv[2]
			model.graph.output.extend([intermediate_layer_value_info])
			onnx.save(model, file_path + '_1')
			sess = onnxruntime.InferenceSession(file_path + '_1') 
			pred = sess.run([intermediate_layer_value_info.name], {input_name: x})
			return pred

		pred = sess.run(None, {input_name: x})
		return pred

	def check_result(self, graph, name, output_shape):
		current_milli_time = lambda: str(int(round(time.time() * 1000)))
		name = name + "_" + current_milli_time()
		model = onnx.helper.make_model(graph, producer_name='onnx-compiler-test')
		model_path = 'model/input.onnx'
		onnx.save(model, model_path)

		# need to create random input and output
		input_dims = common.proto_val_to_dimension_tuple(model.graph.input[0])
		inp = self._get_rnd_float32(shape=input_dims, get_np_array=True, low=0, high=256)	
		op = self.get_onnx_output(model_path, inp)		

		# print(type(inp), type(op))

		test = np.expand_dims(np.concatenate((np.asarray(op).flatten(), inp.flatten()), axis=0), axis=0)


		training_input = 'datasets/' + name + '_train.npy'
		testing_input = 'datasets/' + name + '_test.npy'

		np.save(training_input, test)
		np.save(testing_input, test)

		# call main and run the model
		# obj = main.Main(algo, version, target, trainingInput, testingInput, modelDir, sf, maximisingMetric, dataset, numOutputs, self.args.source)
		config.tempdir = "temp"
		if os.path.exists(config.tempdir):
			import shutil
			shutil.rmtree(config.tempdir)
			os.makedirs(config.tempdir)
		config.outdir = os.path.join(config.tempdir, "arduino")
		os.makedirs(config.outdir, exist_ok=True)	

		config.ddsEnabled = False
		config.vbwEnabled = False

		obj = main.Main(config.Algo.test, config.Version.fixed, config.Target.x86, training_input, testing_input, 
			'model', None, config.Metric.regressionLoss , None, self.get_list_prod(output_shape), config.Source.onnx)
		obj.run()			

	def test_relu(self):
		name = "relu"
		input_shape = [1, 3, 10, 10]
		output_shape = [1, 3, 10, 10]

		X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [self.get_list_prod(input_shape), 1])

		shape_param = helper.make_tensor('shape_param', TensorProto.INT64, [4], input_shape)

		reshape_node = helper.make_node("Reshape", ['X', 'shape_param'], ['state_in'])

		state_out  = helper.make_tensor_value_info('state_out',
		                                               TensorProto.FLOAT, output_shape)
		node_def = helper.make_node("Relu", ['state_in'], ['state_out'])
		graph = helper.make_graph(
		        [reshape_node, node_def],
		        name,
		        [X],
		        [state_out],
		        [shape_param]
		    )
		self.check_result(graph, name, output_shape)

	# TODO: padding -inf vs 0 in case of maxpool?
	def test_maxpool(self):
		name = "maxpool"
		input_shape = [1, 1, 6, 6]
		output_shape = [1, 1, 5, 5]

		X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [self.get_list_prod(input_shape), 1])

		shape_param = helper.make_tensor('shape_param', TensorProto.INT64, [4], input_shape)

		reshape_node = helper.make_node("Reshape", ['X', 'shape_param'], ['state_in'])

		state_out  = helper.make_tensor_value_info('state_out',
		                                               TensorProto.FLOAT, output_shape)
		node_def = helper.make_node("MaxPool", ['state_in'], ['state_out'], kernel_shape=[2, 2], strides=[1,1])
		graph = helper.make_graph(
		        [reshape_node, node_def],
		        name,
		        [X],
		        [state_out],
		        [shape_param]
		    )
		self.check_result(graph, name, output_shape)	

	# sometimes the sum overflows
	# TODO: replace int32 instead of MYINT for sum		
	def test_avgpool(self):
		name = "maxpool"
		input_shape = [1, 1, 6, 6]
		output_shape = [1, 1, 1, 1]

		X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [self.get_list_prod(input_shape), 1])

		shape_param = helper.make_tensor('shape_param', TensorProto.INT64, [4], input_shape)

		reshape_node = helper.make_node("Reshape", ['X', 'shape_param'], ['state_in'])

		state_out  = helper.make_tensor_value_info('state_out',
		                                               TensorProto.FLOAT, output_shape)
		node_def = helper.make_node("GlobalAveragePool", ['state_in'], ['state_out'])
		graph = helper.make_graph(
		        [reshape_node, node_def],
		        name,
		        [X],
		        [state_out],
		        [shape_param]
		    )
		self.check_result(graph, name, output_shape)		

	# TODO: with group!=1 
	def test_conv2d(self):
		name = "conv2d"
		input_shape = [1, 3, 224, 224]
		output_shape = [1, 64, 112, 112]

		X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [self.get_list_prod(input_shape), 1])

		shape_param = helper.make_tensor('shape_param', TensorProto.INT64, [4], input_shape)

		reshape_node = helper.make_node("Reshape", ['X', 'shape_param'], ['state_in'])

		state_out  = helper.make_tensor_value_info('state_out',
		                                               TensorProto.FLOAT, output_shape)
		node_def = helper.make_node("Conv", ['state_in', 'weight'], ['state_out'],
		                                pads=[3, 3, 3, 3], strides=[2, 2], kernel_shape=[7, 7], group=1)

		weight_shape = [64, 3, 7, 7]
		weight_val = self._get_rnd_float32(shape=weight_shape)

		weight = helper.make_tensor('weight', TensorProto.FLOAT, weight_shape, weight_val)

		graph = helper.make_graph(
		        [reshape_node, node_def],
		        name,
		        [X],
		        [state_out],
		        [shape_param, weight]
		    )
		self.check_result(graph, name, output_shape)	

	def test_bn(self):
		name = "batchnormalization"
		input_shape = [1, 2, 3, 3]
		output_shape = [1, 2, 3, 3]

		X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [self.get_list_prod(input_shape), 1])

		shape_param = helper.make_tensor('shape_param', TensorProto.INT64, [4], input_shape)

		reshape_node = helper.make_node("Reshape", ['X', 'shape_param'], ['state_in'])

		state_out  = helper.make_tensor_value_info('state_out',
		                                               TensorProto.FLOAT, output_shape)

		node_def = helper.make_node("BatchNormalization", ['state_in', 'weight', 'bias','mean','var'], ['state_out'],
	                                momentum=0.8999999761581421)

		weight_shape = [2]
		weight_val = self._get_rnd_float32(shape=weight_shape)
		weight = helper.make_tensor('weight', TensorProto.FLOAT, weight_shape, weight_val)

		bias_shape = [2]
		# bias_val = [1e-5,1e-5]
		bias_val = self._get_rnd_float32(shape=weight_shape)
		bias = helper.make_tensor('bias', TensorProto.FLOAT, bias_shape, bias_val)

		mean_shape = [2]
		# mean_val = [1e-5,1e-5]
		mean_val = self._get_rnd_float32(shape=weight_shape)
		mean = helper.make_tensor('mean', TensorProto.FLOAT, mean_shape, mean_val)


		var_shape = [2]
		# var_val = [1,1]
		var_val = self._get_rnd_float32(shape=weight_shape, low=0, high=1)
		var = helper.make_tensor('var', TensorProto.FLOAT, var_shape, var_val)

		graph = helper.make_graph(
		        [reshape_node, node_def],
		        name,
		        [X],
		        [state_out],
		        [shape_param, weight, bias, mean, var]
		    )
		self.check_result(graph, name, output_shape)	

if __name__ == '__main__':
	unittest.main()
