import os
import ctypes as c
import numpy as np
import random
import matplotlib.pyplot as plt


class Spline:
	def __init__(self, n_nodes, n_degree, verbose=False):
		self.n_nodes = n_nodes
		self.n_degree = n_degree
		self.lib = c.cdll.LoadLibrary('myLibSpline.so')
		self.lib.eval_point.argtypes = [c.c_int, c.c_int, c.c_double]
		self.lib.eval_point.restype = c.c_void_p
		self.lib.tangent_point.argtypes = [c.c_int, c.c_int, c.c_double]
		self.lib.tangent_point.restype = c.c_void_p
		self.lib.get_value_in_matrix.argtypes = [c.c_void_p, c.c_int]
		self.lib.get_value_in_matrix.restype = c.c_double
		self.verbose = verbose

	def get_Nodes_Eval_Weight(self, u):
		''' Get the weight of control points at u
		Return a numpy array: [n_nodes]
		'''
		w = []
		p = self.lib.eval_point(self.n_nodes, self.n_degree, u)
		for i in range(self.n_nodes):
			w.append(self.lib.get_value_in_matrix(p, i))
		if self.verbose:
			print(w)
		return np.asarray(w)

	def get_Nodes_Eval_Matrix(self, N):
		''' Get the weight matrix of a spline curve
		return a numpy array: [N,n_nodes]
		'''
		w = []
		for idx in range(N):
			w.append(self.get_Nodes_Eval_Weight(idx/float(N)))
		return np.asarray(w)

	def get_Nodes_Tangent_Weight(self, u):
		w = []
		p = self.lib.tangent_point(self.n_nodes, self.n_degree, u)
		for i in range(self.n_nodes):
			w.append(self.lib.get_value_in_matrix(p, i))
		if self.verbose:
			print(w)
		return np.asarray(p)

	def get_Nodes_Tangent_Matrix(self, N):
		w = []
		for idx in range(N):
			w.append(self.get_Nodes_Tangent_Weight(idx/float(N)))
		return np.asarray(w)



if __name__ == '__main__':
	spline = Spline(3, 3, verbose=True)
	w = spline.get_Nodes_Eval_Matrix(10)
	tan_w = spline.get_Nodes_Tangent_Matrix(10)
