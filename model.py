import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import config
from sklearn.neighbors import NearestNeighbors, KDTree
from nn_model import ResNet34

class MSE_Predict(nn.Module):
	def __init__(self):
		super(MSE_Predict, self).__init__()
		self.img_feature = ResNet34()
		self.hidden_size = 512
		self.gru = nn.GRU(input_size=256*4, hidden_size=self.hidden_size)
		self.predict = nn.Linear(self.hidden_size, 2) # Predict two position here.

	def forward(self, imgs, n_points):
		features = self.img_feature(imgs)
		batch_size = features.size(0)
		hidden = self.init_hidden(batch_size)
		features = features.view(1, batch_size, 256*4)
		outputs = Variable(torch.zeros(batch_size, n_points*config.n_dim)).cuda()

		for idx in range(n_points):
			output, hidden = self.gru(features, hidden)
			output = output.view(batch_size, self.hidden_size)
			# print(output.size())
			output = self.predict(F.relu(output))
			# print(output.size())
			outputs[:, idx*2:(idx+1)*2] = output
		return outputs

	def load_pretrain_weight(self, model_name):
		# Load a pretrained network
		pass

	def init_hidden(self, batch_size):
		result = Variable(torch.zeros(1, batch_size, self.hidden_size), requires_grad=False).cuda()


class Chamfer_Distance(nn.Module):
	def __init__(self, N):
		super(Chamfer_Distance, self).__init__()
		self.N = N

	def forward(self, pred, target):
		pred = pred.view(-1, self.N, config.n_dim)
		target = target.view(-1, self.N, config.n_dim)

		assert  pred.size(0) == target.size(0)
		batch_size = pred.size(0)

		###################################################
		# Use Nearest Neighbor
		###################################################
		target_numpy = target.data.cpu().numpy()
		pred_numpy = pred.data.cpu().numpy()
		# Trying to use CPU to compute NN.
		target_nn = Variable(torch.FloatTensor(batch_size, self.N, 2).cuda())
		for i in range(batch_size):
			nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(target_numpy[i, :, :])
			sample_index = torch.from_numpy(nbrs.kneighbors(pred_numpy[i, :, :])[1].reshape(self.N)).cuda()
			# comma segments each dimension.
			target_nn[i, :, :] = target[i, :, :].index_select(0, Variable(sample_index))
		# Get the l2 Euclidean distance matrix of two point clouds
		pred_to_target = torch.sum(torch.pow(target_nn - pred, 2))

		pred_nn = Variable(torch.FloatTensor(batch_size, self.N, 2).cuda())
		for i in range(batch_size):
			nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(pred_numpy[i, :, :])
			sample_index = torch.from_numpy(nbrs.kneighbors(target_numpy[i, :, :])[1].reshape(self.N)).cuda()
			# print(sample_index)
			# comma segments each dimension.
			pred_nn[i, :, :] = pred[i, :, :].index_select(0, Variable(sample_index))
		# Get the l2 Euclidean distance matrix of two point clouds
		target_to_pred = torch.sum(torch.pow(pred_nn - target, 2))

		return pred_to_target / (self.N * pred.data.size(0)), target_to_pred / (self.N * pred.data.size(0))

class Multi_Point_Model(nn.Module):
	def __init__(self):
		super(Multi_Point_Model, self).__init__()
		self.img_feature = ResNet34()
		self.linear1 = nn.Linear(256*4, 512)
		self.feature_size = 512
		self.hidden_size = 512
		self.gru = nn.GRU(input_size=self.feature_size, hidden_size=self.hidden_size)
		self.predict = nn.Linear(self.hidden_size, 4) # Predict position and existance probability.

	def forward(self, imgs, n_points):
		features = self.img_feature(imgs)
		features = self.linear1(features)
		batch_size = features.size(0)
		hidden = self.init_hidden(batch_size)
		features = features.view(1, batch_size, self.feature_size)
		point_pos_output = Variable(torch.zeros(batch_size, n_points, 2)).cuda()
		point_prob_output = Variable(torch.zeros(batch_size, n_points, 2)).cuda()

		for idx in range(n_points):
			output, hidden = self.gru(features, hidden)
			output = output.view(batch_size, self.hidden_size)
			# print(output.size())
			output = self.predict(F.relu(output))
			# print(output.size())
			point_pos_output[:, idx, :] = output[:,0:2]
			point_prob_output[:, idx, :] = output[:, 2:4]
		# print(point_prob_output.size(), point_pos_output.size())
		point_prob_output = point_prob_output.view(-1, 2)
		point_prob_output = F.softmax(point_prob_output)
		return point_pos_output, point_prob_output

	def load_pretrain_weight(self, model_name):
		# Load a pretrained network
		state_dict = self.state_dict()
		pretrained_model = torch.load(model_name)['net']
		pretrained_dict = pretrained_model.state_dict()
		pretrained_dict = dict(('img_feature.'+key, value) for (key, value) in pretrained_dict.items())
		# filter unuseful layer
		pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in state_dict}
		state_dict.update(pretrained_dict)
		self.load_state_dict(state_dict)
		print(pretrained_dict.keys())
		state_dict = self.state_dict()
		print(state_dict.keys())
		# print(state_dict)

	def init_hidden(self, batch_size):
		result = Variable(torch.zeros(1, batch_size, self.hidden_size), requires_grad=False).cuda()
		return result

class Multi_Point_Scheduled_Sampling_Model(nn.Module):
	def __init__(self):
		super(Multi_Point_Scheduled_Sampling_Model, self).__init__()
		self.img_feature = ResNet34()
		self.linear1 = nn.Linear(256*4, 512)
		self.feature_size = 512
		self.hidden_size = 512
		self.rnn_input_size = self.feature_size+4
		self.gru = nn.GRU(input_size=self.rnn_input_size, hidden_size=self.hidden_size)
		self.predict = nn.Linear(self.hidden_size, 4) # Predict position and existance probability.
		self.ss_prob = 0.1 # probability to choose from ground truth

	def forward(self, imgs, n_points, ground_truth, point_prob):
		ground_truth = ground_truth.view(-1, config.max_point, config.n_dim)
		img_features = self.img_feature(imgs)
		img_features = self.linear1(img_features)
		batch_size = img_features.size(0)
		img_features = img_features.view(1, batch_size, self.feature_size)

		point_pos_output = Variable(torch.zeros(batch_size, n_points, 2)).cuda()
		point_prob_output = Variable(torch.zeros(batch_size, n_points, 2)).cuda()

		for idx in range(n_points):
			if idx == 0:
				hidden, pred = self.init_hidden(batch_size)
				rnn_input_vec = torch.cat((img_features, pred), 2)
				# print(rnn_input_vec.size())
			else:
				pos_gt = ground_truth.data[:, idx - 1, :].clone()
				pos_gt = pos_gt.view(-1, 2)
				prob_gt = point_prob.data[:, idx - 1].clone()
				prob_gt = prob_gt.float().view(-1, 1)
				inv_prob_gt = 1 - prob_gt
				it = torch.cat((pos_gt, inv_prob_gt, prob_gt), 1)
				if self.ss_prob > 0.0: # Need to use network's predict result
					sample_prob = img_features.data.new(batch_size).uniform_(0,1)
					sample_mask = sample_prob < self.ss_prob
					if sample_mask.sum() != 0:
						sample_ind = sample_mask.nonzero().view(-1)
						# print(sample_ind)
						it.index_copy_(0, sample_ind, output.data.index_select(0, sample_ind))
				it = Variable(it, requires_grad=False)
				it = it.view(1, batch_size, self.rnn_input_size - self.feature_size)
				rnn_input_vec = torch.cat((img_features, it), 2)

			output, hidden = self.gru(rnn_input_vec, hidden)
			output = output.view(batch_size, self.hidden_size)
			# print(output.size())
			output = self.predict(F.relu(output))
			# print(output.size())
			point_pos_output[:, idx, :] = output[:, 0:2]
			point_prob_output[:, idx, :] = output[:, 2:4]
			# Replace RNN output Vector

		# print(point_prob_output.size(), point_pos_output.size())
		point_prob_output = point_prob_output.view(-1, 2)
		point_prob_output = F.softmax(point_prob_output)
		return point_pos_output, point_prob_output

	def load_pretrain_weight(self, model_name):
		# Load a pretrained network
		state_dict = self.state_dict()
		pretrained_model = torch.load(model_name)['net']
		pretrained_dict = pretrained_model.state_dict()
		pretrained_dict = dict(('img_feature.'+key, value) for (key, value) in pretrained_dict.items())
		# filter unuseful layer
		pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in state_dict}
		state_dict.update(pretrained_dict)
		self.load_state_dict(state_dict)
		print(pretrained_dict.keys())
		state_dict = self.state_dict()
		print(state_dict.keys())
		# print(state_dict)

	def init_hidden(self, batch_size):
		hidden = Variable(torch.zeros(1, batch_size, self.hidden_size)).cuda()
		pred = Variable(torch.zeros(1, batch_size, self.rnn_input_size-self.feature_size), requires_grad=False).cuda()
		return hidden, pred

class Multi_Point_Loss(nn.Module):
	def __init__(self):
		super(Multi_Point_Loss, self).__init__()
		self.clf_criterion = nn.CrossEntropyLoss(size_average = False)
	def forward(self, pos_pred, prob_pred, target_pos, point_mask, clf_point):
		'''

		:param pos_pred:
		:param prob_pred:
		:param target_pos:
		:param point_mask:
		:return:
		'''
		if config.debugging:
			print(pos_pred.cpu())
			print(prob_pred.cpu())
			print(target_pos.cpu())
			print(point_mask.cpu())
			print(clf_point.cpu())

		target_pos = target_pos.view(-1, config.max_point, config.n_dim)
		# print(type(target_pos))
		# print(target_pos.cpu())
		square_mse = torch.sum(torch.pow(pos_pred - target_pos, 2), dim=2)*point_mask
		if config.debugging: print(square_mse.cpu())
		mask_mse = torch.sum(square_mse) / (torch.sum(point_mask))
		# print(square_mse.size())

		#Classification loss
		clf_point = clf_point.view(-1, 1)
		if config.debugging: print(clf_point)
		lg_sftmx = -torch.log(prob_pred)
		if config.debugging: print(lg_sftmx)
		clf_loss = torch.gather(lg_sftmx, 1, clf_point) * point_mask.view(-1,1)
		if config.debugging: print(clf_loss)
		clf_loss = torch.sum(clf_loss)
		clf_loss = clf_loss / (torch.sum(point_mask))
		# clf_loss = lg_sftmx.select

		return mask_mse, clf_loss
