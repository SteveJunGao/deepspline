import numpy as np
from dataset_Reader import Spline_Dataset
from utils import progress_bar, init_params
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from model import ShallowCNN, Control_Point_Point_Cloud_Loss
import config
import random
from spline_interface import Spline
from spline_Utils_jiahui import plot_Spline_Curve_Control_Point, plot_Spline_Curve_Point_Cloud


score_state = torch.load(os.path.join(config.checkpoint_path, 'prediction/result_epoch_%d'%(
                        config.check_epoch)))

test_predict = score_state['test_predict']
test_label = score_state['test_label']
# test_input = score_state['test_input']
s = Spline(config.n_point, config.n_degree)
w = s.get_Nodes_Eval_Matrix(config.N)

if not os.path.exists(config.check_save_path):
    os.makedirs(config.check_save_path)

if not os.path.exists(config.check_save_input_path):
    os.makedirs(config.check_save_input_path)

for i in range(config.check_num):
    batch_idx = int(i/config.test_batch_size)
    idx = i%config.test_batch_size

    label = test_label[batch_idx][idx].float().view(5,2).numpy()*128
    pred = test_predict[batch_idx][idx].cpu().float().view(5,2).numpy()*128
    # print(target.shape)

    plot_Spline_Curve_Control_Point([label, pred], w, ['Target', 'Predict'],save=True,
                                    path=os.path.join(config.check_save_path, str(i)+'.jpg'))

    progress_bar(i, config.check_num)
