# Used for dataset
n_point = 5
max_point = 6
min_point = 4
N = 1024
max_line = 3
min_line = 1
n_degree = 3
n_dim = 2
n_data = 500000
n_per_patch = 100
dataset_path = './Spline_Matrix_Dataset'
gen_verbose = False

# Used for dataset reader
reader_verbose = False

# Used for training
manual_seed = 1
train_part = 0.7
train_batch_size = 32
continue_train = False
continue_model_name = './multi_spline_cc_pretrain_VGG_more_data_attention_large_map_continue/chkt_15'
lr = 1e-6
weight_decay = 1e-6
max_epoch = 100
fix_pretrained = False
checkpoint_path = './multi_spline_cc_pretrain_VGG_more_data_attention_large_map_continue'
log_file_path = './log/'
log_file_name = 'multi_spline_cc_pretrain_VGG_more_data_attention_large_map_continue'
debugging = False
clf_weight = 0.1
fix_img_feature = False

# Used for testing & visualiztion
test_batch_size = 50
check_epoch = 33
check_save_path = checkpoint_path + '/plot_predict_result'
check_save_input_path = checkpoint_path + '/plot_input'

# Used for code checking
check_num = 1000
save_check_path = './plot_check'
