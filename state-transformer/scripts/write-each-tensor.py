import torch
import os


def save_obj(obj, path):
	base_dir = './data/each'
	file_name = base_dir + path
	file_name = file_name + '.pt'
	dir_name = os.path.dirname(file_name)
	os.makedirs(dir_name, exist_ok=True)
	print(f'file name {file_name}')
	torch.save(obj, file_name)


def walk_pt(obj, path):
	if isinstance(obj, dict):
		for k, v in sorted(obj.items()):
			walk_pt(v, path + '/' + k)
	elif type(obj) is list:
		for i, x in enumerate(obj):
			walk_pt(x, path + '/' + str(i))
	elif type(obj) is tuple:
		for i, x in enumerate(obj):
			walk_pt(x, path + '/' + str(i))
	else:
		save_obj(obj, path)

def read_pt_file(file_name):
	ckpt_dict = torch.load(file_name, map_location=torch.device('cpu'))
	return ckpt_dict


def main():
	# file_name = './data/ckpt/global_step5/mp_rank_00_model_states.pt'
	file_name = './data/ckpt/global_step5/zero_pp_rank_0_mp_rank_00_optim_states.pt'
	ckpt_dict = read_pt_file(file_name)
	walk_pt(ckpt_dict, '')


if __name__ == '__main__':
	main()