import torch
import numpy as np


def parse_slice(slice_str):
	no_space = slice_str.replace(' ', '')
	no_square_brackets = no_space.replace('[', '')
	no_square_brackets = no_square_brackets.replace(']', '')
	dimensions = no_square_brackets.split(',')
	
	slices = []
	for dim in dimensions:
		boarders = dim.split(':')
		parsed_boarders = []
		for boarder in boarders:
			if boarder != '':
				parsed_boarders.append(int(boarder))
			else:
				parsed_boarders.append(boarder)
		slices.append(parsed_boarders)
		
	return slices


def slice_tensor(tensor, slice_str):
	slices = parse_slice(slice_str)
	
	if len(tensor.shape) < len(slices):
		print(f'more slices {len(slices)} than tensor dimensions {len(tensor.shape)}')
	
	if len(slices) == 1:
		a = slices[0][0]
		b = slices[0][1]
		if a == '':
			tensor_slice = tensor[:b]
		elif b == '':
			tensor_slice = tensor[a:]
		else:
			tensor_slice = tensor[a:b]
	elif len(slices) == 2:
		a = slices[0][0]
		b = slices[0][1]
		c = slices[1][0]
		d = slices[1][1]
		if a == '':
			if c == '':
				tensor_slice = tensor[:b, :d]
			elif d == '':
				tensor_slice = tensor[:b, c:]
			else:
				tensor_slice = tensor[:b, c:d]
		elif b == '':
			if c == '':
				tensor_slice = tensor[a:, :d]
			elif d == '':
				tensor_slice = tensor[a:, c:]
			else:
				tensor_slice = tensor[a:, c:d]
		else:
			if c == '':
				tensor_slice = tensor[a:b, :d]
			elif d == '':
				tensor_slice = tensor[a:b, c:]
			else:
				tensor_slice = tensor[a:b, c:d]
	else:
		print(f'slices length {len(slices)} is too long')
		tensor_slice = None
			
	return tensor_slice
	
	
def test_tensor_slicing(tensor, slice_str):
	print(f'slice str {slice_str}')
	tensor_slice = slice_tensor(tensor, slice_str)
	if isinstance(tensor_slice, type(None)):
		print(f'tensor slice is None')
	else:
		print(f'tensor slice shape {tensor_slice.shape}')


def main():
	torch_tensor = torch.rand(10, 20, 30)
	np_tensor = np.random.rand(40, 50, 60)
	tensors = [torch_tensor, np_tensor]
	
	for tensor in tensors:
		print(f'tensor type {type(tensor)}')
		print(f'tensor shape {tensor.shape}')
		
		test_tensor_slicing(tensor, '[2:10, :12]')
		test_tensor_slicing(tensor, '[:10, 2:5]')
		test_tensor_slicing(tensor, '[0:10, 2:5, 0:4]')
		test_tensor_slicing(tensor, '[:10, :9]')
		test_tensor_slicing(tensor, '[5:, 4:]')
		test_tensor_slicing(tensor, '[5:8, 4:9]')
	
	
if __name__ == '__main__':
	main()