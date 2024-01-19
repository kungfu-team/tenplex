def main():
    old_index_file_path = '/data/megatron-lm/bert/test/indices.txt'
    new_index_file_path = './new_indices.txt'

    with open(old_index_file_path, 'r') as old_index_file:
        old_index_lines = old_index_file.readlines()

    old_indices = [int(l) for l in old_index_lines]

    with open(new_index_file_path, 'w') as new_index_file:
        for i in range(len(old_indices) - 1):
            new_index_file.write(f'{old_indices[i]} {old_index_lines[i+1]}')


if __name__ == '__main__':
    main()
