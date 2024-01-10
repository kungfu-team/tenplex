import re


def parse_hostfile(path):
    with open(path, 'r') as host_file:
        host_lines = host_file.readlines()

    hosts = []
    for line in host_lines:
        m = re.match(r'(.*) slots=(\d+)', line)
        ip = m.group(1)
        num_slots = int(m.group(2))
        hosts.append((ip, num_slots))

    return hosts

def get_rank_ip(hosts, rank):
    i = 0
    for ip, num_slots in hosts:
        next_host = i + num_slots
        if i <= rank < next_host:
            return ip
        i = next_host
    return None

def main():
    path = '/data/lg/ckpt/0/hostfile.txt'
    rank = 3

    hosts = parse_hostfile(path)

    print(f'hosts {hosts}')

    ip = get_rank_ip(hosts, rank)

    print(f'ip {ip}')



if __name__ == '__main__':
    main()
