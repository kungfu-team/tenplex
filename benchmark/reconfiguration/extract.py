import re


def extract_scale_up():
    num_hosts = 2

    log_file_name = "run_scale_up.log"
    with open(log_file_name, "r", encoding="utf-8") as log_file:
        lines = log_file.readlines()

    pattern = r"Finished docker run -d (\d+\.\d+)"
    start_timestamp = -1.0
    for line in lines:
        mat = re.match(pattern, line)
        if mat:
            start_timestamp = float(mat.group(1))
            break

    log_file_name = f"scale_up_{num_hosts-1}.log"
    with open(log_file_name, "r", encoding="utf-8") as log_file:
        lines = log_file.readlines()

    pattern = r"start iteration 0 at (\d+\.\d+)"
    iteration_timestamp = -1.0
    for line in lines:
        mat = re.match(pattern, line)
        if mat:
            iteration_timestamp = float(mat.group(1))
            break

    latency = iteration_timestamp - start_timestamp
    print(latency)


def extract_scale_down():
    num_hosts = 2

    log_file_name = "run_scale_down.log"
    with open(log_file_name, "r", encoding="utf-8") as log_file:
        lines = log_file.readlines()

    pattern = r"Finished docker stop (\d+\.\d+)"
    stop_timestamp = -1.0
    for line in lines:
        mat = re.match(pattern, line)
        if mat:
            stop_timestamp = float(mat.group(1))
            break

    log_file_name = "scale_up_0.log"
    with open(log_file_name, "r", encoding="utf-8") as log_file:
        lines = log_file.readlines()

    pattern = r"start iteration 0 at (\d+\.\d+)"
    iteration_timestamp = -1.0
    for line in lines:
        mat = re.match(pattern, line)
        if mat:
            iteration_timestamp = float(mat.group(1))
            break

    latency = iteration_timestamp - start_timestamp
    print(latency)


if __name__ == "__main__":
    extract_scale_down()
