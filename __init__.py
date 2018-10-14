def read_data(filename):
    file = open(filename,'r')

    data_file = [[float(val) for val in line.split()] for line in file if len(line.strip()) > 0]

    file.close()
    return data_file
