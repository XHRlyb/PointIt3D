filename = 'R.txt'
with open(filename, 'r') as f:
	lines = f.readlines()
	for line in lines:
		print('python3 make_data.py --scanID {} --data_dir ../ScanNet/scans --seed 55 --visualize'.format(line.split()[1]))
