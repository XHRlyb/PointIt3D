filename = 'A.txt'
with open(filename, 'r') as f:
	lines = f.readlines()
	for line in lines:
		print('cp -r ScanNet_with_eric_/{}_with_eric ScanNet_with_eric'.format(line.split()[1]))
