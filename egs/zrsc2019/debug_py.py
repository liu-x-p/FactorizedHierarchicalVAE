id_map = {}
with open('debug.txt') as f:
	id_content = f.readlines()
for l in id_content:
	if not l.strip(): continue
	l = l.split(' ')
	id_map[int(l[1])] = l[0]
print id_map

