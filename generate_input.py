import sys
from random import randint

f = open('input', 'w')

if (len(sys.argv) < 2):
    print('python3 generate_input.py <nlines> <ncolums>')
    sys.exit()

nl = int(sys.argv[1])
nc = int(sys.argv[2])

f.write(str(nl)+'\n')
f.write(str(nc)+'\n')

for x in range(0, nl):
    for y in range(0, nc):
        if (y != 0):
            f.write(' ' + str(randint(0, 20) - 10))
        else:
            f.write(str(randint(0, 20) - 10))
    f.write('\n')

for x in range(0, nl):
    for y in range(0, nc):
        if (y != 0):
            f.write(' ' + str(randint(0, 20) - 10))
        else:
            f.write(str(randint(0, 20) - 10))
    f.write('\n')

f.close()
