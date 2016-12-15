import sys
from random import randint

f = open('input', 'w')

if (len(sys.argv) < 2):
    print('python3 generate_input.py <a_nlines> <a_ncolums> <b_ncolums>')
    sys.exit()

a_nl = int(sys.argv[1])
a_nc = int(sys.argv[2])
b_nc = int(sys.argv[3])

f.write(str(a_nl)+'\n')
f.write(str(a_nc)+'\n')
f.write(str(a_nc)+'\n')
f.write(str(b_nc)+'\n')

for x in range(0, a_nl):
    for y in range(0, a_nc):
        if (y != 0):
            f.write(' ' + str(randint(0, 20) - 10))
        else:
            f.write(str(randint(0, 20) - 10))
    f.write('\n')

for x in range(0, a_nc):
    for y in range(0, b_nc):
        if (y != 0):
            f.write(' ' + str(randint(0, 20) - 10))
        else:
            f.write(str(randint(0, 20) - 10))
    f.write('\n')

f.close()
