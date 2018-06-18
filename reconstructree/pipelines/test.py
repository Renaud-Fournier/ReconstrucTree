import sys

for i in range(1,100):
    print("\r{}".format(i), end="")
    sys.stdout.flush()
print("\nend")
