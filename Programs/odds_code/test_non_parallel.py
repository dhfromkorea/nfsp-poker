from time import time


l = range(int(1e7))


def f(n):
    return n+1


def main():
    t0 = time()
    for res in map(f, l):
        res += 1
    print(time() - t0)

main()
