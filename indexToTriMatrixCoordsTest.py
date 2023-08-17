import math

def __main__():
    length = 1000
    actualLength = int(length * (length - 1) / 2)
    for index in range(0, actualLength):
        i, j = __indexToTriMatrixCoords(index)
        print(f"{index} - ({i}, {j})")

def __indexToTriMatrixCoords(index):
    i = int(math.ceil(math.sqrt(2 * int(index + 1) + 0.25) - 0.5)) # Upper Index
    j = int(int(index) - (i - 1) * i / 2) # Lower Index
    return i, j

if __name__ == "__main__":
    __main__()