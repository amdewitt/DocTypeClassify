import math

# Test File for converting 1d index to triangular matrix to make the CSV file easier to make
def __main__():
    length = 10000
    for i in range(0, __len__(length)):
        r, c = __indexToTriMatrixCoords__(i, length)
        print(str(i) + "(" + str(r) + ", " + str(c) + ")")
    print("-"*20 + "\n")

def __len__(length):
    return int(length * (length - 1) / 2)

# Turns a 1d index into a pair of distinct coordinates to fetch a pair of images from the CSV file
def __indexToTriMatrixCoords__(index, len):
    # default if index is not in range
    if(index < 0 or index >= __len__(len)):
        return 0, 0
    # Optimized formulas for getting triangular matrix coordinates (sourced from https://stackoverflow.com/questions/40950460/how-to-convert-triangular-matrix-indexes-in-to-row-column-coordinates)
    i = math.ceil(math.sqrt(2 * (index + 1) + 0.25) - 0.5)
    j = int((index + 1) - (i - 1) * i / 2 - 1)
    return i, j 

if __name__ == "__main__":
    __main__()