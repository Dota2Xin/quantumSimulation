import numpy as np
import matplotlib.pyplot as plt

#we make a grid based around a point in the BZ given a cutoff.
def makeGrid(k, cutoff, reciprocalVecs):
    G1=reciprocalVecs[0]
    G2=reciprocalVecs[1]
    G3=reciprocalVecs[2]