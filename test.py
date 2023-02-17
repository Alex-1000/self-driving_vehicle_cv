import numpy

a = numpy.arange(9).reshape((3, 3))
# a = numpy.append(a, [[9, 10, 11]], axis=0)
print(a)
b = numpy.array([a[0]])
b = numpy.append(b, [a[1]], axis=0)
print(b)
