import pyximport, numpy
pyximport.install(setup_args={"include_dirs":numpy.get_include()}, reload_support=True)

import linreg_regularization as lrr
lrr.run_bbox()

import linreg_test as lrt
lrt.run_bbox()
