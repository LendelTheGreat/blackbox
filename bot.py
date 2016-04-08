import pyximport, numpy
pyximport.install(setup_args={"include_dirs":numpy.get_include()}, reload_support=True)

import super_fast_bot as sfb 
# sfb.run_bbox()

import theano_memory_bot_test as thb
thb.run_bbox()
