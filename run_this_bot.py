# This is what I used for cython
# import pyximport; pyximport.install()
# import setuptools
# import distutils

# import linreg_update as bot
# bot.run_bbox()

# import linreg_squared_test as test_bot
# test_bot.run_bbox()

import linreg_squared_test_numpy as test_bot
test_bot.run_bbox()

# 'coefs_squared.txt'
# previous train score: 3091
# previous test score : 2491

# 'coefs_squared_more.txt'
# train: 3172
# test : 2479