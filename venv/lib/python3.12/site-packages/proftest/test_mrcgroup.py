import os
import sys

# Add path to allow import of code
file_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.split(file_directory)[0]
sys.path.insert(0, os.path.abspath(parent_directory))

# Import the actual stuff
import knockpy

# Other things to import
import pytest
import numpy as np
import networkx as nx
from scipy import stats
import unittest

import time


class MRCGroup(unittest.TestCase):

	def test_mrcgroup(self):

		np.random.seed(110)
		p = 300

		# Sample data
		dgprocess = knockpy.dgp.DGP()
		dgprocess.sample_data(method='ar1', p=p)
		Sigma = dgprocess.Sigma
		groups = np.around(np.arange(1, p+1)/2)

		# Compute
		# Sgroup = knockpy.mrc.solve_mvr(Sigma, groups, verbose=True)


	

if __name__ == '__main__':

	time0 = time.time()
	tester = MRCGroup()
	tester.test_mrcgroup()
	print(f"I am done after {time.time() - time0}!")