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

from knockpy import utilities
from knockpy import dgp
from knockpy import metro
import time


class TestMetroSample(unittest.TestCase):

	@pytest.mark.slow
	def test_independent_data(self):

		# Fake data
		np.random.seed(110)
		n = 30000
		p = 8
		X = np.random.randn(n,p)
		V = np.eye(p)
		Q = np.eye(p)
		order = np.arange(p)
		active_frontier = [[] for _ in range(p)]

		# likelihood
		mvn = stats.multivariate_normal(mean=np.zeros(p), cov=V)
		def mvn_likelihood(X):
			return mvn.logpdf(X)

		# Gamma
		gamma = 0.9999
		metro_sampler = metro_generic.MetropolizedKnockoffSampler(
			lf=mvn_likelihood,
			X=X,
			mu=np.zeros(p),
			V=V,
			order=order,
			active_frontier=active_frontier,
			S=np.eye(p),
			gamma=0.9999,
			metro_verbose=True,
		)

		# Output
		Xk = metro_sampler.sample_knockoffs()

	@pytest.mark.slow
	def test_ar1_sample(self):

		# Fake data
		np.random.seed(110)
		n = 1000
		p = 500
		dgprocess = dgp.DGP()
		X,_,_,Q,V = dgprocess.sample_data(method='AR1', rho=0.3, n=n, p=p)
		# _, S = knockpy.knockoffs.gaussian_knockoffs(
		# 	X=X, Sigma=V, method='mcv', return_S=True
		# )
		S = np.eye(p)

		# Graph structure + junction tree
		Q_graph = (np.abs(Q) > 1e-5)
		Q_graph = Q_graph - np.eye(p)
		undir_graph = nx.Graph(Q_graph)
		width, T = tree_processing.treewidth_decomp(undir_graph)
		order, active_frontier = tree_processing.get_ordering(T)

		# Metro sampler + likelihood
		mvn = stats.multivariate_normal(mean=np.zeros(p), cov=V)
		def mvn_likelihood(X):
			return mvn.logpdf(X)
		gamma = 0.9999
		metro_sampler = metro_generic.MetropolizedKnockoffSampler(
			lf=mvn_likelihood,
			X=X,
			mu=np.zeros(p),
			V=V,
			order=order,
			active_frontier=active_frontier,
			S=S,
			gamma=0.9999,
			metro_verbose=True
		)

		# Output knockoffs
		Xk = metro_sampler.sample_knockoffs()

		# Acceptance rate should be exactly one
		acc_rate = metro_sampler.final_acc_probs.mean()
		self.assertTrue(
			acc_rate - gamma > -1e-3, 
			msg = f'For AR1 gaussian design, metro has acc_rate={acc_rate} < gamma={gamma}'
		)

		# Check covariance matrix
		features = np.concatenate([X, Xk], axis=1)
		emp_corr_matrix = np.corrcoef(features.T)
		G = np.concatenate(
			[np.concatenate([V, V-S]),
			 np.concatenate([V-S, V]),], 
			axis=1
		)

		np.testing.assert_almost_equal(
			emp_corr_matrix, G, decimal=2,
			err_msg=f"For AR1 gaussian design, metro does not match theoretical matrix"
		)

	@pytest.mark.slow
	def test_dense_sample(self):

		# Fake data
		np.random.seed(110)
		n = 10000
		p = 4
		dgprocess = dgp.DGP()
		X,_,_,Q,V = dgprocess.sample_data(
			method='daibarber2016',
			rho=0.6, n=n, p=p,
			gamma=1, group_size=p
		)
		_, S = knockpy.knockoffs.gaussian_knockoffs(
			X=X, Sigma=V, method='mcv', return_S=True
		)

		# Network graph
		Q_graph = (np.abs(Q) > 1e-5)
		Q_graph = Q_graph - np.eye(p)
		undir_graph = nx.Graph(Q_graph)
		width, T = tree_processing.treewidth_decomp(undir_graph)
		order, active_frontier = tree_processing.get_ordering(T)

		# Metro sampler and likelihood
		mvn = stats.multivariate_normal(mean=np.zeros(p), cov=V)
		def mvn_likelihood(X):
			return mvn.logpdf(X)
		gamma = 0.99999
		metro_sampler = metro_generic.MetropolizedKnockoffSampler(
			lf=mvn_likelihood,
			X=X,
			mu=np.zeros(p),
			V=V,
			order=order,
			active_frontier=active_frontier,
			gamma=gamma,
			S=S,
			metro_verbose=True
		)

		# Output knockoffs
		Xk = metro_sampler.sample_knockoffs()

		# Acceptance rate should be exactly one
		acc_rate = metro_sampler.final_acc_probs.mean()
		self.assertTrue(
			acc_rate - gamma > -1e-3, 
			msg = f'For equi gaussian design, metro has acc_rate={acc_rate} < gamma={gamma}'
		)

		# Check covariance matrix
		features = np.concatenate([X, Xk], axis=1)
		emp_corr_matrix = np.corrcoef(features.T)
		G = np.concatenate(
			[np.concatenate([V, V-S]),
			 np.concatenate([V-S, V]),], 
			axis=1
		)

		# Show
		# import matplotlib.pyplot as plt
		# import seaborn as sns
		# fig, (ax0, ax1) = plt.subplots(ncols=2)
		# sns.heatmap(G, ax=ax0)
		# sns.heatmap(emp_corr_matrix, ax=ax1)
		# plt.show()

		np.testing.assert_almost_equal(
			emp_corr_matrix, G, decimal=2,
			err_msg=f"For equi gaussian design, metro does not match theoretical matrix"
		)


class TestARTK(unittest.TestCase):

	@pytest.mark.slow
	def test_tmarkov_likelihood(self):

		# Data
		np.random.seed(110)
		n = 100
		p = 10
		df_t = 5
		X1 = np.random.randn(n, p)
		X2 = np.random.randn(n, p)
		V = np.eye(p)
		Q = np.eye(p)

		# Scipy likelihood ratio for X, scale matrix
		inv_scale = np.sqrt(df_t / (df_t - 2))
		sp_like1 = stats.t.logpdf(inv_scale*X1, df=df_t).sum(axis=1)
		sp_like2 = stats.t.logpdf(inv_scale*X2, df=df_t).sum(axis=1)
		sp_ratio = sp_like1 - sp_like2

		# General likelihood
		rhos = np.zeros(p-1)
		ar1_like1 = metro_generic.t_markov_loglike(X1, rhos, df_t=df_t)
		ar1_like2 = metro_generic.t_markov_loglike(X2, rhos, df_t=df_t)
		ar1_ratio = ar1_like1 - ar1_like2

		self.assertTrue(
			np.abs(ar1_ratio - sp_ratio).sum() < 0.01,
			f"AR1 ratio {ar1_ratio} and scipy ratio {sp_ratio} disagree for independent t vars"
		)

		# Test again with df_t --> infinity, so it should be approx gaussian
		dgprocess = dgp.DGP()
		X1,_,_,Q,V = dgprocess.sample_data(
			n=n, p=p, method='AR1', a=3, b=1
		)
		X2 = np.random.randn(n, p)

		# Ratio using normals
		df_t = 100000
		mu = np.zeros(p)
		norm_like1 = stats.multivariate_normal(mean=mu, cov=V).logpdf(X1)
		norm_like2 = stats.multivariate_normal(mean=mu, cov=V).logpdf(X2)
		norm_ratio = norm_like1 - norm_like2

		# Ratios using T
		rhos = np.diag(V, 1)
		ar1_like1 = metro_generic.t_markov_loglike(X1, rhos, df_t=df_t)
		ar1_like2 = metro_generic.t_markov_loglike(X2, rhos, df_t=df_t)
		ar1_ratio = ar1_like1 - ar1_like2

		self.assertTrue(
			np.abs(ar1_ratio - norm_ratio).mean() < 0.01,
			f"AR1 ratio {ar1_ratio} and gaussian ratio {norm_ratio} disagree for corr. t vars, df={df_t}"
		)

		# Check consistency of tsampler class
		tsampler = metro_generic.ARTKSampler(
			X=X1,
			V=V,
			df_t=df_t,
		)
		new_ar1_like1 = tsampler.lf(tsampler.X)
		self.assertTrue(
			np.abs(ar1_like1 - new_ar1_like1).sum() < 0.01,
			f"AR1 loglike inconsistent between class ({new_ar1_like1}) and function ({ar1_ratio})"
		)
	@pytest.mark.slow
	def test_tmarkov_samples(self):

		# Test to make sure low df --> heavy tails
		# and therefore acceptances < 1
		np.random.seed(110)
		n = 1000
		p = 500
		df_t = 5
		dgprocess = dgp.DGP()
		X,_,_,Q,V = dgprocess.sample_data(
			n=n, p=p, method='AR1', rho=0.3, x_dist='ar1t'
		)
		S = np.eye(p)

		# Sample t 
		tsampler = metro_generic.ARTKSampler(
			X=X,
			V=V,
			df_t=df_t,
			S = S,
			metro_verbose=True
		)

		# Sample
		Xk = tsampler.sample_knockoffs(cache=None)


class TestBlockT(unittest.TestCase):

	@pytest.mark.slow
	def test_tmvn_log_likelihood(self):

		# Fake data
		np.random.seed(110)
		n = 10
		p = 10
		df_t = 100000

		# Test that the likelihood --> gaussian as df_t --> infinity
		dgprocess = dgp.DGP()
		X1,_,_,Q,V = dgprocess.sample_data(
			n=n, p=p, method='daibarber2016', gamma=0.3, rho=0.8, x_dist='blockt'
		)
		X2 = np.random.randn(n, p)


		# Ratio using normals
		mu = np.zeros(p)
		norm_like1 = stats.multivariate_normal(mean=mu, cov=V).logpdf(X1)
		norm_like2 = stats.multivariate_normal(mean=mu, cov=V).logpdf(X2)
		norm_ratio = norm_like1 - norm_like2

		# Ratios using T
		tmvn_like1 = metro_generic.t_mvn_loglike(X1, Q, df_t=df_t)
		tmvn_like2 = metro_generic.t_mvn_loglike(X2, Q, df_t=df_t)
		tmvn_ratio = tmvn_like1 - tmvn_like2
		self.assertTrue(
			np.abs(tmvn_ratio - norm_ratio).mean() < 0.01,
			f"T MVN ratio {tmvn_ratio} and gaussian ratio {norm_ratio} disagree for corr. t vars, df={df_t}"
		)

	@pytest.mark.slow
	def test_blockt_samples(self):

		# Test to make sure low df --> heavy tails
		# and therefore acceptances < 1
		np.random.seed(110)
		n = 10000
		p = 40
		df_t = 5
		dgprocess = dgp.DGP()
		X,_,_,Q,V = dgprocess.sample_data(
			n=n, 
			p=p,
			method='daibarber2016',
			rho=0.4,
			gamma=0,
			group_size=10,
			x_dist='blockt',
			df_t=df_t,
		)
		S = np.eye(p)

		# Sample t 
		tsampler = metro_generic.BlockTSampler(
			X=X,
			V=V,
			df_t=df_t,
			S=S,
			metro_verbose=True
		)

		# Sample
		Xk = tsampler.sample_knockoffs()

		# Check empirical means
		# Check empirical covariance matrix
		muk_hat = np.mean(Xk, axis=0)
		np.testing.assert_almost_equal(
			muk_hat, np.zeros(p), decimal=2,
			err_msg=f"For block T sampler, empirical mean of Xk does not match mean of X" 
		)

		# Check empirical covariance matrix
		Vk_hat = np.cov(Xk.T)
		np.testing.assert_almost_equal(
			V, Vk_hat, decimal=2,
			err_msg=f"For block T sampler, empirical covariance of Xk does not match cov of X" 
		)

		# Check that marginal fourth moments match
		X4th = np.mean(np.power(X, 4), axis=0)
		Xk4th = np.mean(np.power(Xk, 4), axis=0)
		np.testing.assert_almost_equal(
			X4th / 10, Xk4th / 10, decimal=1,
			err_msg=f"For block T sampler, fourth moment of Xk does not match theoretical fourth moment" 
		)

class TestIsing(unittest.TestCase):

	def test_large_ising_samples(self):

		# Test that sampling does not throw an error
		np.random.seed(110)
		n = 1000
		p = 625
		mu = np.zeros(p)
		dgprocess = dgp.DGP()
		X,_,_,undir_graph,_ = dgprocess.sample_data(
			n=n, 
			p=p,
			method='ising',
			x_dist='gibbs',
		)
		np.fill_diagonal(undir_graph, 1)

		# We load custom cov/q matrices for this
		file_directory = os.path.dirname(os.path.abspath(__file__))
		V = np.loadtxt(f'{parent_directory}/test/test_covs/vout{p}.txt')
		Q = np.loadtxt(f'{parent_directory}/test/test_covs/qout{p}.txt')
		max_nonedge = np.max(np.abs(Q[undir_graph == 0]))
		self.assertTrue(
			max_nonedge < 1e-5,
			f"Estimated precision for ising{p} has max_nonedge {max_nonedge} >= 1e-5"
		)

		# Initialize sampler
		metro_sampler = metro.IsingKnockoffSampler(
			X=X,
			undir_graph=undir_graph,
			mu=mu,
			V=V,
			Q=Q,
			max_width=5,
			method='equicorrelated',
			metro_verbose=True
		)

		# Sample and hope for no errors
		Xk = metro_sampler.sample_knockoffs()


if __name__ == '__main__':

	tester = TestIsing()#TestBlockT()#TestMetroSample()
	#tester.test_independent_data()
	#tester.test_dense_sample()
	tester.test_large_ising_samples()
	print(f"I am done!")