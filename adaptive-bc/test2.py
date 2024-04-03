import unittest
import numpy as np
from model import Model

class TestModel(unittest.TestCase):

    def setUp(self):
        kwparams = {
            'trial': 1,
            'max_steps': 1000,
            'N': 10,
            'p': 0.5,
            'tolerance': 0.01,
            'alpha': 0.1,
            'C': 0.5,
            'beta': 0.5,
            'M': 5,
            'K': 5,
            'full_time_series': True
        }
        self.model = Model(np.random.seed(123), **kwparams)
    
    def test_model_initialization(self):
        self.assertEqual(self.model.trial, 1)
        self.assertEqual(self.model.max_steps, 1000)
        self.assertEqual(self.model.N, 10)
        self.assertEqual(self.model.p, 0.5)
        self.assertEqual(self.model.tolerance, 0.01)
        self.assertEqual(self.model.alpha, 0.1)
        self.assertEqual(self.model.C, 0.5)
        self.assertEqual(self.model.beta, 0.5)
        self.assertEqual(self.model.M, 5)
        self.assertEqual(self.model.K, 5)
        self.assertTrue(self.model.full_time_series)

    def test_model_run(self):
        self.model.run()
        self.assertIsNotNone(self.model.convergence_time)
    
    def test_get_edges(self):
        self.model.run()
        edges = self.model.get_edges()
        self.assertIsInstance(edges, list)
        self.assertEqual(len(edges), len(self.model.edges))
    
    def test_get_network(self):
        self.model.run()
        network = self.model.get_network()
        self.assertIsInstance(network, nx.Graph)
        self.assertEqual(network.order(), self.model.N)
        self.assertEqual(network.size(), len(self.model.edges))
    
    def test_get_opinions(self):
        self.model.run()
        opinions = self.model.get_opinions()
        self.assertIsInstance(opinions, np.ndarray)
        self.assertEqual(opinions.shape[1], self.model.N)
    
    def test_save_model(self):
        self.model.run()
        self.model.save_model()
        # Check if file is created successfully
        
    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()