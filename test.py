import numpy as np
import os
import unittest
from model import *

class TestSingleAreaModule(unittest.TestCase):

    def test_rec_weights(self):

        module = SingleAreaModule('ppc',0.36,0.2839)
        self.assertEqual(module.J_rec[0,0], 0.5*(0.36 + 0.2839))
        self.assertEqual(module.J_rec[1,1], 0.5*(0.36 + 0.2839))
        self.assertEqual(module.J_rec[0,1], 0.5*(0.2839 - 0.36))
        self.assertEqual(module.J_rec[1,0], 0.5*(0.2839 - 0.36))

    def test_fwd_weights(self):

        module = SingleAreaModule('ppc',0.36,0.2839,0.08)
        self.assertEqual(module.J_fwd[0,0], 0.5*(0.08 + 0))
        self.assertEqual(module.J_fwd[1,1], 0.5*(0.08 + 0))
        self.assertEqual(module.J_fwd[0,1], 0.5*(0 - 0.08))
        self.assertEqual(module.J_fwd[1,0], 0.5*(0 - 0.08))

    def test_fbk_weights(self):

        module = SingleAreaModule('ppc',0.36,0.2839,0.08,0.08)
        self.assertEqual(module.J_fbk[0,0], 0.5*(0.08 + 0))
        self.assertEqual(module.J_fbk[1,1], 0.5*(0.08 + 0))
        self.assertEqual(module.J_fbk[0,1], 0.5*(0 - 0.08))
        self.assertEqual(module.J_fbk[1,0], 0.5*(0 - 0.08))

    def test_intrisic_noise(self):

        module = SingleAreaModule('ppc',0.36,0.2839,0.08,0.08)
        self.assertEqual(module.i_noise.shape[0],2)
        self.assertEqual(module.i_noise.shape[1],10000)
        self.assertEqual(module.i_noise.shape[2],999)
        self.assertTrue(abs(np.std(np.squeeze(module.i_noise[0,:,100])) - module.sigma_noise) <= 1e-3)
        self.assertTrue(abs(np.std(np.squeeze(module.i_noise[1,:,100])) - module.sigma_noise) <= 1e-3)

    def test_initial_conditions(self):

        module = SingleAreaModule('ppc',0.36,0.2839,0.08,0.08)
        self.assertEqual(module.ics.shape[0],2)
        self.assertEqual(module.ics.shape[1],10000)
        self.assertTrue(abs(np.mean(np.squeeze(module.ics[0,:])) - 0) <= 1e-3)
        self.assertTrue(abs(np.mean(np.squeeze(module.ics[1,:])) - 0) <= 1e-3)
        self.assertTrue(abs(np.std(np.squeeze(module.ics[0,:])) - module.sigma_noise) <= 1e-3)
        self.assertTrue(abs(np.std(np.squeeze(module.ics[1,:])) - module.sigma_noise) <= 1e-3)

if __name__ == '__main__':
    unittest.main()
