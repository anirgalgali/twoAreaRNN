import numpy as np
import math
import os
from collections import defaultdict


class SingleAreaModule:

    def __init__(self, module_name, J_s_rec, J_t_rec, J_s_fwd = 0, J_s_fbk = 0,
                J_t_cross = 0, dt = 0.0001, bias = 0.334, tau_nmda = 0.06, tau_ampa = 0.002,
                sigma_noise = 0.009, a  = 270, b = 108, c = 0.154, gamma = 0.641):

        self.name = module_name
        self.J_s_rec = J_s_rec
        self.J_t_rec = J_t_rec
        self.J_s_fwd = J_s_fwd
        self.J_s_fbk = J_s_fbk
        self.J_t_cross = J_t_cross
        self.J_rec = self.__createRecurrentWeights()
        self.J_fwd = self.__createFeedForwardWeights()
        self.J_fbk = self.__createFeedbackWeights()
        self.J_inp = []
        self.dt = dt
        self.tau_nmda = tau_nmda
        self.tau_ampa = tau_ampa
        self.sigma_noise = sigma_noise
        self.bias = bias
        self.gamma = gamma
        self.a = a
        self.b = b
        self.c = c
        self.i_noise = None
        self.latents = None

    def __createRecurrentWeights(self):
        J_splust = 0.5*(self.J_s_rec + self.J_t_rec)
        J_tminuss = 0.5*(self.J_t_rec - self.J_s_rec)
        J_rec = np.array([[J_splust, J_tminuss],
                        [J_tminuss, J_splust]])

        return J_rec

    def __createFeedForwardWeights(self):
        J_splust = 0.5*(self.J_s_fwd + self.J_t_cross)
        J_tminuss = 0.5*(self.J_t_cross - self.J_s_fwd)
        J_fwd = np.array([[J_splust, J_tminuss],
                        [J_tminuss, J_splust]])

        return J_fwd

    def __createFeedbackWeights(self):
        J_splust = 0.5*(self.J_s_fbk + self.J_t_cross)
        J_tminuss = 0.5*(self.J_t_cross - self.J_s_fbk)
        J_fbk = np.array([[J_splust, J_tminuss],
                        [J_tminuss, J_splust]])

        return J_fbk

    def transfer_func(self,x):
        return (self.a*x - self.b)/(1.0 - np.exp(-self.c*(self.a*x - self.b)))


    def run_dynamics(self,latents, input, noise):

        # len(input) = number of "distinct" inputs that
        # come into the module (from all possible sources)

        weighted_input = np.zeros(np.shape(latents))
        for idx, _ in enumerate(self.J_inp):
            weighted_input = weighted_input + np.dot(self.J_inp[idx],input[idx])

        total_input = weighted_input + self.bias + noise

        s_dot = (-1/self.tau_nmda)*latents + self.gamma*(1 - latents)*self.transfer_func(total_input)

        return latents + s_dot*self.dt


    def generate_intrinsic_noise(self, noise_state):
        p, K = noise_state.shape
        i_dot = (1/self.tau_ampa)*(-1*noise_state +
                np.sqrt(self.tau_ampa*(self.sigma_noise**2))*
                np.random.randn(p, K))

        return noise_state + i_dot*self.dt;



class TwoAreaRNN:

    def __init__(self, connections):
        self.model = defaultdict(list)
        self.connection = connections
        self.add_connections(connections)

    def add_connections(self, connections):
        for node1,node2 in connections:
                node2.J_inp.append(node1.J_fwd)
                node1.J_inp.append(node2.J_fbk)
                self.model[node1].append(node2)
                self.model[node2].append(node1)

    def add_external_input(self,J_inp):
        nodes = list(self.model.keys())
        for i,node in enumerate(nodes):
            node_ = node
            node_.J_inp.append(J_inp[i])
            self.model[node_] = self.model.pop(node)

    def forward(self,inputs):

        p, K, T = inputs[0].shape
        nodes = list(self.model.keys())
        # initializing output arrays

        for i,node in enumerate(nodes):
            node_ = node
            node_.i_noise = np.zeros(inputs[0].shape)
            node_.latents = np.zeros(inputs[0].shape)
            self.model[node_] = self.model.pop(node)

        nodes = list(self.model.keys())
        # Setting intial conditions
        for i,node in enumerate(nodes):
            node_ = node
            noise_state = np.zeros((p,K))
            node_.latents[:,:,0] = np.sqrt(node.sigma_noise**2)*np.random.randn(p, K);
            node_.i_noise[:,:,0] = node.generate_intrinsic_noise(noise_state);
            self.model[node_] = self.model.pop(node)


        # Run the model
        for tt in range(1,T):
            nodes = list(self.model.keys())
            for i, node in enumerate(nodes):
                if tt == 2:
                    print(node.latents[:,100,tt])
                node_ = node
                inputs_ = list()
                for neighbor in self.model[node]:
                    inputs_.append(neighbor.latents[:,:,tt-1])

                inputs_.append(inputs[i][:,:,tt-1])
                assert len(inputs_) == len(node.J_inp)
                node_.latents[:,:,tt] = node.run_dynamics(node.latents[:,:,tt-1],inputs_, node.i_noise[:,:,tt-1])
                node_.i_noise[:,:,tt] = node.generate_intrinsic_noise(node.i_noise[:,:,tt-1])
                self.model[node_] = self.model.pop(node)
