# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable


class RNN(nn.Module):
    def __init__(self, data_size, device, args):
        super(RNN, self).__init__()
        self.init_lin_h = nn.Linear(args.noise_dim, args.latent_dim)
        self.init_lin_c = nn.Linear(args.noise_dim, args.latent_dim)
        self.init_input = nn.Linear(args.noise_dim, args.latent_dim)

        self.rnn = nn.LSTM(args.latent_dim, args.latent_dim, args.num_rnn_layer)

        # Transforming LSTM output to vector shape
        self.lin_transform_down = nn.Sequential(
            nn.Linear(args.latent_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(
                args.hidden_dim, data_size * 128 + 128 * 128 + 128 + 128 + 128 + 1
            ),
        )
        # Transforming vector to LSTM input shape
        self.lin_transform_up = nn.Sequential(
            nn.Linear(
                data_size * 128 + 128 * 128 + 128 + 128 + 128 + 1, args.hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.latent_dim),
        )

        self.num_rnn_layer = args.num_rnn_layer
        self.data_size = data_size
        self.device = device

    def nn_construction(self, E):
        m_1 = E[:, : self.data_size * 128]
        m_2 = E[:, self.data_size * 128 : (self.data_size * 128 + 128 * 128)]
        m_3 = E[
            :,
            (self.data_size * 128 + 128 * 128) : (
                self.data_size * 128 + 128 * 128 + 128
            ),
        ]
        b_1 = E[
            :,
            (self.data_size * 128 + 128 * 128 + 128) : (
                self.data_size * 128 + 128 * 128 + 128 + 128
            ),
        ]
        b_2 = E[
            :,
            (self.data_size * 128 + 128 * 128 + 128 + 128) : (
                self.data_size * 128 + 128 * 128 + 128 + 128 + 128
            ),
        ]
        b_3 = E[:, -1:]
        m_1, m_2, m_3 = m_1.view(-1, 128), m_2.view(-1, 128), m_3.view(-1, 1)
        b_1, b_2, b_3 = b_1.view(-1, 128), b_2.view(-1, 128), b_3
        return [m_1, m_2, m_3], [b_1, b_2, b_3]

    def forward(self, X, z, E=None, hidden=None):
        if hidden == None and E == None:
            init_c, init_h = [], []
            for _ in range(self.num_rnn_layer):
                init_c.append(torch.tanh(self.init_lin_h(z)))
                init_h.append(torch.tanh(self.init_lin_c(z)))
            # Initialize hidden inputs for the LSTM
            hidden = (torch.stack(init_c, dim=0), torch.stack(init_h, dim=0))
        
            # Initialize an input for the LSTM
            inputs = torch.tanh(self.init_input(z))
        else:
            inputs = self.lin_transform_up(E)

        out, hidden = self.rnn(inputs.unsqueeze(0), hidden)

        E = self.lin_transform_down(out.squeeze(0))
        
        m_list, bias_list = self.nn_construction(E)
        batch_size = X.shape[0]
        pred = torch.reshape(X, (batch_size, 28 * 28))
        for i, m in enumerate(m_list):
            if i != len(m_list)-1:

                pred = torch.relu(torch.add(torch.mm(pred, m), bias_list[i]))
            else:
                pred = torch.sigmoid(torch.add(torch.mm(pred, m), bias_list[i]))
        
        return E, hidden, pred