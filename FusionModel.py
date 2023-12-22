import copy
import pennylane as qml
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from math import pi
import torch.nn.functional as F
from Arguments import Arguments
args = Arguments()


def gen_arch(change_code, base_code=args.base_code):
    arch_code = base_code[1:] * base_code[0]
    if change_code != None:
        if type(change_code[0]) != type([]):
            change_code = [change_code]
        change_qubit = change_code[-1][0]
        if change_code is not None:
            for i in range(len(change_code)):
                q = change_code[i][0]  # the qubit changed
                for i, t in enumerate(change_code[i][1:]):
                    arch_code[q + i * args.n_qubits] = t
    return arch_code


def translator(change_code, trainable='partial', base_code=args.base_code):
    # if type(change_code[0]) != type([]):
    #     change_code = [change_code]
    net = gen_arch(change_code, base_code)
    updated_design = {}
    if trainable == 'full' or change_code is None:
        updated_design['change_qubit'] = None
    else:
        if type(change_code[0]) != type([]): change_code = [change_code]
        updated_design['change_qubit'] = change_code[-1][0]

    # num of layers
    updated_design['n_layers'] = args.n_layers

    for layer in range(updated_design['n_layers']):
        # categories of single-qubit parametric gates
        for i in range(args.n_qubits):
            updated_design['rot' + str(layer) + str(i)] = 'U3'

        # categories and positions of entangled gates
        for j in range(args.n_qubits):
            updated_design['enta' + str(layer) + str(j)] = ('CU3', [j, net[j + layer * args.n_qubits]])

        updated_design['total_gates'] = updated_design['n_layers'] * args.n_qubits
    return updated_design


dev = qml.device("lightning.qubit", wires=args.n_qubits)


@qml.qnode(dev, interface="torch", diff_method="adjoint")
def quantum_net(q_input_features, q_weights_rot, q_weights_enta, **kwargs):
    current_design = kwargs['design']
    q_input_features = torch.transpose(q_input_features, 0, 1)  # (n_qubits, batches)
    q_input_features = q_input_features.reshape(args.n_qubits, 3, -1)  # (7, 3, 32)

    for layer in range(current_design['n_layers']):
        # data reuploading
        for i in range(args.n_qubits):
            qml.Rot(*q_input_features[i], wires=i)
        # single-qubit parametric gates and entangled gates
        for j in range(args.n_qubits):
            if current_design['rot' + str(layer) + str(j)] == 'Rx':
                qml.RX(q_weights_rot[j][layer], wires=j)
            else:
                qml.RY(q_weights_rot[j][layer], wires=j)

            if current_design['enta' + str(layer) + str(j)][1][0] != current_design['enta' + str(layer) + str(j)][1][1]:
                if current_design['enta' + str(layer) + str(j)][0] == 'IsingXX':
                    qml.IsingXX(q_weights_enta[j][layer], wires=current_design['enta' + str(layer) + str(j)][1])
                else:
                    qml.IsingZZ(q_weights_enta[j][layer], wires=current_design['enta' + str(layer) + str(j)][1])

    return [qml.expval(qml.PauliZ(i)) for i in range(args.n_qubits)]


class QuantumLayer(nn.Module):
    def __init__(self, arguments, design):
        super(QuantumLayer, self).__init__()
        self.args = arguments
        self.design = design
        self.q_params_rot, self.q_params_enta = nn.ParameterList(), nn.ParameterList()
        for i in range(self.args.n_qubits):
            if self.design['change_qubit'] is None:
                rot_trainable = True
                enta_trainable = True
            elif i == self.design['change_qubit']:
                rot_trainable = False
                enta_trainable = True
            else:
                rot_trainable = False
                enta_trainable = False
            self.q_params_rot.append(
                nn.Parameter(pi * torch.rand(self.design['n_layers']), requires_grad=rot_trainable))
            self.q_params_enta.append(
                nn.Parameter(pi * torch.rand(self.design['n_layers']), requires_grad=enta_trainable))

    def forward(self, input_features):
        output = quantum_net(input_features, self.q_params_rot, self.q_params_enta, design=self.design)
        q_out = torch.stack([output[i] for i in range(len(output))]).float()  # (n_qubits, batch)
        if len(q_out.shape) == 1:
            q_out = q_out.unsqueeze(1)
        q_out = torch.transpose(q_out, 0, 1)  # (batch, n_qubits)
        return q_out


class TQLayer(tq.QuantumModule):
    def __init__(self, arguments, design):
        super().__init__()
        self.args = arguments
        self.design = design
        self.n_wires = self.args.n_qubits

        self.rots, self.entas = tq.QuantumModuleList(), tq.QuantumModuleList()

        self.q_params_rot, self.q_params_enta = [], []
        for i in range(self.args.n_qubits):
            self.q_params_rot.append(pi * torch.rand(self.design['n_layers']))
            self.q_params_enta.append(pi * torch.rand(3 * self.design['n_layers'])) # each CU3 gate needs 3 parameters

        for layer in range(self.design['n_layers']):
            for q in range(self.n_wires):
                # 'trainable' option
                if self.design['change_qubit'] is None:
                    rot_trainable = True
                    enta_trainable = True
                elif q == self.design['change_qubit']:
                    rot_trainable = False
                    enta_trainable = True
                else:
                    rot_trainable = False
                    enta_trainable = False
                # single-qubit parametric gates
                if self.design['rot' + str(layer) + str(q)] == 'U3':
                    self.rots.append(tq.U3(has_params=True, trainable=rot_trainable,
                                           init_params=self.q_params_rot[q][layer].reshape((1,))))
                # else:
                #     self.rots.append(tq.RY(has_params=True, trainable=rot_trainable,
                #                            init_params=self.q_params_rot[q][layer].reshape((1,))))
                # entangled gates
                if self.design['enta' + str(layer) + str(q)][0] == 'CU3':
                    self.entas.append(tq.CU3(has_params=True, trainable=enta_trainable,
                                             init_params=self.q_params_enta[q][layer*3:(layer+1)*3].reshape((3,))))
                # else:
                #     self.entas.append(tq.RZZ(has_params=True, trainable=enta_trainable,
                #                              init_params=self.q_params_enta[q][layer].reshape((1,))))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6)  # 'down_sample_kernel_size' = 6
        x = x.view(bsz, -1)

        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)

        # encode input image with '4x4_ryzxy' gates
        func_list = [{'input_idx': [0], 'func': 'ry', 'wires': [0]}, {'input_idx': [1], 'func': 'ry', 'wires': [1]}, {'input_idx': [2], 'func': 'ry', 'wires': [2]}, {'input_idx': [3], 'func': 'ry', 'wires': [3]}, {'input_idx': [4], 'func': 'rz', 'wires': [0]}, {'input_idx': [5], 'func': 'rz', 'wires': [1]}, {'input_idx': [6], 'func': 'rz', 'wires': [2]}, {'input_idx': [7], 'func': 'rz', 'wires': [3]}, {'input_idx': [8], 'func': 'rx', 'wires': [0]}, {'input_idx': [9], 'func': 'rx', 'wires': [1]}, {'input_idx': [10], 'func': 'rx', 'wires': [2]}, {'input_idx': [11], 'func': 'rx', 'wires': [3]}, {'input_idx': [12], 'func': 'ry', 'wires': [0]}, {'input_idx': [13], 'func': 'ry', 'wires': [1]}, {'input_idx': [14], 'func': 'ry', 'wires': [2]}, {'input_idx': [15], 'func': 'ry', 'wires': [3]}]
        self.encoder = tq.GeneralEncoder(func_list)
        self.encoder(qdev, x)

        for layer in range(self.design['n_layers']):
            # for i in range(self.n_wires):
            #     tqf.rot(qdev, wires=i, params=x[:, i])
            for j in range(self.n_wires):
                self.rots[j + layer * self.n_wires](qdev, wires=j)
                if self.design['enta' + str(layer) + str(j)][1][0] != self.design['enta' + str(layer) + str(j)][1][1]:
                    self.entas[j + layer * self.n_wires](qdev, wires=self.design['enta' + str(layer) + str(j)][1])
        return self.measure(qdev)


class QNet(nn.Module):
    def __init__(self, arguments, design):
        super(QNet, self).__init__()
        self.args = arguments
        self.design = design
        if args.backend == 'pennylane':
            self.QuantumLayer = QuantumLayer(self.args, self.design)
        else:
            self.QuantumLayer = TQLayer(self.args, self.design)
        self.Regressor = nn.Linear(self.args.n_qubits, 1)
        for name, param in self.named_parameters():
            if "QuantumLayer" not in name:
                param.requires_grad = False

    def forward(self, x_image):
        exp_val = self.QuantumLayer(x_image)
        # output = torch.tanh(self.Regressor(exp_val).squeeze(dim=1)) * 3
        output = F.log_softmax(exp_val, dim=1)
        return output
