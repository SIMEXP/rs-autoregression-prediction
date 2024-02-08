import torch
from fmri_autoreg.tools import load_model
from thop import profile
from torch_geometric.nn import ChebConv
from torchinfo import summary

path_model = {
    "big": 'outputs/scaling_nonlinearchebnet/multiruns/2023-11-21_20-06-29/++experiment.scaling.n_sample=18000,++experiment.scaling.segment=1,++model.FK="128,32,128,32,128,32,128,32,128,32,64,16,64,16",++model.M="128,64,32,16,8,1",++model.dropout=0.1,++random_state=1/model.pkl',
    "medium": 'outputs/scaling_nonlinearchebnet/multiruns/2023-11-12_18-58-06/++experiment.scaling.n_sample=1000,++experiment.scaling.segment=1,++model.FK="32,6,32,6,32,6,16,6,16,6",++model.M="32,16,8,1",++model.dropout=0.1,++random_state=1/model.pkl',
    "small": "outputs/scaling_nonlinearchebnet/multiruns/2023-11-14_14-54-04/++experiment.scaling.n_sample=11000,++experiment.scaling.segment=1,++random_state=1/model.pkl",
}

# def count_your_model(module, input, output

batch_size = 1
n_roi = 197
n_tr = 16
test_input = torch.randn(batch_size, n_roi, n_tr).to(torch.device("cuda:0"))
for key, value in path_model.items():
    model = load_model(value)
    # print(f'Loaded model {key}')
    total_ops, total_params = profile(
        model,
        inputs=(test_input,)
        # custom_ops={ChebConv: count_your_model}
    )
    print(
        f"number of operations: {total_ops}, number of parameters: {total_params} for model {key}"
    )
    print(summary(model))
