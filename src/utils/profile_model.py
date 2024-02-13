import torch
from fmri_autoreg.tools import load_model
from thop import profile
from torch_geometric.nn import ChebConv
from torchinfo import summary

path_model = {
    "medium": "outputs/ukbb_downstream_m/model.pkl",
    "small": "outputs/ukbb_downstream_s/model.pkl",
}

# def count_your_model(module, input, output

# batch_size = 100
# n_roi = 197
# n_tr = 321
# test_input = torch.randn(
#     batch_size, n_roi, n_tr
# ).to(torch.device("cuda:0"))
for key, value in path_model.items():
    model = load_model(value)
    print(f"Loaded model {key}")
    # total_ops, total_params = profile(
    #     model,
    #     inputs=(test_input,)
    #     # custom_ops={ChebConv: count_your_model}
    # )
    # print(
    #     f"number of operations: {total_ops}, "
    #     f"number of parameters: {total_params} for model {key}"
    # )
    print(summary(model))
