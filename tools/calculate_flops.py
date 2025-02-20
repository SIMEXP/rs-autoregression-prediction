# from calflops import calculate_flops

# cp = torch.load(cfg.ckpt_path)
# if "net" not in cp["hyper_parameters"]:
#     model: LightningModule = GraphAutoRegModule.load_from_checkpoint(
#         cfg.ckpt_path, net=hydra.utils.instantiate(cfg.model.net)
#     )
# else:
#     model: LightningModule = GraphAutoRegModule.load_from_checkpoint(
#         cfg.ckpt_path
#     )
# edge = torch.tensor(hydra.utils.instantiate(cfg.model.edge_index))
# input_shape = (
#     batch_size,
#     n_parcel,
#     window_size,
# )
# x = torch.ones(()).new_empty(
#     (*input_shape,), dtype=next(model.parameters()).dtype
# )
# flops, macs, params = calculate_flops(
#     model=model,
#     output_as_string=False,
#     print_results=False,
#     output_unit=None,
#     kwargs={"x": x, "edge_index": edge},
# )
