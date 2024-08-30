import torch.onnx

model = torch.load("./resnet18.pth", map_location="cpu")

model.eval()
dummy_input = torch.randn([8, 3, 224, 224], requires_grad=True)

# Export the model
torch.onnx.export(model,  # model being run
                  dummy_input,  # model input (or a tuple for multiple inputs)
                  "resnet18.onnx",  # where to save the model
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=10,  # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['modelInput'],  # the model's input names
                  output_names=['modelOutput'],  # the model's output names
                  dynamic_axes={'modelInput': {0: 'batch_size'},  # variable length axes
                                'modelOutput': {0: 'batch_size'}})
