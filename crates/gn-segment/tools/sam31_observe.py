"""Expose decoder boundaries for sam31_stages; diagnostic graphs can change ORT fusion."""
import argparse
from pathlib import Path
import onnx

parser = argparse.ArgumentParser()
parser.add_argument("model", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
model = onnx.load(args.model)
for name in ["add_218", "layer_norm_47", "conv2d", "group_norm", "conv2d_1", "group_norm_1", "conv2d_2", "linear_211"]:
    output = "observe_" + name
    model.graph.node.append(onnx.helper.make_node("Cast", [name], [output], to=onnx.TensorProto.FLOAT))
    model.graph.output.append(onnx.helper.make_tensor_value_info(output, onnx.TensorProto.FLOAT, None))
args.output.parent.mkdir(parents=True, exist_ok=True)
onnx.save(model, args.output)
