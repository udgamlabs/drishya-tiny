"""Store large fp32 weights as fp16 and cast them back to fp32 inside the graph (fp16 download, fp32 maths).

Full fp16 compute gave NaNs on WebGPU for the DiT and a broken T5 encoder, so the browser files keep fp32 compute.
"""
import sys
import numpy as np, onnx
from onnx import helper, numpy_helper, TensorProto

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
g = m.graph
casts, keep = [], []
n = 0
for init in g.initializer:
    if init.data_type == TensorProto.FLOAT and np.prod(init.dims) >= 1024:
        a = numpy_helper.to_array(init)
        name16 = init.name + "__fp16"
        keep.append(numpy_helper.from_array(a.astype(np.float16), name16))
        casts.append(helper.make_node("Cast", [name16], [init.name], to=TensorProto.FLOAT, name=init.name + "__cast"))
        n += 1
    else:
        keep.append(init)
del g.initializer[:]
g.initializer.extend(keep)
nodes = list(g.node)
del g.node[:]
g.node.extend(casts + nodes)
onnx.checker.check_model(m, full_check=False)
onnx.save(m, dst)
print("converted", n, "initializers ->", dst)
