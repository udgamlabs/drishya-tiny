"""fp32 ONNX -> fp16 weights/compute with fp32 inputs and outputs (keep_io_types)."""
import sys, onnx
from onnxruntime.transformers.float16 import convert_float_to_float16
src, dst = sys.argv[1], sys.argv[2]
block = sys.argv[3].split(",") if len(sys.argv) > 3 and sys.argv[3] else None
m = onnx.load(src)
m16 = convert_float_to_float16(m, keep_io_types=True, op_block_list=block)
onnx.save(m16, dst)
print("saved", dst)
