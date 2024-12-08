import math
import time
import torch
import typing
import logging

import numpy as np
import torch.nn as nn
import torch.distributed as dist

from functools import partial
from collections import Counter, defaultdict, OrderedDict
from itertools import zip_longest
from tqdm import tqdm


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    num_layers = len(model.blocks) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
            
        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    else:
        return num_layers


def warmup_adjust_learning_rate(optimizer, epoch, lr, min_lr, warmup_epochs, epochs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        lr = lr * epoch / warmup_epochs 
    else:
        lr = min_lr + (lr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
            
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def convert_weights_to_fp16(model: nn.Module):
    """Convert applicable model parameters to fp16"""
    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        # if isinstance(l, (nn.MultiheadAttention, Attention)):
        #     for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
        #         tensor = getattr(l, attr)
        #         if tensor is not None:
        #             tensor.data = tensor.data.half()

    model.apply(_convert_weights_to_fp16)


def adjust_learning_rate(optimizer, init_lr, min_lr, epoch, epochs):
    cur_lr = min_lr + max(init_lr - min_lr, 0) * 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr


def get_params_groups(model):
    regularized, not_regularized = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        
        if param.ndim > 1:
            regularized.append(param)
        else:
            not_regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


@torch.no_grad()
def accuracy_at_k(
        outputs: torch.Tensor, targets: torch.Tensor, top_k = (1, 5)
):
    max_k = max(top_k)
    batch_size = targets.size(0)

    _, pred = outputs.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = []
    for k in top_k:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


@torch.no_grad()
def evaluate(model, loader):
    count, acc1, acc5 = [], [], []
    for (images, labels) in tqdm(loader, total=len(loader)):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        
        # Inference
        logits = model(images)
        
        # Accuracy
        _acc1, _acc5 = accuracy_at_k(logits, targets=labels, top_k=(1, 5))
        _count = labels.shape[0]
        count.append(images.shape[0])
        acc1.append(_acc1.item() * _count)
        acc5.append(_acc5.item() * _count)
    return sum(acc1)/sum(count), sum(acc5)/sum(count)


"""
   Ref - https://github.com/facebookresearch/ToMe 
"""
def benchmark(
    model: torch.nn.Module,
    device: torch.device = 0,
    input_size: typing.Tuple[int] = (3, 224, 224),
    batch_size: int = 64,
    runs: int = 40,
    throw_out: float = 0.25,
    use_fp16: bool = False,
    verbose: bool = False,
) -> float:
    """
    Benchmark the given model with random inputs at the given batch size.

    Args:
     - model: the module to benchmark
     - device: the device to use for benchmarking
     - input_size: the input size to pass to the model (channels, h, w)
     - batch_size: the batch size to use for evaluation
     - runs: the number of total runs to do
     - throw_out: the percentage of runs to throw out at the start of testing
     - use_fp16: whether or not to benchmark with float16 and autocast
     - verbose: whether or not to use tqdm to print progress / print throughput at end

    Returns:
     - the throughput measured in images / second
    """
    if not isinstance(device, torch.device):
        device = torch.device(device)
    is_cuda = torch.device(device).type == "cuda"

    model = model.eval().to(device)
    input = torch.rand(batch_size, *input_size, device=device)
    if use_fp16:
        input = input.half()

    warm_up = int(runs * throw_out)
    total = 0
    start = time.time()

    with torch.autocast(device.type, enabled=use_fp16):
        with torch.no_grad():
            for i in tqdm(range(runs), disable=not verbose, desc="Benchmarking"):
                if i == warm_up:
                    if is_cuda:
                        torch.cuda.synchronize()
                    total = 0
                    start = time.time()

                model(input)
                total += batch_size

    if is_cuda:
        torch.cuda.synchronize()

    end = time.time()
    elapsed = end - start
    throughput = total / elapsed

    if verbose:
        print(f"Throughput: {throughput:.2f} im/s")
    return throughput



"""
    flop_count
"""
def get_shape(val: object) -> typing.List[int]:
    """
    Get the shapes from a jit value object.
    Args:
        val (torch._C.Value): jit value object.
    Returns:
        list(int): return a list of ints.
    """
    if val.isCompleteTensor():  # pyre-ignore
        r = val.type().sizes()  # pyre-ignore
        if not r:
            r = [1]
        return r
    elif val.type().kind() in ("IntType", "FloatType"):
        return [1]
    else:
        return [0]
        # raise ValueError()


def addmm_flop_jit(
    inputs: typing.List[object], outputs: typing.List[object]
) -> typing.Counter[str]:
    """
    This method counts the flops for fully connected layers with torch script.
    Args:
        inputs (list(torch._C.Value)): The input shape in the form of a list of
            jit object.
        outputs (list(torch._C.Value)): The output shape in the form of a list
            of jit object.
    Returns:
        Counter: A Counter dictionary that records the number of flops for each
            operation.
    """
    # Count flop for nn.Linear
    # inputs is a list of length 3.
    input_shapes = [get_shape(v) for v in inputs[1:3]]
    # input_shapes[0]: [batch size, input feature dimension]
    # input_shapes[1]: [batch size, output feature dimension]
    assert len(input_shapes[0]) == 2
    assert len(input_shapes[1]) == 2
    batch_size, input_dim = input_shapes[0]
    output_dim = input_shapes[1][1]
    flop = batch_size * input_dim * output_dim
    flop_counter = Counter({"addmm": flop})
    return flop_counter


def bmm_flop_jit(inputs, outputs):
    # Count flop for nn.Linear
    # inputs is a list of length 3.
    input_shapes = [get_shape(v) for v in inputs]
    # input_shapes[0]: [batch size, input feature dimension]
    # input_shapes[1]: [batch size, output feature dimension]
    assert len(input_shapes[0]) == 3
    assert len(input_shapes[1]) == 3
    T, batch_size, input_dim = input_shapes[0]
    output_dim = input_shapes[1][2]
    flop = T * batch_size * input_dim * output_dim
    flop_counter = Counter({"bmm": flop})
    return flop_counter


def basic_binary_op_flop_jit(inputs, outputs, name):
    input_shapes = [get_shape(v) for v in inputs]
    # for broadcasting
    input_shapes = [s[::-1] for s in input_shapes]
    max_shape = np.array(list(zip_longest(*input_shapes, fillvalue=1))).max(1)
    flop = np.prod(max_shape)
    flop_counter = Counter({name: flop})
    return flop_counter


def rsqrt_flop_jit(inputs, outputs):
    input_shapes = [get_shape(v) for v in inputs]
    flop = np.prod(input_shapes[0]) * 2
    flop_counter = Counter({"rsqrt": flop})
    return flop_counter


def dropout_flop_jit(inputs, outputs):
    input_shapes = [get_shape(v) for v in inputs[:1]]
    flop = np.prod(input_shapes[0])
    flop_counter = Counter({"dropout": flop})
    return flop_counter


def softmax_flop_jit(inputs, outputs):
    # from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/profiler/internal/flops_registry.py
    input_shapes = [get_shape(v) for v in inputs[:1]]
    flop = np.prod(input_shapes[0]) * 5
    flop_counter = Counter({"softmax": flop})
    return flop_counter


def _reduction_op_flop_jit(inputs, outputs, reduce_flops=1, finalize_flops=0):
    input_shapes = [get_shape(v) for v in inputs]
    output_shapes = [get_shape(v) for v in outputs]

    in_elements = np.prod(input_shapes[0])
    out_elements = np.prod(output_shapes[0])

    num_flops = in_elements * reduce_flops + out_elements * (
        finalize_flops - reduce_flops
    )

    return num_flops


def conv_flop_count(
    x_shape: typing.List[int],
    w_shape: typing.List[int],
    out_shape: typing.List[int],
) -> typing.Counter[str]:
    """
    This method counts the flops for convolution. Note only multiplication is
    counted. Computation for addition and bias is ignored.
    Args:
        x_shape (list(int)): The input shape before convolution.
        w_shape (list(int)): The filter shape.
        out_shape (list(int)): The output shape after convolution.
    Returns:
        Counter: A Counter dictionary that records the number of flops for each
            operation.
    """
    batch_size, Cin_dim, Cout_dim = x_shape[0], w_shape[1], out_shape[1]
    out_size = np.prod(out_shape[2:])
    kernel_size = np.prod(w_shape[2:])
    flop = batch_size * out_size * Cout_dim * Cin_dim * kernel_size
    flop_counter = Counter({"conv": flop})
    return flop_counter


def conv_flop_jit(
    inputs: typing.List[object], outputs: typing.List[object]
) -> typing.Counter[str]:
    """
    This method counts the flops for convolution using torch script.
    Args:
        inputs (list(torch._C.Value)): The input shape in the form of a list of
            jit object before convolution.
        outputs (list(torch._C.Value)): The output shape in the form of a list
            of jit object after convolution.
    Returns:
        Counter: A Counter dictionary that records the number of flops for each
            operation.
    """
    # Inputs of Convolution should be a list of length 12. They represent:
    # 0) input tensor, 1) convolution filter, 2) bias, 3) stride, 4) padding,
    # 5) dilation, 6) transposed, 7) out_pad, 8) groups, 9) benchmark_cudnn,
    # 10) deterministic_cudnn and 11) user_enabled_cudnn.
    # + 12) allowTF32CuDNN for PyTorch 1.7
    assert len(inputs) == 13
    x, w = inputs[:2]
    x_shape, w_shape, out_shape = (
        get_shape(x),
        get_shape(w),
        get_shape(outputs[0]),
    )
    return conv_flop_count(x_shape, w_shape, out_shape)


def einsum_flop_jit(
    inputs: typing.List[object], outputs: typing.List[object]
) -> typing.Counter[str]:
    """
    This method counts the flops for the einsum operation. We currently support
    two einsum operations: "nct,ncp->ntp" and "ntg,ncg->nct".
    Args:
        inputs (list(torch._C.Value)): The input shape in the form of a list of
            jit object before einsum.
        outputs (list(torch._C.Value)): The output shape in the form of a list
            of jit object after einsum.
    Returns:
        Counter: A Counter dictionary that records the number of flops for each
            operation.
    """
    # Inputs of einsum should be a list of length 2.
    # Inputs[0] stores the equation used for einsum.
    # Inputs[1] stores the list of input shapes.
    assert len(inputs) == 2 or len(inputs) == 1
    equation = inputs[0].toIValue()  # pyre-ignore
    # Get rid of white space in the equation string.
    equation = equation.replace(" ", "")
    # Re-map equation so that same equation with different alphabet
    # representations will look the same.
    letter_order = OrderedDict((k, 0) for k in equation if k.isalpha()).keys()
    mapping = {ord(x): 97 + i for i, x in enumerate(letter_order)}
    equation = equation.translate(mapping)
    input_shapes_jit = inputs[1].node().inputs()  # pyre-ignore
    input_shapes = [get_shape(v) for v in input_shapes_jit]

    if equation == "abc,abd->acd":
        n, c, t = input_shapes[0]
        p = input_shapes[-1][-1]
        flop = n * c * t * p
        flop_counter = Counter({"einsum": flop})
        return flop_counter

    elif equation == "abc,adc->adb":
        n, t, g = input_shapes[0]
        c = input_shapes[-1][1]
        flop = n * t * g * c
        flop_counter = Counter({"einsum": flop})
        return flop_counter

    elif equation == "abcd,acd->abc":
        n, t, g, c = input_shapes[0]
        flop = n * t * g * c
        flop_counter = Counter({"einsum": flop})
        return flop_counter

    elif equation == "abcd,abce->acde":
        n, t, g, i = input_shapes[0]
        c = input_shapes[-1][-1]
        flop = n * t * g * i * c
        flop_counter = Counter({"einsum": flop})
        return flop_counter

    elif equation == "abcd,aced->abce":
        n, t, g, i = input_shapes[0]
        c = input_shapes[-1][-2]
        flop = n * t * g * c * i
        flop_counter = Counter({"einsum": flop})
        return flop_counter

    elif equation == "abcd,ced->abce":
        n, t, g, i = input_shapes[0]
        c = input_shapes[-1][-2]
        flop = n * t * g * c * i
        flop_counter = Counter({"einsum": flop})
        return flop_counter

    elif equation == "abc,adc->abd":
        n, t, g = input_shapes[0]
        c = input_shapes[-1][-2]
        flop = n * t * g * c
        flop_counter = Counter({"einsum": flop})
        return flop_counter

    elif equation == "abcd,ed->abce":
        n, t, g, i = input_shapes[0]
        c = input_shapes[-1][0]
        flop = n * t * g * c * i
        flop_counter = Counter({"einsum": flop})
        return flop_counter

    elif equation == "abc,acd->abd":
        n, t, g = input_shapes[0]
        c = input_shapes[-1][-1]
        flop = n * t * g * c
        flop_counter = Counter({"einsum": flop})
        return flop_counter

    elif equation == "abcd,abed->abce":
        n, t, g, i = input_shapes[0]
        c = input_shapes[-1][-2]
        flop = n * t * g * c * i
        flop_counter = Counter({"einsum": flop})
        return flop_counter

    elif equation == "abcd,abde->abce":
        n, t, g, i = input_shapes[0]
        c = input_shapes[-1][-1]
        flop = n * t * g * c * i
        flop_counter = Counter({"einsum": flop})
        return flop_counter

    elif equation == "abcd,abcd->abc":
        n, t, g, i = input_shapes[0]
        flop = n * t * g * i
        flop_counter = Counter({"einsum": flop})
        return flop_counter

    elif equation == "abcde,abfde->abdcf":
        n, t, g, i, j = input_shapes[0]
        c = input_shapes[-1][-3]
        flop = n * t * g * i * j * c
        flop_counter = Counter({"einsum": flop})
        return flop_counter

    elif equation == "abcde,abecf->abdcf":
        n, t, g, i, j = input_shapes[0]
        c = input_shapes[-1][-1]
        flop = n * t * g * i * j * c
        flop_counter = Counter({"einsum": flop})
        return flop_counter

    elif equation == "abcdefg->abdfceg":
        flop = 0
        flop_counter = Counter({"einsum": flop})
        return flop_counter

    elif equation == "abcdefg->abecfdg":
        flop = 0
        flop_counter = Counter({"einsum": flop})
        return flop_counter

    else:
        raise NotImplementedError("Unsupported einsum operation. {}".format(equation))


"""
def matmul_flop_jit(
    inputs: typing.List[object], outputs: typing.List[object]
) -> typing.Counter[str]:
    This method counts the flops for matmul.
    Args:
        inputs (list(torch._C.Value)): The input shape in the form of a list of
            jit object before matmul.
        outputs (list(torch._C.Value)): The output shape in the form of a list
            of jit object after matmul.
    Returns:
        Counter: A Counter dictionary that records the number of flops for each
            operation.
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two matrices.
    print("+_+in matmul")
    for v in inputs:
        print(get_shape(v))
    input_shapes = [get_shape(v) for v in inputs]
    assert len(input_shapes) == 2
    assert len(input_shapes[1]) == 2
    assert input_shapes[0][-1] == input_shapes[1][0]
    batch_dim = input_shapes[0][0]
    m1_dim, m2_dim = input_shapes[1]
    flop = m1_dim * m2_dim * batch_dim
    flop_counter = Counter({"matmul": flop})
    return flop_counter
"""


def matmul_flop_jit(inputs: typing.List[typing.Any], outputs: typing.List[typing.Any]) -> typing.Counter[str]:
    """
    This method counts the flops for matmul.
    Args:
        inputs (list(torch._C.Value)): The input shape in the form of a list of
            jit object before matmul.
        outputs (list(torch._C.Value)): The output shape in the form of a list
            of jit object after matmul.
    Returns:
        Counter: A Counter dictionary that records the number of flops for each
            operation.
    """
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two matrices.
    input_shapes = [get_shape(v) for v in inputs]
    assert len(input_shapes) == 2, input_shapes
    assert input_shapes[0][-1] == input_shapes[1][-2], input_shapes
    flop = np.prod(input_shapes[0]) * input_shapes[-1][-1]
    flop_counter = Counter({"matmul": flop})
    return flop_counter


def batchnorm_flop_jit(
    inputs: typing.List[object], outputs: typing.List[object]
) -> typing.Counter[str]:
    """
    This method counts the flops for batch norm.
    Args:
        inputs (list(torch._C.Value)): The input shape in the form of a list of
            jit object before batch norm.
        outputs (list(torch._C.Value)): The output shape in the form of a list
            of jit object after batch norm.
    Returns:
        Counter: A Counter dictionary that records the number of flops for each
            operation.
    """
    # Inputs[0] contains the shape of the input.
    input_shape = get_shape(inputs[0])
    assert 2 <= len(input_shape) <= 5
    flop = np.prod(input_shape) * 4
    flop_counter = Counter({"batchnorm": flop})
    return flop_counter

def linear_flop_jit(
    inputs: typing.List[object], outputs: typing.List[object]
) -> typing.Counter[str]:
    """
    This method counts the flops for linear.
    Args:
        inputs (list(torch._C.Value)): The input shape in the form of a list of
            jit object before linear.
        outputs (list(torch._C.Value)): The output shape in the form of a list
            of jit object after linear.
    Returns:
        Counter: A Counter dictionary that records the number of flops for each
            operation.
    """
    flop = 0
    input_shapes = [get_shape(v) for v in inputs]
    assert len(input_shapes) == 3
    *input_shape, input_dim = input_shapes[0]
    output_dim = input_shapes[1][0]
    flop = np.prod(input_shape) * output_dim * (input_dim + 1)
    return Counter({"linear": flop})

# A dictionary that maps supported operations to their flop count jit handles.
_SUPPORTED_OPS: typing.Dict[str, typing.Callable] = {
    "aten::addmm": addmm_flop_jit,
    "aten::_convolution": conv_flop_jit,
    "aten::einsum": einsum_flop_jit,
    "aten::matmul": matmul_flop_jit,
    "aten::batch_norm": batchnorm_flop_jit,
    "aten::layer_norm": batchnorm_flop_jit,
    "aten::bmm": bmm_flop_jit,
    "aten::add": partial(basic_binary_op_flop_jit, name="aten::add"),
    "aten::add_": partial(basic_binary_op_flop_jit, name="aten::add_"),
    "aten::mul": partial(basic_binary_op_flop_jit, name="aten::mul"),
    "aten::sub": partial(basic_binary_op_flop_jit, name="aten::sub"),
    "aten::div": partial(basic_binary_op_flop_jit, name="aten::div"),
    "aten::exp_": partial(basic_binary_op_flop_jit, name="aten::exp_"),
    "aten::floor_divide": partial(basic_binary_op_flop_jit, name="aten::floor_divide"),
    "aten::relu": partial(basic_binary_op_flop_jit, name="aten::relu"),
    "aten::relu_": partial(basic_binary_op_flop_jit, name="aten::relu_"),
    "aten::gelu": partial(basic_binary_op_flop_jit, name="aten::gelu"),
    "aten::linalg_norm": partial(basic_binary_op_flop_jit, name="aten::linalg_norm"),
    "aten::cumsum": partial(basic_binary_op_flop_jit, name="aten::cumsum"),
    "aten::linear": linear_flop_jit,
    "aten::rsqrt": rsqrt_flop_jit,
    "aten::softmax": softmax_flop_jit,
    "aten::dropout": dropout_flop_jit,
}

# A list that contains ignored operations.
_IGNORED_OPS: typing.List[str] = [
    "aten::Int",
    "aten::__and__",
    "aten::arange",
    "aten::cat",
    "aten::clamp",
    "aten::clamp_",
    "aten::contiguous",
    "aten::copy_",
    "aten::detach",
    "aten::empty",
    "aten::eq",
    "aten::expand",
    "aten::expand_as",
    "aten::range",
    "aten::flatten",
    "aten::floor",
    "aten::full",
    "aten::gt",
    "aten::index",
    "aten::index_put_",
    "aten::max",
    "aten::nonzero",
    "aten::permute",
    "aten::remainder",
    "aten::reshape",
    "aten::select",
    "aten::size",
    "aten::slice",
    "aten::split_with_sizes",
    "aten::squeeze",
    "aten::t",
    "aten::to",
    "aten::transpose",
    "aten::unsqueeze",
    "aten::view",
    "aten::zeros",
    "aten::zeros_like",
    "aten::ones",
    "aten::ones_like",
    "aten::eye",
    "aten::rsub",
    "prim::Constant",
    "prim::Int",
    "prim::ListConstruct",
    "prim::ListUnpack",
    "prim::NumToTensor",
    "prim::TupleConstruct",
    "aten::chunk",
    "aten::gather",
    "aten::where",
    "aten::ne",
    "aten::scatter",
    "aten::min",
    "aten::sort",
    "aten::constant_pad_nd",
    "aten::type_as",
    "aten::repeat",
    "aten::reciprocal",
    "aten::linspace",
    "aten::repeat",
    "aten::abs",
]

_HAS_ALREADY_SKIPPED = False


def flop_count(
    model: nn.Module,
    inputs: typing.Tuple[object, ...],
    whitelist: typing.Union[typing.List[str], None] = None,
    customized_ops: typing.Union[typing.Dict[str, typing.Callable], None] = None,
) -> typing.DefaultDict[str, float]:
    """
    Given a model and an input to the model, compute the Gflops of the given
    model. Note the input should have a batch size of 1.
    Args:
        model (nn.Module): The model to compute flop counts.
        inputs (tuple): Inputs that are passed to `model` to count flops.
            Inputs need to be in a tuple.
        whitelist (list(str)): Whitelist of operations that will be counted. It
            needs to be a subset of _SUPPORTED_OPS. By default, the function
            computes flops for all supported operations.
        customized_ops (dict(str,Callable)) : A dictionary contains customized
            operations and their flop handles. If customized_ops contains an
            operation in _SUPPORTED_OPS, then the default handle in
             _SUPPORTED_OPS will be overwritten.
    Returns:
        defaultdict: A dictionary that records the number of gflops for each
            operation.
    """
    # Copy _SUPPORTED_OPS to flop_count_ops.
    # If customized_ops is provided, update _SUPPORTED_OPS.
    flop_count_ops = _SUPPORTED_OPS.copy()
    if customized_ops:
        flop_count_ops.update(customized_ops)

    # If whitelist is None, count flops for all suported operations.
    if whitelist is None:
        whitelist_set = set(flop_count_ops.keys())
    else:
        whitelist_set = set(whitelist)

    # Torch script does not support parallell torch models.
    if isinstance(
        model,
        (nn.parallel.distributed.DistributedDataParallel, nn.DataParallel),
    ):
        model = model.module  # pyre-ignore

    assert set(whitelist_set).issubset(
        flop_count_ops
    ), "whitelist needs to be a subset of _SUPPORTED_OPS and customized_ops."
    assert isinstance(inputs, tuple), "Inputs need to be in a tuple."

    # Compatibility with torch.jit.
    if hasattr(torch.jit, "get_trace_graph"):
        trace, _ = torch.jit.get_trace_graph(model, inputs)
        trace_nodes = trace.graph().nodes()
    else:
        trace, _ = torch.jit._get_trace_graph(model, inputs)
        trace_nodes = trace.nodes()

    skipped_ops = Counter()
    total_flop_counter = Counter()

    for node in trace_nodes:
        kind = node.kind()
        # print("+_+  +_+",kind)

        if kind not in whitelist_set:
            # If the operation is not in _IGNORED_OPS, count skipped operations.
            if kind not in _IGNORED_OPS:
                skipped_ops[kind] += 1
            continue

        handle_count = flop_count_ops.get(kind, None)
        if handle_count is None:
            continue

        inputs, outputs = list(node.inputs()), list(node.outputs())
        flops_counter = handle_count(inputs, outputs)
        total_flop_counter += flops_counter

    global _HAS_ALREADY_SKIPPED
    if len(skipped_ops) > 0 and not _HAS_ALREADY_SKIPPED:
        _HAS_ALREADY_SKIPPED = True
        for op, freq in skipped_ops.items():
            logging.warning("Skipped operation {} {} time(s)".format(op, freq))

    # Convert flop count to gigaflops.
    final_count = defaultdict(float)
    for op in total_flop_counter:
        final_count[op] = total_flop_counter[op] / 1e9

    return final_count


# Ref) https://github.com/facebookresearch/dino/blob/main/utils.py
def print_setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print



def dist_init(local_rank):
    dist.init_process_group(backend='nccl', init_method="env://", rank=local_rank)
    torch.cuda.set_device(local_rank)
    print_setup_for_distributed(local_rank == 0)
    torch.distributed.barrier()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0