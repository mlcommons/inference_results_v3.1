import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import quant


from cnn_dm_dataset import CNNDAILYMAIL
from torch.utils.data import DataLoader

from gptq import GPTQ, Observer
from utils import find_layers, DEV

def get_gptj(model):

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import GPTJForCausalLM, AutoModelForCausalLM
    model = GPTJForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16) # TODO: Change to fp16?? --> Error: "LayerNormKernelImpl" not implemented for 'Half'
    model.seqlen = 2048
    return model


@torch.no_grad()
def gptj_sequential(model, dataloader, dev, quantizers=dict()):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.transformer.h
    model.transformer.wte = model.transformer.wte.to(dev)
    model.transformer.ln_f = model.transformer.ln_f.to(dev)

    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            #position_ids = kwargs['position_ids']
            #cache['position_ids'] = position_ids #kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.transformer.wte = model.transformer.wte.to(dev)

    model.transformer.ln_f = model.transformer.ln_f.to(dev)
    if dev != torch.device('cpu'):
        torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    #position_ids = cache['position_ids']

    print('Ready.')

    observer = Observer()
    for i in range(len(layers)):

        print(f'Quantizing layer {i+1}/{len(layers)}..')
        print('+------------------+--------------+------------+-----------+-------+')
        print('|       name       | weight_error | fp_inp_SNR | q_inp_SNR | time  |')
        print('+==================+==============+============+===========+=======+')

        layer = layers[i].to(dev)
        full = find_layers(layer)
        sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}
            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name], observe=args.observe)
                gptq[name].quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False) # TODO: Set 'perchannel' to args.perchannel

            def add_batch(name):

                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            for h in handles:
                h.remove()

            for name in subset:
                scale, zero, g_idx, error = gptq[name].fasterquant(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, name=name)
                quantizers['transformer.h.%d.%s' % (i, name)] = (gptq[name].quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu(), args.wbits, args.groupsize)

                if args.observe:
                    observer.submit(name=name, layerid=i, gptq=gptq[name], error=error)
                else:
                    gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq

        # TODO: Clean
        if dev != torch.device('cpu'):
            torch.cuda.empty_cache()

        inps, outs = outs, inps
        print('+------------------+--------------+------------+-----------+-------+')
        print('\n')
    model.config.use_cache = use_cache

    return quantizers


def quantize_lm_head(model, dataloader, dev, quantizers=dict()):

    model = model.to(dev)
    lm_head = model.lm_head

    print('+------------------+--------------+------------+-----------+-------+')
    print('|       name       | weight_error | fp_inp_SNR | q_inp_SNR | time  |')
    print('+==================+==============+============+===========+=======+')

    gptq_lmhead = GPTQ(lm_head, observe=args.observe)
    gptq_lmhead.quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False) # TODO: Set 'perchannel' to args.perchannel
    
    def add_batch():
        def tmp(_, inp, out):
            gptq_lmhead.add_batch(inp[0].data, out.data)
        return tmp
    
    lm_head.register_forward_hook(add_batch())
    for batch in dataloader:
        transformer_outputs = model.transformer(input_ids=batch[0], attention_mask=batch[1])
        out = lm_head(transformer_outputs[0])
    
    scale, zero, g_idx, error = gptq_lmhead.fasterquant(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, name="lm_head")
    quantizers['lm_head'] = (gptq_lmhead.quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu(), args.wbits, args.groupsize)

    print('+------------------+--------------+------------+-----------+-------+')
    print('\n')

    return quantizers

# TODO: perform packing on GPU
def gptj_pack(model, quantizers, wbits, groupsize, quant_params_output="weight_config.json", compression_dim="K"):
    import json
    layers = find_layers(model)
    
    layers = {n: layers[n] for n in quantizers}
    quant.make_quant_linear(model, quantizers, wbits, groupsize, compression_factor=args.compression_factor, compression_dim=compression_dim)
    qlayers = find_layers(model, [quant.QuantLinear])
    print('Packing ...')
    QConfig = dict()
    
    for name in qlayers:
        print(name)
        quantizers[name], scale, zero, g_idx, _, _ = quantizers[name]
        qlayers[name].pack(layers[name], scale, zero, g_idx)
        QConfig[name] = qlayers[name].get_quant_params(scale, zero, g_idx)
    print('Done.')

    # No need to save params
    # with open(quant_params_output, "w") as fid:
    #     json.dump(QConfig, fid, indent=4)

    return model

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, help='path to model checkpoint')

    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration data samples.')
    parser.add_argument('--percdamp', type=float, default=.01, help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--nearest', action='store_true', help='Whether to run the RTN baseline.')
    parser.add_argument('--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16], help='#bits to use for quantization; use 16 for evaluating base model.')
    parser.add_argument('--trits', action='store_true', help='Whether to use trits for quantization.')
    parser.add_argument('--groupsize', type=int, default=-1, help='Groupsize to use for quantization; default uses full row.')
    parser.add_argument('--save', type=str, default='', help='Save quantized checkpoint under this name.')

    parser.add_argument('--sym', action='store_true', help='Whether to perform symmetric quantization.')
    parser.add_argument('--act-order', action='store_true', help='Whether to apply the activation order GPTQ heuristic')
    parser.add_argument('--true-sequential', action='store_true', help='Whether to run in true sequential model.')

    parser.add_argument('--observe',
                        action='store_true',
                        help='Auto upgrade layer precision to higher precision, for example int2 to int4, groupsize 128 to 64. \
            When this feature enabled, `--save` or `--save_safetensors` would be disable.')

    parser.add_argument('--calib-data-path', type=str, help="Path to calibration json file")

    parser.add_argument('--calib-iters', type=int, default=100, help="Number of samples for calibration")
    parser.add_argument("--quant-config-output", type=str, default="quant-params.json", help="Where to save quantization scales and zeros")
    parser.add_argument("--quantize-lm-head", action='store_true', help="Whether to quantize the lm_head (the output layer) in addition to transformer layers")
    parser.add_argument("--compression-factor", type=int, default=8, help="Compressoin factor for quantized weights")
    parser.add_argument("--compression-dim", choices=["K", "N"], default="K", help="Dimension along which to compress. Assumption: Layer weight shape: [out_features, in_features] ---> [N, K]")

    args = parser.parse_args()

    model = get_gptj(args.model)
    model.eval()
    calib_dataset = CNNDAILYMAIL(args.model, args.calib_data_path,is_calib=True,num_samples=args.calib_iters)
    dataloader=DataLoader(calib_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=calib_dataset.collate_batch
    )
    try:
        quantizers = {}
        quantizers = gptj_sequential(model, dataloader, DEV, quantizers)
        if args.quantize_lm_head:
            quantizers = quantize_lm_head(model, dataloader, DEV, quantizers )
        
        gptj_pack(model, quantizers, args.wbits, args.groupsize, quant_params_output=args.quant_config_output, compression_dim=args.compression_dim)
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        torch.save(model.state_dict(), args.save)
        print("Model saved at {}".format(args.save))
    except Exception as e:
        print("Quantization failed: {}".format(e))
