"""Capture real LowVramPatch FP8 weights and repeated-resolve residency on CUDA."""
import json, sys
from pathlib import Path
from types import SimpleNamespace
COMFY=Path("C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI")
sys.path.insert(0,str(COMFY))
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from comfy_kitchen.tensor import QuantizedTensor
import comfy.ops, comfy.utils
from comfy.quant_ops import TensorCoreFP8Layout
from comfy.model_patcher import LowVramPatch
from comfy.weight_adapter.lora import LoRAAdapter
OUT=Path(__file__).resolve().parents[3]/"target/edit2511/evidence"
OUT.mkdir(parents=True,exist_ok=True)
NAMES=["Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors", "Qwen-Image-Edit-2511-Object-Remover-v2-9200.safetensors", "qwen-image-edit-2511-multiple-angles-lora.safetensors"]
rows=[]
for label,module,adapters in [("attention","transformer_blocks.0.attn.to_q",[0]),("mlp","transformer_blocks.0.img_mlp.net.2",[0,1]),("modulation","transformer_blocks.1.img_mod.1",[2,1])]:
    seed_key="diffusion_model."+module;key=seed_key+".weight";seed=comfy.utils.string_to_seed(seed_key)
    with safe_open(COMFY/"models/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors",framework="pt",device="cuda") as f:
        q=f.get_tensor(module+".weight");scale=f.get_tensor(module+".weight_scale")
    tensors={"q":q.cpu().clone(),"scale":scale.cpu().clone()}
    weight=QuantizedTensor(q,"TensorCoreFP8Layout",TensorCoreFP8Layout.Params(scale=scale,orig_dtype=torch.bfloat16,orig_shape=tuple(q.shape)))
    original=weight.dequantize();reference=original.clone();patches=[]
    for i,adapter in enumerate(adapters):
        with safe_open(COMFY/"models/loras"/NAMES[adapter],framework="pt",device="cuda") as f:
            keys=set(f.keys());found=None
            for prefix in ["","diffusion_model.","transformer."]:
                for aa,bb in [(".lora_down.weight",".lora_up.weight"),(".lora_A.default.weight",".lora_B.default.weight"),(".lora_A.weight",".lora_B.weight")]:
                    if prefix+module+aa in keys: found=(prefix+module,aa,bb);break
                if found:break
            if not found:raise RuntimeError((module,adapter))
            stem,aa,bb=found;a=f.get_tensor(stem+aa);b=f.get_tensor(stem+bb)
            alpha=f.get_tensor(stem+".alpha").item() if stem+".alpha" in keys else None
        strength=1. if alpha is None else alpha/a.shape[0]
        reference=reference+torch.mm(b,a)*strength
        patches.append((1.,LoRAAdapter(set(),(b,a,alpha,None,None,None)),1.,None,None))
        tensors.update({f"a{i}":a.cpu(),f"b{i}":b.cpu(),f"strength{i}":torch.tensor(strength)})
    expected=weight.requantize_from_float(reference,scale="recalculate",stochastic_rounding=seed)
    tensors.update(expected_q=expected._qdata.cpu().clone(),expected_scale=expected._params.scale.cpu().clone(),expected_weight=expected.dequantize().cpu())
    x=torch.linspace(-.5,.5,8*q.shape[1],device="cuda",dtype=torch.float32).bfloat16().reshape(8,q.shape[1])
    tensors.update(input=x.cpu(),output=torch.nn.functional.linear(x,expected.dequantize()).cpu())
    patch=LowVramPatch(key,{key:patches});calls=[0]
    def apply(w):calls[0]+=1;return patch(w)
    interpret=comfy.memory_management.interpret_gathered_like
    comfy.memory_management.interpret_gathered_like=lambda *args:[weight,None]
    s=SimpleNamespace(weight=weight,bias=None,seed_key=seed_key,weight_lowvram_function=apply,weight_function=[],_prefetch={"resident":False,"signature":1,"xfer_dest":None,"cast_geometry":None,"needs_cast":False})
    try:
        first,_=comfy.ops.resolve_cast_module_with_vbar(s,torch.bfloat16,torch.device("cuda"),torch.bfloat16,torch.bfloat16,False)
        second,_=comfy.ops.resolve_cast_module_with_vbar(s,torch.bfloat16,torch.device("cuda"),torch.bfloat16,torch.bfloat16,False)
    finally:comfy.memory_management.interpret_gathered_like=interpret
    row=dict(module=module,seed=seed,patch_calls_for_two_resolves=calls[0],resident_bytes_match=torch.equal(weight._qdata,expected._qdata),resident_scale_match=torch.equal(weight._params.scale,expected._params.scale),first_forward_uses_bf16_patch=torch.equal(first,reference),second_forward_uses_requantized_weight=torch.equal(second,expected.dequantize()))
    rows.append(row);print(json.dumps(row),flush=True)
    save_file(tensors,OUT/("fp8-patch-"+label+".safetensors"))
(OUT/"fp8-probe.json").write_text(json.dumps(rows,indent=2))
