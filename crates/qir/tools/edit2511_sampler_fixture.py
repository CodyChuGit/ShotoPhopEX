"""Small CPU oracle from installed ComfyUI sampler classes, with fixed velocities."""
import sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,'C:/Users/5090/ComfyUI/ComfyUI_windows_portable/ComfyUI')
sys.argv=[sys.argv[0],'--cpu']
import torch
from safetensors.torch import save_file
from comfy.model_sampling import CONST, ModelSamplingDiscreteFlow
from comfy.samplers import KSamplerX0Inpaint, simple_scheduler
from comfy_extras.nodes_cfg import CFGNorm
class Sampling(ModelSamplingDiscreteFlow,CONST):pass
sampling=Sampling(SimpleNamespace(sampling_settings={'shift':3.1,'multiplier':1.0}))
sigmas=simple_scheduler(sampling,8)
class Patch:
    def clone(self):return self
    def set_model_sampler_post_cfg_function(self,fn):self.fn=fn
patch=Patch();CFGNorm.execute(patch,1.0)
shape=(1,16,1,3,5);torch.manual_seed(7)
noise=torch.randn(shape);target=torch.linspace(-1.,1.,240).reshape(shape)
mask=torch.linspace(0.,1.,15).reshape(1,1,1,3,5).repeat(1,16,1,1,1)
pos=torch.linspace(-.7,.3,240).reshape(shape);neg=torch.linspace(.4,-.8,240).reshape(shape)
values=dict(sigmas=sigmas,noise=noise,target=target,mask=mask,positive=pos,negative=neg)
class Model:
    inner_model=SimpleNamespace(scale_latent_inpaint=lambda **kw:sampling.noise_scaling(kw['sigma'],kw['noise'],kw['latent_image']))
    def __call__(self,x,sigma,**kw):
        values[f'step_{self.step}.input']=x.clone()
        p=sampling.calculate_denoised(sigma,pos,x);n=sampling.calculate_denoised(sigma,neg,x)
        return patch.fn(dict(cond_denoised=p,denoised=n+(p-n)*4.))
model=Model();wrapped=KSamplerX0Inpaint(model,sigmas);wrapped.noise=noise;wrapped.latent_image=target
x=sampling.noise_scaling(sigmas[0:1],noise,target);values['initial']=x.clone()
for i in range(8):
    model.step=i;denoised=wrapped(x,sigmas[i:i+1],mask)
    values[f'step_{i}.denoised']=denoised.clone()
    x=x+((x-denoised)/sigmas[i])*(sigmas[i+1]-sigmas[i]);values[f'step_{i}.latent']=x.clone()
save_file(values,ROOT/'crates/qir/tests/fixtures/edit2511_sampler.safetensors')
