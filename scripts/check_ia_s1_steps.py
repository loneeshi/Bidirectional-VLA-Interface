"""CPU numerical checks of the exact masked-gradient and optimizer helpers."""
import dataclasses
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu')
import sys
from pathlib import Path
import openpi
sys.path.insert(0, str(Path(openpi.__file__).resolve().parents[2]))
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from types import SimpleNamespace
from ia_s1_steps import micro_gradient, apply_average, masked_action_mean, SampleSchedule


class Toy(nnx.Module):
    def __init__(self):
        self.weight = nnx.Param(jnp.array(0.5))

    def compute_action_and_progress_chunk_prefix(self, rng, obs, actions, train):
        return (self.weight.value*obs-actions)**2, None


@dataclasses.dataclass
class State:
    model_def: object
    params: object
    tx: object
    opt_state: object
    step: int = 0
    ema_decay: object = None


cfg=SimpleNamespace(trainable_filter=nnx.Param,progress_readout_mode='chunk_prefix')
graph,params=nnx.split(Toy())
tx=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(1e-3))
state=State(graph,params,tx,tx.init(params))
for n in (8,3):
    obs=jnp.arange(1,n*3+1,dtype=jnp.float32).reshape(n,3)/10
    target=jnp.ones_like(obs)*.7
    valid=jnp.broadcast_to(jnp.array([True,True,False]),obs.shape)
    _,full=micro_gradient(cfg,state,jax.random.key(0),(obs,target,valid))
    summed=None
    for i in range(n):
        _,g=micro_gradient(cfg,state,jax.random.key(i),(obs[i:i+1],target[i:i+1],valid[i:i+1]))
        summed=g if summed is None else jax.tree.map(lambda x,y:x+y,summed,g)
    accumulated,_=apply_average(cfg,state,summed,n)
    reference,_=apply_average(cfg,state,full,1)
    for x,y in zip(jax.tree.leaves((accumulated.params,accumulated.opt_state)),jax.tree.leaves((reference.params,reference.opt_state))):
        np.testing.assert_allclose(x,y,rtol=1e-5,atol=1e-7)
    assert accumulated.step==1
    _,g1=micro_gradient(cfg,state,jax.random.key(0),(obs,target,valid))
    _,g2=micro_gradient(cfg,state,jax.random.key(0),(obs,target.at[:,-1].set(999),valid))
    for x,y in zip(jax.tree.leaves(g1),jax.tree.leaves(g2)):np.testing.assert_array_equal(x,y)
    _,g3=micro_gradient(cfg,state,jax.random.key(0),(obs,target.at[:,0].set(9),valid))
    assert any(not np.array_equal(x,y) for x,y in zip(jax.tree.leaves(g1),jax.tree.leaves(g3)))
weights=jnp.array([[1.,9.,999.],[3.,999.,999.]])
mask=jnp.array([[1,1,0],[1,0,0]],dtype=bool)
assert float(masked_action_mean(weights,mask,jnp))==4.
class Clock:
    def create(self):return lambda x:x
assert SampleSchedule(Clock()).create()(854)==6832
print('{"status":"passed","device":"cpu","checks":["masked_gradient","valid_gradient","batch8_equivalence","tail3_equivalence","one_optimizer_update","sample_lr_clock","per_sample_mask_mean"]}')
