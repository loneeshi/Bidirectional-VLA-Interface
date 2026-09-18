"""Action-only masked microgradients; one optimizer update per accumulation."""
import dataclasses


def masked_action_mean(values, valid, xp):
    # The model already averages over its 32 action dimensions. Only time is masked.
    weights = valid.astype(values.dtype)
    per_sample = xp.sum(values * weights, axis=-1) / xp.maximum(xp.sum(weights, axis=-1), 1)
    return xp.mean(per_sample)


@dataclasses.dataclass(frozen=True)
class SampleSchedule:
    base: object
    accumulation: int = 8

    def create(self):
        schedule = self.base.create()
        # Optimizer count k starts after exactly 8*k processed samples.
        return lambda k: schedule(k * self.accumulation)


def micro_gradient(config, state, key, batch):
    import flax.nnx as nnx
    import jax.numpy as jnp
    from scripts.train import _compute_action_and_progress_with_mode
    model = nnx.merge(state.model_def, state.params)
    model.train()
    obs, actions, valid = batch

    def loss(model):
        values, _ = _compute_action_and_progress_with_mode(model, config, key, obs, actions, train=True)
        return masked_action_mean(values, valid, jnp)

    return nnx.value_and_grad(loss, argnums=nnx.DiffState(0, config.trainable_filter))(model)


def apply_average(config, state, summed_grads, count):
    import flax.nnx as nnx
    import jax
    import optax
    model = nnx.merge(state.model_def, state.params)
    grads = jax.tree.map(lambda g: g / count, summed_grads)
    params = state.params.filter(config.trainable_filter)
    updates, opt_state = state.tx.update(grads, state.opt_state, params)
    nnx.update(model, optax.apply_updates(params, updates))
    if state.ema_decay is not None:
        raise ValueError('S1-IA fixes EMA off')
    return dataclasses.replace(state, step=state.step+1, params=nnx.state(model), opt_state=opt_state), optax.global_norm(grads)


def evaluate(config, state, key, batch):
    import flax.nnx as nnx
    import jax.numpy as jnp
    from scripts.train import _compute_action_and_progress_with_mode
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    obs, actions, valid = batch
    values, _ = _compute_action_and_progress_with_mode(model, config, key, obs, actions, train=False)
    return masked_action_mean(values, valid, jnp)
