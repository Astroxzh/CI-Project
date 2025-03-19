import jax.numpy as jnp


def complex_parameterization(params: jnp.ndarray) -> jnp.ndarray:
    """Convert dual-channel parameters [log_amp, phase] into complex fields (simulating CDTools' complex processing logic)[^3]"""
    log_amp, phase = params[...,0], params[...,1]
    return jnp.exp(log_amp) * jnp.exp(1j * phase)  # Amplitude > 0
