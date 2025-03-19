import jax.numpy as jnp


def complex_parameterization(params: jnp.ndarray) -> jnp.ndarray:
    """Convert dual-channel parameters [log_amp, phase] into complex fields (simulating CDTools' complex processing logic)[^3]"""
    log_amp, phase = params[...,0], params[...,1]
    return jnp.exp(log_amp) * jnp.exp(1j * phase)  # Amplitude > 0

def scaled_fft(field: jnp.ndarray) -> jnp.ndarray:
    """Forward far-field propagation simulation (replacing CDTools' propagators.far_field) [^6]"""
    scaling = 1.0 / jnp.sqrt(jnp.prod(field.shape[-2:]))
    return scaling * jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.ifftshift(field)))

def forward_model(params: jnp.ndarray, probe: jnp.ndarray) -> jnp.ndarray:
    """Complex-to-real conversion and diffraction intensity calculation (consistent with neutron wave diffraction physics logic in the tutorial)[^3]"""
    obj = complex_parameterization(params)
    exit_wave = probe[0] * obj  # Assume probe is multimodal [^2]
    diff_field = scaled_fft(exit_wave)
    return jnp.abs(diff_field)**2  # Light intensity calculation



