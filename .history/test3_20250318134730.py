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

def loss_function(
    params: jnp.ndarray, 
    probe: jnp.ndarray, 
    measurements: jnp.ndarray,
    beta: float = 0.01  # Phase smoothness regularization strength[^6]
) -> float:
    """Combine intensity error and phase smoothness constraints (similar to constraints in the tutorial)[^1][^3]"""
    # Calculate diffraction intensity residual
    predicted = forward_model(params, probe)
    intensity_error = jnp.mean((jnp.sqrt(predicted) - jnp.sqrt(measurements))**2)
    
    # Phase smoothness constraint
    phase = params[...,1]
    phase_x_grad = jnp.diff(phase, axis=-1)
    phase_y_grad = jnp.diff(phase, axis=-2)
    smoothness_loss = jnp.mean(phase_x_grad**2 + phase_y_grad**2)
    
    return intensity_error + beta * smoothness_loss


