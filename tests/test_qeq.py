"""QEq solver (pure JAX)."""

import jax.numpy as jnp
import numpy as np

from expaloma.nn.qeq import charge_equilibrium


def test_charge_equilibrium_two_atoms():
    e = jnp.array([[0.1], [0.2]], dtype=jnp.float32)
    s = jnp.array([[1.0], [2.0]], dtype=jnp.float32)
    segment_ids = jnp.array([0, 0], dtype=jnp.int32)
    q = charge_equilibrium(e, s, segment_ids, 1, 0.0, q_ref=None)
    assert q.shape == (2, 1)
    assert np.allclose(float(q.sum()), 0.0, atol=1e-5)


def test_charge_equilibrium_batch():
    e = jnp.array([[0.1], [0.2], [0.3], [0.4]], dtype=jnp.float32)
    s = jnp.array([[1.0], [1.0], [2.0], [2.0]], dtype=jnp.float32)
    segment_ids = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
    tc = jnp.array([1.0, -1.0], dtype=jnp.float32)
    q = charge_equilibrium(e, s, segment_ids, 2, tc, q_ref=None)
    s0 = float(q[:2].sum())
    s1 = float(q[2:].sum())
    assert np.allclose(s0, 1.0, atol=1e-4)
    assert np.allclose(s1, -1.0, atol=1e-4)
