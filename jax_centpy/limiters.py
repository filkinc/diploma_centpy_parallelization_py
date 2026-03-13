import jax.numpy as jnp


def minmod(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * (jnp.sign(a) + jnp.sign(b)) * jnp.minimum(jnp.abs(a), jnp.abs(b))


def superbee(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    def _mm(x, y):
        return 0.5 * (jnp.sign(x) + jnp.sign(y)) * jnp.minimum(jnp.abs(x), jnp.abs(y))

    sb1 = _mm(a, 2.0 * b)
    sb2 = _mm(2.0 * a, b)

    return jnp.where(jnp.abs(sb1) > jnp.abs(sb2), sb1, sb2)


def monotonized_central(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    central = 0.5 * (a + b)
    sign_match = (jnp.sign(a) * jnp.sign(b)) > 0

    abs_min = jnp.minimum(jnp.abs(2 * a), jnp.abs(2 * b))
    abs_min = jnp.minimum(abs_min, jnp.abs(central))

    return jnp.where(sign_match, jnp.sign(a) * abs_min, 0.0)


def van_leer(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    eps = 1e-10
    return (a * jnp.abs(b) + jnp.abs(a) * b) / (jnp.abs(a) + jnp.abs(b) + eps)

