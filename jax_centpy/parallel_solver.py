
import jax
import jax.numpy as jnp
import numpy as np

from core import Pars1d, Equation1d
import schemes
import limiters
import time_integration


class ParallelSolver1d:
    def __init__(self, pars: Pars1d, eqn: Equation1d, scheme_name="sd2", limiter_name="minmod"):
        self.pars = pars
        self.eqn = eqn
        self.num_devices = jax.device_count()

        if pars.J % self.num_devices != 0:
            raise ValueError(f"Grid size J={pars.J} must be divisible by device count {self.num_devices}")

        self.J_sub = pars.J // self.num_devices

        limiter_map = {"minmod": limiters.minmod, "mc": limiters.monotonized_central}
        self.limiter = limiter_map.get(limiter_name, limiters.minmod)

        def halo_exchange(u_sub):
            left_edge = u_sub[:2]
            right_edge = u_sub[-2:]

            recv_from_left = jax.lax.ppermute(right_edge, axis_name='i',
                                              perm=[(i, (i + 1) % self.num_devices) for i in range(self.num_devices)])

            recv_from_right = jax.lax.ppermute(left_edge, axis_name='i',
                                               perm=[(i, (i - 1) % self.num_devices) for i in range(self.num_devices)])

            u_padded = jnp.concatenate([recv_from_left, u_sub, recv_from_right])
            return u_padded

        def local_rhs(t, u_sub):
            u_padded = halo_exchange(u_sub)

            u_R_all, u_L_all = schemes.reconstruction_sd2(u_padded, limiter=self.limiter)

            u_minus = u_R_all[:-1]
            u_plus = u_L_all[1:]

            a = jnp.maximum(eqn.spectral_radius(u_minus), eqn.spectral_radius(u_plus))

            f_minus = eqn.flux(u_minus)
            f_plus = eqn.flux(u_plus)

            flux = 0.5 * (f_minus + f_plus - a * (u_plus - u_minus))

            flux_diff = flux[1:] - flux[:-1]
            rhs = -flux_diff / pars.dx

            return rhs

        def update_step_pmap(t, u_sub, dt):
            return time_integration.step_ssp_rk3(t, u_sub, dt, local_rhs)

        self.parallel_update = jax.pmap(update_step_pmap, axis_name='i', in_axes=(None, 0, None))

        def compute_local_dt(u_sub):
            max_speed = jnp.max(eqn.spectral_radius(u_sub))
            global_max_speed = jax.lax.pmax(max_speed, axis_name='i')
            safe_speed = jnp.maximum(global_max_speed, 1e-6)
            return pars.cfl * pars.dx / safe_speed

        self.parallel_dt = jax.pmap(compute_local_dt, axis_name='i')

    def solve(self):
        dx = self.pars.dx
        x_global = jnp.linspace(self.pars.x_init + dx / 2, self.pars.x_final - dx / 2, self.pars.J)
        u_global = self.eqn.initial_data(x_global)

        u_sharded = u_global.reshape((self.num_devices, self.J_sub))

        u_devices = jax.device_put_sharded(list(u_sharded), jax.devices())

        t = 0.0
        results = {'t': [t], 'u_n': [u_global]}  # Сохраняем склеенное
        next_out = self.pars.dt_out

        print(f"Parallel Solver: {self.num_devices} devices x {self.J_sub} points")

        while t < self.pars.t_final:
            dt_array = self.parallel_dt(u_devices)
            dt = float(dt_array[0])

            if t + dt > next_out: dt = next_out - t
            if t + dt > self.pars.t_final: dt = self.pars.t_final - t

            u_devices = self.parallel_update(t, u_devices, dt)

            t += dt

            if t >= next_out - 1e-9:
                u_cpu = np.array(u_devices).flatten()  # Работает как gather
                results['u_n'].append(u_cpu)
                results['t'].append(t)
                next_out += self.pars.dt_out

        results['x'] = x_global
        return results
