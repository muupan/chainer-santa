import numpy

from chainer import cuda
from chainer import optimizer


class Santa(optimizer.GradientMethod):

    """Santa with SSS.

    See: http://arxiv.org/abs/1512.07962v1

    """

    def __init__(self, eta=1e-6, sigma=0.95, eps=1e-8, C=0.5, gamma=0.5,
                 burnin=1000):
        self.eta = eta
        self.sigma = sigma
        self.eps = eps
        self.C = C
        self.gamma = gamma
        self.burnin = burnin

    def init_state(self, param, state):
        xp = cuda.get_array_module(param.data)
        state['v'] = xp.zeros_like(param.data, dtype=xp.float32)
        state['u'] = xp.zeros_like(param.data, dtype=xp.float32)
        state['alpha'] = xp.full_like(param.data, self.C, dtype=xp.float32)

    def update_one_cpu(self, param, state):
        v, alpha, u = state['v'], state['alpha'], state['u']
        grad = param.grad
        v *= self.sigma
        v += (1 - self.sigma) * grad * grad

        g = 1 / numpy.sqrt(numpy.sqrt(v) + self.eps)
        param.data += g * u / 2
        if self.t < self.burnin:
            # exploration
            inv_beta = 1.0 / self.t ** self.gamma
            alpha += (u * u - self.eta * inv_beta) / 2
            u *= numpy.exp(-alpha / 2)
            u += - g * grad * self.eta + \
                numpy.sqrt(2 * v * self.eta * inv_beta) * \
                numpy.random.normal(size=param.data.shape)
            u *= numpy.exp(-alpha / 2)
            alpha += (u * u - self.eta * inv_beta) / 2
        else:
            # refinement
            u *= numpy.exp(-alpha / 2)
            u -= g * grad * self.eta
            u *= numpy.exp(-alpha / 2)
        param.data += g * u / 2

    def update_one_gpu(self, param, state):
        xp = cuda.get_array_module(param.data)
        g = xp.empty(param.data.shape, dtype=xp.float32)

        cuda.elementwise(
            'T grad, T sigma, T eps, T u',
            'T data, T v, T g',
            '''v *= sigma;
               v += (1 - sigma) * grad * grad;
               g = 1 / sqrt(sqrt(v) + eps);
               data += g * u / 2;''',
            'santa_pre')(param.grad, self.sigma, self.eps, state['u'],
                         param.data, state['v'], g)
        if self.t < self.burnin:
            # exploration
            zeta = xp.random.normal(size=param.data.shape, dtype=xp.float32)
            inv_beta = 1.0 / self.t ** self.gamma
            cuda.elementwise(
                'T v, T inv_beta, T eta, T g, T zeta, T grad',
                'T alpha, T u',
                '''alpha += (u * u - eta * inv_beta) / 2;
                   u *= exp(-alpha/2);
                   u += -g * grad * eta + sqrt(2 * v * eta * inv_beta) * zeta;
                   u *= exp(-alpha/2);
                   alpha += (u * u - eta * inv_beta)/2;
                ''',
                'santa_exploration')(
                    state['v'], inv_beta, self.eta, g, zeta, param.grad,
                    state['alpha'], state['u'])
        else:
            # refinement
            cuda.elementwise(
                'T alpha, T g, T grad, T eta',
                'T u',
                '''u *= exp(-alpha/2);
                   u -= g * grad * eta;
                   u *= exp(-alpha/2);''',
                'santa_refinement')(
                    state['alpha'], g, param.grad, self.eta, state['u'])
        param.data += g * state['u'] / 2


class SantaE(Santa):

    """Santa with Euler scheme.

    See: http://arxiv.org/abs/1512.07962v1

    """

    def update_one_cpu(self, param, state):
        v, alpha, u = state['v'], state['alpha'], state['u']
        grad = param.grad
        v *= self.sigma
        v += (1 - self.sigma) * grad * grad

        g = 1 / numpy.sqrt(numpy.sqrt(v) + self.eps)

        if self.t < self.burnin:
            # exploration
            inv_beta = 1.0 / self.t ** self.gamma
            alpha += (u * u - self.eta * inv_beta)
            u *= 1 - alpha
            u += - g * grad * self.eta + \
                numpy.sqrt(2 * v * self.eta * inv_beta) * \
                numpy.random.normal(size=param.data.shape)
        else:
            # refinement
            u *= 1 - alpha
            u -= g * grad * self.eta
        param.data += g * u

    def update_one_gpu(self, param, state):
        xp = cuda.get_array_module(param.data)
        g = xp.empty(param.data.shape, dtype=xp.float32)

        cuda.elementwise(
            'T grad, T sigma, T eps, T u',
            'T v, T g',
            '''v *= sigma;
               v += (1 - sigma) * grad * grad;
               g = 1 / sqrt(sqrt(v) + eps);''',
            'santa_e_pre')(param.grad, self.sigma, self.eps, state['u'],
                           state['v'], g)
        if self.t < self.burnin:
            # exploration
            zeta = xp.random.normal(size=param.data.shape, dtype=xp.float32)
            inv_beta = 1.0 / self.t ** self.gamma
            cuda.elementwise(
                'T v, T inv_beta, T eta, T g, T zeta, T grad',
                'T alpha, T u',
                '''alpha += (u * u - eta * inv_beta);
                   u *= 1 - alpha;
                   u += -g * grad * eta + sqrt(2 * v * eta * inv_beta) * zeta;
                ''',
                'santa_e_exploration')(
                    state['v'], inv_beta, self.eta, g, zeta, param.grad,
                    state['alpha'], state['u'])
        else:
            # refinement
            cuda.elementwise(
                'T alpha, T g, T grad, T eta',
                'T u',
                '''u *= 1 - alpha;
                   u -= g * grad * eta;''',
                'santa_e_refinement')(
                    state['alpha'], g, param.grad, self.eta, state['u'])
        param.data += g * state['u']
