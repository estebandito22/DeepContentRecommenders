"""Class that impements the SWATS optimization algorithm."""

import logging
import sys
import math
import torch
from torch.optim import Optimizer

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class Swats(Optimizer):

    """
    Implements Swats algorithm.

    Adapted from the PyTorch Adam implmentation.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
        epsilon (float, opitional): e for switching to SGD (default: 1e-9)

    .. _Adam: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ

    .. Improving Generalization Performance by Switching from Adam to SGD
       Nitish Shirish Keskar, Richard Socher
       https://arxiv.org/abs/1712.07628
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, epsilon=1e-9):
        """Init SWATS optimizer."""
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        lk=0, phase='ADAM', epsilon=epsilon,
                        momentum=betas[0])
        super(Swats, self).__init__(params, defaults)

    def __setstate__(self, state):
        """Set optimizer state."""
        super(Swats, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Perform a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, \
                        please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # take an SGD step and skip ADAM steps
                if group['phase'] == 'SGD':
                    if p.grad is None:
                        continue
                    d_p = p.grad.data
                    if group['weight_decay'] != 0:
                        d_p.add_(group['weight_decay'], p.data)
                    if group['momentum'] != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = \
                                torch.zeros_like(p.data)
                            buf.mul_(group['momentum']).add_(d_p)
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(group['momentum']).add_(d_p)
                        d_p = (1 - group['momentum']) * buf

                    p.data.add_(-group['lr'], d_p)
                    continue

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad.
                        # values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg.
                    # till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(
                    bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

                # check if its time to switch to SGD and compute SGD lr
                pk = -step_size * (exp_avg / denom)
                pk = pk.view(1, -1).squeeze()
                if pk.dot(grad.view(1, -1).squeeze()) != 0:
                    yk = torch.div(
                        pk.dot(pk), -pk.dot(grad.view(1, -1).squeeze()))
                    group['lk'] = beta2 * group['lk'] + (1 - beta2) * yk
                    switch_test = torch.abs(
                        torch.div(group['lk'], 1 - beta2 ** state['step']) - yk)
                    if state['step'] > 1 and switch_test < group['epsilon']:
                        logging.info(
                            "Switching from ADAM to SGD at step %s",
                            state['step'])
                        group['phase'] = 'SGD'
                        group['lr'] = torch.div(
                            group['lk'], 1 - beta2 ** state['step'])

        return loss
