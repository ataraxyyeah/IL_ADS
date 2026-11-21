"""Two independent local OT problems, one per phase."""
import numpy as np
import torch
from rewarder import optimal_transport_plan, cosine_distance


def sample_post_repeat(values, repeat):
    """Select the end of each repeated-action group, including a partial last group."""
    if repeat < 1 or len(values) == 0:
        raise ValueError('repeat must be positive and values must be nonempty')
    indexes = np.minimum(np.arange(0, len(values), repeat) + repeat - 1, len(values) - 1)
    return np.asarray(values)[indexes]


def first_contact(flags):
    indexes = np.flatnonzero(np.asarray(flags, dtype=bool))
    return int(indexes[0]) if indexes.size else len(flags)


class StagedOT:
    def __init__(self, agent, expert_force, expert_flags, force_scale=1.0):
        self.agent = agent
        self.force_scale = float(force_scale)
        if self.force_scale <= 0:
            raise ValueError('force_scale must be positive')
        self.expert_force = [np.asarray(x, np.float32) for x in expert_force]
        self.expert_flags = [np.asarray(x, bool) for x in expert_flags]
        if len(agent.demos) != len(self.expert_force):
            raise ValueError('Expert visual and force demo counts differ')
        for z, f, flags in zip(agent.demos, self.expert_force, self.expert_flags):
            if len(z) != len(f) or len(f) != len(flags):
                raise ValueError('Expert video/force/contact length mismatch')
            if first_contact(flags) == len(flags):
                raise ValueError('Expert demonstration has no target contact after sampling')

    def _solve(self, x, y, metric, total_len):
        if not len(x):
            return np.empty(0, np.float32)
        if not len(y):
            raise ValueError('Expert stage has no samples')
        if metric == 'visual':
            cost = cosine_distance(x, y)
        else:
            # Euclidean distance preserves force magnitude; avoids undefined
            # cosine distance on all-zero vectors after contact is lost.
            cost = torch.cdist(x / self.force_scale, y / self.force_scale, p=2)
        if not torch.isfinite(cost).all():
            raise ValueError('Non-finite OT cost')
        # Log-domain Sinkhorn avoids underflow for physical force distances.
        import ot
        cm = cost.detach().cpu().numpy().astype(np.float64)
        plan_np = ot.bregman.sinkhorn_log(
            np.full(len(x), 1.0 / len(x)), np.full(len(y), 1.0 / len(y)),
            cm, reg=0.01, numItermax=1000, stopThr=1e-7, warn=False)
        if not np.isfinite(plan_np).all() or plan_np.sum() < 0.99:
            raise ValueError('Sinkhorn failed; inspect force_scale and cost values')
        plan = torch.as_tensor(plan_np, device=x.device, dtype=x.dtype)
        # Local OT row mass is 1/len(x); rescale to original global 1/T.
        values = -(plan * cost).sum(dim=1) * (len(x) / total_len)
        return values.detach().cpu().numpy()

    def reward(self, images, forces, flags):
        n = len(images)
        if n == 0 or len(forces) != n or len(flags) != n:
            raise ValueError('Agent video/force/contact lengths differ or are empty')
        with torch.no_grad():
            z = self.agent.cost_encoder(torch.as_tensor(images, device=self.agent.device).float())
        f = torch.as_tensor(forces, device=self.agent.device, dtype=torch.float32)
        boundary = first_contact(flags)
        best = None
        for expert_z, expert_force, expert_flags in zip(
                self.agent.demos, self.expert_force, self.expert_flags):
            ez = torch.as_tensor(expert_z, device=self.agent.device).float()
            ef = torch.as_tensor(expert_force, device=self.agent.device, dtype=torch.float32)
            split = first_contact(expert_flags)
            # If expert starts in contact, its pre-contact segment is empty.
            if boundary and split == 0:
                continue
            pre = self._solve(z[:boundary], ez[:split], 'visual', n)
            post = self._solve(f[boundary:], ef[split:], 'force', n)
            result = np.concatenate([pre, post]) * self.agent.sinkhorn_rew_scale
            if best is None or result.sum() > best.sum():
                best = result
        if best is None:
            raise ValueError('No expert has a pre-contact segment to match this rollout')
        return best, boundary
