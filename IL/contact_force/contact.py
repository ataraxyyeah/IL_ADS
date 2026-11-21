"""Extract a fixed-size, object-directed contact force vector from mujoco-py."""
import warnings
import numpy as np

ROBOT_BODIES = ('leftclaw', 'rightclaw', 'hand')
SLOTS_PER_BODY = 4


def _is_descendant(model, body, root):
    while body > 0:
        if body == root:
            return True
        body = int(model.body_parentid[body])
    return body == root


class ContactReader:
    """Body-group slots; points sorted by position in target-body frame.

    Torque is deliberately omitted. Force is in world coordinates and points
    toward the target body. The slot ordering is deterministic, not a physical
    persistent point identity.
    """
    def __init__(self, sim, target_body='soccer_ball', threshold=0.01,
                 slots=SLOTS_PER_BODY):
        import mujoco_py
        self.mj = mujoco_py
        self.sim = sim
        self.target_body = target_body
        self.threshold = float(threshold)
        self.slots = int(slots)
        if self.threshold < 0 or self.slots <= 0:
            raise ValueError('threshold must be >= 0 and slots must be > 0')
        m = sim.model
        try:
            self.target = m.body_name2id(target_body)
            self.roots = [m.body_name2id(name) for name in ROBOT_BODIES]
        except Exception as exc:
            raise ValueError('Unknown target or robot body; inspect the task XML: '
                             f'{target_body}, {ROBOT_BODIES}') from exc
        self._warned = False

    def read(self):
        # Synchronize derived forces with the returned post-step state.
        self.sim.forward()
        model, data = self.sim.model, self.sim.data
        groups = [[] for _ in ROBOT_BODIES]
        for i in range(data.ncon):
            c = data.contact[i]
            if c.efc_address < 0:
                continue
            g1, g2 = int(c.geom1), int(c.geom2)
            b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
            t1 = _is_descendant(model, b1, self.target)
            t2 = _is_descendant(model, b2, self.target)
            if t1 == t2:
                continue
            robot_body = b2 if t1 else b1
            # Left/right include pad descendants. Hand slot excludes the claws.
            if _is_descendant(model, robot_body, self.roots[0]):
                group = 0
            elif _is_descendant(model, robot_body, self.roots[1]):
                group = 1
            elif robot_body == self.roots[2]:
                group = 2
            else:
                continue
            wrench = np.zeros(6, dtype=np.float64)
            self.mj.functions.mj_contactForce(model, data, i, wrench)
            # Contact axes are rows; force is exerted by geom1 on geom2.
            force_world = np.asarray(c.frame).reshape(3, 3).T @ wrench[:3]
            if t1:
                force_world = -force_world
            target_pos = np.asarray(data.body_xpos[self.target])
            target_rot = np.asarray(data.body_xmat[self.target]).reshape(3, 3)
            point_local = target_rot.T @ (np.asarray(c.pos) - target_pos)
            groups[group].append((tuple(point_local), force_world))
        vector = np.zeros((len(groups), self.slots, 3), dtype=np.float32)
        total_norm = 0.0
        for k, points in enumerate(groups):
            points.sort(key=lambda x: x[0])
            if len(points) > self.slots and not self._warned:
                warnings.warn('Contact slots overflow: points were truncated; increase force_slots.')
                self._warned = True
            for j, (_, force) in enumerate(points):
                total_norm += float(np.linalg.norm(force))
                if j < self.slots:
                    vector[k, j] = force
        return vector.reshape(-1), bool(total_norm > self.threshold)
