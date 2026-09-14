"""Experimental nominal IK reference, isolated from live contact dynamics."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


class NominalJointReference:
    def __init__(self, live_solver, tracking_limit=0.35):
        from robosuite.utils.ik_utils import IKSolver

        if (
            len(live_solver.site_ids) != 1
            or live_solver.input_action_repr != "absolute"
            or live_solver.input_ref_frame != "world"
            or live_solver.input_rotation_repr != "axis_angle"
        ):
            raise ValueError("Nominal reference requires one absolute world axis-angle target")
        model = live_solver.full_model
        ids = live_solver.dof_ids
        if not (
            np.array_equal(model.jnt_qposadr[ids], ids)
            and np.array_equal(model.jnt_dofadr[ids], ids)
        ):
            raise ValueError(
                "Nominal reference requires scalar AIRBOT joints with matching addresses"
            )
        self.live_solver = live_solver
        self.tracking_limit = tracking_limit
        shadow = mujoco.MjData(model)
        mujoco.mj_copyData(shadow, model, live_solver.full_model_data)
        self.solver = IKSolver(
            model,
            shadow,
            {
                "joint_names": live_solver.joint_names,
                "end_effector_sites": live_solver.site_names,
                "nullspace_gains": live_solver.Kn.copy(),
            },
            damping=live_solver.damping,
            integration_dt=live_solver.integration_dt,
            max_dq=live_solver.max_dq,
            max_dq_torso=live_solver.max_dq_torso,
            input_action_repr="absolute",
            input_rotation_repr="axis_angle",
            input_ref_frame="world",
        )
        self.solver.q0 = live_solver.q0.copy()
        self.last_goal = shadow.qpos[ids].copy()
        self.velocity = np.zeros(len(ids))
        self.max_tracking_error = 0.0

    def __getattr__(self, name):
        return getattr(self.solver, name)

    def solve(self, action):
        solver = self.solver
        model, shadow = solver.full_model, solver.full_model_data
        mujoco.mj_kinematics(model, shadow)
        mujoco.mj_comPos(model, shadow)
        goal = solver.solve(action).copy()
        tracking_error = float(
            np.max(np.abs(goal - self.live_solver.full_model_data.qpos[solver.dof_ids]))
        )
        self.max_tracking_error = max(self.max_tracking_error, tracking_error)
        if tracking_error > self.tracking_limit:
            raise RuntimeError(
                f"Nominal reference tracking guard: {tracking_error:.6f} rad > {self.tracking_limit}"
            )
        shadow.qpos[solver.dof_ids] = goal
        mujoco.mj_kinematics(model, shadow)
        site = solver.site_ids[0]
        pose_error = np.linalg.norm(shadow.site_xpos[site] - np.asarray(action)[:3])
        angle = (
            Rotation.from_rotvec(np.asarray(action)[3:6]).inv()
            * Rotation.from_matrix(shadow.site_xmat[site].reshape(3, 3))
        ).magnitude()
        if pose_error > 0.003 or angle > np.deg2rad(2):
            raise RuntimeError(
                f"Nominal reference infeasible: {pose_error:.6f} m, {np.rad2deg(angle):.3f} deg"
            )
        self.velocity = (goal - self.last_goal) / solver.integration_dt
        self.last_goal = goal.copy()
        return goal
