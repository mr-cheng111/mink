#!/usr/bin/env python3
"""Build a unified dual-arm ARM620 robot with a shared base at (0, 0, 0).

This builder creates one shared root body (`base_link`) at world origin,
and mounts left/right arms under it.
"""

from pathlib import Path

import mujoco

_HERE = Path(__file__).parent
_SINGLE_ARM_XML = _HERE / "arm620.xml"
_ROT_Z_90_QUAT = [0.70710678, 0.0, 0.0, 0.70710678]


def _prepare_arm_spec(prefix: str) -> mujoco.MjSpec:
    arm_spec = mujoco.MjSpec.from_file(_SINGLE_ARM_XML.as_posix())
    arm_spec.modelname = f"{prefix[:-1]}_arm620"

    # Remove scene-level elements from single-arm model.
    try:
        arm_spec.delete(arm_spec.geom("floor"))
    except Exception:
        pass

    for light in list(arm_spec.worldbody.lights):
        arm_spec.delete(light)

    if arm_spec.keys:
        try:
            arm_spec.delete(arm_spec.key("home"))
        except Exception:
            pass

    return arm_spec


def _build_dual_unified_spec(y_offset: float = 0.28) -> mujoco.MjSpec:
    root = mujoco.MjSpec()
    root.modelname = "dual_arm620_unified_base"

    root.option.timestep = 0.001
    root.visual.global_.azimuth = 120
    root.visual.global_.elevation = -20

    root.worldbody.add_light(
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[0.5, 0.5, 0.5],
        specular=[0.1, 0.1, 0.1],
        pos=[0.0, 0.0, 0.0],
        dir=[0.0, 0.0, -1.0],
    )

    # Shared robot base at world origin.
    base_link = root.worldbody.add_body(name="base_link", pos=[0.0, 0.0, 0.0])
    # Visualize shared base frame axes (X-red, Y-green, Z-blue).
    base_link.add_site(name="base_link_frame", pos=[0.0, 0.0, 0.0], size=[0.001], group=5)
    base_link.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.06, 0.0, 0.0],
        size=[0.06, 0.003, 0.003],
        contype=0,
        conaffinity=0,
        rgba=[1.0, 0.15, 0.15, 0.95],
    )
    base_link.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.0, 0.06, 0.0],
        size=[0.003, 0.06, 0.003],
        contype=0,
        conaffinity=0,
        rgba=[0.15, 1.0, 0.15, 0.95],
    )
    base_link.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.0, 0.0, 0.06],
        size=[0.003, 0.003, 0.06],
        contype=0,
        conaffinity=0,
        rgba=[0.15, 0.45, 1.0, 0.95],
    )

    left_mount = base_link.add_site(
        name="left_mount",
        pos=[0.0, y_offset, 0.0],
        quat=_ROT_Z_90_QUAT,
        group=5,
    )
    right_mount = base_link.add_site(
        name="right_mount",
        pos=[0.0, -y_offset, 0.0],
        quat=_ROT_Z_90_QUAT,
        group=5,
    )

    for prefix, mount_site in (("left/", left_mount), ("right/", right_mount)):
        arm_spec = _prepare_arm_spec(prefix)
        root.attach(arm_spec, site=mount_site, prefix=prefix)

    # Optional floor.
    root.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[2.0, 2.0, 0.05],
        material="left/groundplane",
        condim=3,
    )

    # IK mocap targets.
    # Target markers: do not bake fixed initial XYZ here.
    # They will be initialized dynamically from current */tools_link pose at runtime.
    left_target = root.worldbody.add_body(name="left_target", mocap=True, pos=[0.0, 0.0, 0.0])
    left_target.add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.018],
        contype=0,
        conaffinity=0,
        rgba=[0.95, 0.25, 0.25, 0.9],
    )
    left_target.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.03, 0.0, 0.0],
        size=[0.03, 0.002, 0.002],
        contype=0,
        conaffinity=0,
        rgba=[1.0, 0.1, 0.1, 0.95],
    )
    left_target.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.0, 0.03, 0.0],
        size=[0.002, 0.03, 0.002],
        contype=0,
        conaffinity=0,
        rgba=[0.1, 1.0, 0.1, 0.95],
    )
    left_target.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.0, 0.0, 0.03],
        size=[0.002, 0.002, 0.03],
        contype=0,
        conaffinity=0,
        rgba=[0.1, 0.4, 1.0, 0.95],
    )
    left_target.add_site(name="left_target_site", size=[0.001])

    right_target = root.worldbody.add_body(name="right_target", mocap=True, pos=[0.0, 0.0, 0.0])
    right_target.add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.018],
        contype=0,
        conaffinity=0,
        rgba=[0.25, 0.35, 0.95, 0.9],
    )
    right_target.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.03, 0.0, 0.0],
        size=[0.03, 0.002, 0.002],
        contype=0,
        conaffinity=0,
        rgba=[1.0, 0.1, 0.1, 0.95],
    )
    right_target.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.0, 0.03, 0.0],
        size=[0.002, 0.03, 0.002],
        contype=0,
        conaffinity=0,
        rgba=[0.1, 1.0, 0.1, 0.95],
    )
    right_target.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.0, 0.0, 0.03],
        size=[0.002, 0.002, 0.03],
        contype=0,
        conaffinity=0,
        rgba=[0.1, 0.4, 1.0, 0.95],
    )
    right_target.add_site(name="right_target_site", size=[0.001])

    # Goal markers used by telegrip overlays.
    left_goal = root.worldbody.add_body(name="left_goal", mocap=True, pos=[0.0, 0.0, 0.0])
    left_goal.add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.015],
        contype=0,
        conaffinity=0,
        rgba=[0.2, 1.0, 0.35, 0.85],
    )

    right_goal = root.worldbody.add_body(name="right_goal", mocap=True, pos=[0.0, 0.0, 0.0])
    right_goal.add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.015],
        contype=0,
        conaffinity=0,
        rgba=[0.2, 1.0, 1.0, 0.85],
    )

    return root


def build_dual_unified_model(y_offset: float = 0.28) -> mujoco.MjModel:
    return _build_dual_unified_spec(y_offset=y_offset).compile()


def build_dual_unified_data(y_offset: float = 0.28) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = build_dual_unified_model(y_offset=y_offset)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def save_dual_unified_xml(output_path: str | Path, y_offset: float = 0.28) -> Path:
    root = _build_dual_unified_spec(y_offset=y_offset)
    output = Path(output_path)
    xml_text = root.to_xml()
    xml_text = xml_text.replace('file="arm620/', 'file="meshes/arm620/')
    xml_text = xml_text.replace('file="robotiq/', 'file="meshes/robotiq/')
    output.write_text(xml_text)
    return output
