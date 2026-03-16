#!/usr/bin/env python3
"""Build a dual-ARM620 MuJoCo scene from the single-arm model.

This avoids MJCF include conflicts from loading the same file twice in a static XML.
"""

from pathlib import Path

import mujoco

_HERE = Path(__file__).parent
_SINGLE_ARM_XML = _HERE / "arm620.xml"
_ROT_Z_90_QUAT = [0.70710678, 0.0, 0.0, 0.70710678]


def _build_dual_arm620_spec(y_offset: float = 0.28) -> mujoco.MjSpec:
    """Create dual-arm scene spec before compilation."""
    root = mujoco.MjSpec()
    root.modelname = "dual_arm620_scene"

    # Keep scene style close to examples/arm620/scene.xml
    root.option.timestep = 0.001
    root.visual.global_.azimuth = 120
    root.visual.global_.elevation = -20
    root.visual.headlight.diffuse[:] = (0.6, 0.6, 0.6)
    root.visual.headlight.ambient[:] = (0.1, 0.1, 0.1)
    root.visual.headlight.specular[:] = (0.0, 0.0, 0.0)

    root.worldbody.add_light(
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[0.5, 0.5, 0.5],
        specular=[0.1, 0.1, 0.1],
        pos=[0.0, 0.0, 0.0],
        dir=[0.0, 0.0, -1.0],
    )

    root.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[2.0, 2.0, 0.05],
        material="left/groundplane",
        condim=3,
    )

    left_mount = root.worldbody.add_site(
        name="left_mount",
        pos=[0.0, y_offset, 0.0],
        quat=_ROT_Z_90_QUAT,
        group=5,
    )
    right_mount = root.worldbody.add_site(
        name="right_mount",
        pos=[0.0, -y_offset, 0.0],
        quat=_ROT_Z_90_QUAT,
        group=5,
    )

    for prefix, mount_site in (("left/", left_mount), ("right/", right_mount)):
        arm_spec = _prepare_arm_spec(prefix)
        root.attach(arm_spec, site=mount_site, prefix=prefix)

    # Per-arm mocap targets for IK
    left_target = root.worldbody.add_body(name="left_target", mocap=True, pos=[0.4, y_offset, 0.4])
    left_target.add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.018],
        contype=0,
        conaffinity=0,
        rgba=[0.95, 0.25, 0.25, 0.9],
    )
    # XYZ axes (robot/world frame): X-red, Y-green, Z-blue
    left_target.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.03, 0.0, 0.0], size=[0.03, 0.002, 0.002], contype=0, conaffinity=0, rgba=[1.0, 0.1, 0.1, 0.95])
    left_target.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.0, 0.03, 0.0], size=[0.002, 0.03, 0.002], contype=0, conaffinity=0, rgba=[0.1, 1.0, 0.1, 0.95])
    left_target.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.0, 0.0, 0.03], size=[0.002, 0.002, 0.03], contype=0, conaffinity=0, rgba=[0.1, 0.4, 1.0, 0.95])

    right_target = root.worldbody.add_body(name="right_target", mocap=True, pos=[0.4, -y_offset, 0.4])
    right_target.add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.018],
        contype=0,
        conaffinity=0,
        rgba=[0.25, 0.35, 0.95, 0.9],
    )
    right_target.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.03, 0.0, 0.0], size=[0.03, 0.002, 0.002], contype=0, conaffinity=0, rgba=[1.0, 0.1, 0.1, 0.95])
    right_target.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.0, 0.03, 0.0], size=[0.002, 0.03, 0.002], contype=0, conaffinity=0, rgba=[0.1, 1.0, 0.1, 0.95])
    right_target.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.0, 0.0, 0.03], size=[0.002, 0.002, 0.03], contype=0, conaffinity=0, rgba=[0.1, 0.4, 1.0, 0.95])

    # Goal markers used by telegrip UI/debug overlays.
    left_goal = root.worldbody.add_body(name="left_goal", mocap=True, pos=[0.4, y_offset, 0.4])
    left_goal.add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.015],
        contype=0,
        conaffinity=0,
        rgba=[0.2, 1.0, 0.35, 0.85],
    )

    right_goal = root.worldbody.add_body(name="right_goal", mocap=True, pos=[0.4, -y_offset, 0.4])
    right_goal.add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.015],
        contype=0,
        conaffinity=0,
        rgba=[0.2, 1.0, 1.0, 0.85],
    )

    return root


def _prepare_arm_spec(prefix: str) -> mujoco.MjSpec:
    """Load one ARM620 spec and remove scene-level elements before attaching."""
    arm_spec = mujoco.MjSpec.from_file(_SINGLE_ARM_XML.as_posix())
    arm_spec.modelname = f"{prefix[:-1]}_arm620"

    # The single-arm XML already contains a floor, a light and a keyframe.
    # Remove them here so the dual-arm root scene owns those elements only once.
    arm_spec.delete(arm_spec.geom("floor"))
    for light in list(arm_spec.worldbody.lights):
        arm_spec.delete(light)
    if arm_spec.keys:
        arm_spec.delete(arm_spec.key("home"))

    return arm_spec


def build_dual_arm620_model(y_offset: float = 0.28) -> mujoco.MjModel:
    """Create a dual-arm ARM620 model with unique left/right namespaces.

    Args:
        y_offset: Half of the distance between two arm bases in meters.

    Returns:
        Compiled MuJoCo model.
    """
    root = _build_dual_arm620_spec(y_offset=y_offset)
    return root.compile()


def save_dual_arm620_xml(output_path: str | Path, y_offset: float = 0.28) -> Path:
    """Save generated dual-arm scene XML for external loaders (e.g. telegrip visualizer)."""
    root = _build_dual_arm620_spec(y_offset=y_offset)
    output = Path(output_path)
    xml_text = root.to_xml()

    # `to_xml()` flattens file paths (e.g. arm620/Link1.STL). For this project layout,
    # meshes live under examples/arm620/meshes/*, so patch the relative paths.
    xml_text = xml_text.replace('file="arm620/', 'file="meshes/arm620/')
    xml_text = xml_text.replace('file="robotiq/', 'file="meshes/robotiq/')

    output.write_text(xml_text)
    return output


def build_dual_arm620_data(y_offset: float = 0.28) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Convenience helper returning both model and data."""
    model = build_dual_arm620_model(y_offset=y_offset)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data
