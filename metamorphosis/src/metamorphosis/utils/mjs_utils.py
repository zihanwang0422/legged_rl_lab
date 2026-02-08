import mujoco

def add_capsule_geom_(
    parent_body: mujoco.MjsBody,
    radius: float,
    fromto: list[float],
) -> mujoco.MjsGeom:
    geom = parent_body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE)
    geom.size = [radius, 0, 0.]
    geom.fromto = fromto
    return geom

