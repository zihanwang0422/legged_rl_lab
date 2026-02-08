import numpy as np
import mujoco

from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf
from scipy.spatial.transform import Rotation as R

try:
    # only available in Isaac Sim's custom USD build
    from pxr import PhysxSchema
except ImportError:
    PhysxSchema = None


def add_default_transform_(prim: Usd.Prim):
    vec3_dtype, vec3_cls = Sdf.ValueTypeNames.Float3, Gf.Vec3f
    quat_dtype, quat_cls = Sdf.ValueTypeNames.Quatf, Gf.Quatf
    order = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
    # add xform ops [scale, orient, translate]
    prim.CreateAttribute("xformOp:scale", vec3_dtype, False).Set(vec3_cls(1.0, 1.0, 1.0))
    prim.CreateAttribute("xformOp:orient", quat_dtype, False).Set(quat_cls(1.0, 0.0, 0.0, 0.0))
    prim.CreateAttribute("xformOp:translate", vec3_dtype, False).Set(vec3_cls(0.0, 0.0, 0.0))
    prim.CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.TokenArray, False).Set(order)


def create_capsule(stage: Usd.Stage, path: str, radius: float, fromto: np.ndarray):
    capsule = UsdGeom.Capsule.Define(stage, path)
    add_default_transform_(capsule.GetPrim())
    direction = fromto[3:] - fromto[:3]
    length = np.linalg.norm(direction)
    direction = direction / length
    axis = np.cross(direction, [0, 0, 1])
    angle = np.arccos(np.dot(direction, [0, 0, 1]))
    translation = (fromto[:3] + fromto[3:]) / 2
    orient = R.from_rotvec(angle * axis).as_quat(scalar_first=True)
    capsule.CreateAxisAttr("Z")
    capsule.CreateRadiusAttr(radius)
    capsule.CreateHeightAttr(length)
    capsule.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3f(*translation))
    capsule.GetPrim().GetAttribute("xformOp:orient").Set(Gf.Quatf(*orient))
    return capsule


def create_cylinder(stage: Usd.Stage, path: str, radius: float, height: float, quat: np.ndarray = None):
    """Create a cylinder geometry in USD.
    
    Args:
        stage: USD stage
        path: Path for the cylinder prim
        radius: Cylinder radius
        height: Cylinder height
        quat: Quaternion for orientation [w, x, y, z], if None uses identity
    """
    cylinder = UsdGeom.Cylinder.Define(stage, path)
    add_default_transform_(cylinder.GetPrim())
    cylinder.CreateAxisAttr("Z")  # Default axis
    cylinder.CreateRadiusAttr(radius)
    cylinder.CreateHeightAttr(height)
    
    # Apply orientation if provided
    if quat is not None:
        cylinder.GetPrim().GetAttribute("xformOp:orient").Set(Gf.Quatf(*quat))
    
    return cylinder


def create_fixed_joint(stage: Usd.Stage, path: str, body_0: Usd.Prim, body_1: Usd.Prim):
    joint = UsdPhysics.FixedJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([body_0.GetPath()])
    joint.CreateBody1Rel().SetTargets([body_1.GetPath()])
    xfCache = UsdGeom.XformCache()
    body_0_pose = xfCache.GetLocalToWorldTransform(body_0)
    body_1_pose = xfCache.GetLocalToWorldTransform(body_1)
    rel_pose = body_1_pose * body_0_pose.GetInverse()
    rel_pose = rel_pose.RemoveScaleShear()
    pos1 = Gf.Vec3f(rel_pose.ExtractTranslation())
    rot1 = Gf.Quatf(rel_pose.ExtractRotationQuat())
    joint.CreateLocalPos0Attr().Set(pos1)
    joint.CreateLocalRot0Attr().Set(rot1)
    return joint


def create_revolute_joint(stage: Usd.Stage, path: str, body_0: Usd.Prim, body_1: Usd.Prim, axis: str = "Z"):
    assert axis in ["X", "Y", "Z"], f"Invalid axis: {axis}"
    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([body_0.GetPath()])
    joint.CreateBody1Rel().SetTargets([body_1.GetPath()])
    joint.CreateAxisAttr(axis)
    xfCache = UsdGeom.XformCache()
    body_0_pose = xfCache.GetLocalToWorldTransform(body_0)
    body_1_pose = xfCache.GetLocalToWorldTransform(body_1)
    rel_pose = body_1_pose * body_0_pose.GetInverse()
    rel_pose = rel_pose.RemoveScaleShear()
    pos1 = Gf.Vec3f(rel_pose.ExtractTranslation())
    rot1 = Gf.Quatf(rel_pose.ExtractRotationQuat())
    joint.CreateLocalPos0Attr().Set(pos1)
    joint.CreateLocalRot0Attr().Set(rot1)

    prim = joint.GetPrim()
    # check if prim has joint drive applied on it
    usd_drive_api = UsdPhysics.DriveAPI(prim, "angular")
    if not usd_drive_api:
        usd_drive_api = UsdPhysics.DriveAPI.Apply(prim, "angular")
    # check if prim has Physx joint drive applied on it
    if PhysxSchema is not None:
        physx_joint_api = PhysxSchema.PhysxJointAPI(prim)
        if not physx_joint_api:
            physx_joint_api = PhysxSchema.PhysxJointAPI.Apply(prim)
    return joint


def from_mjspec(stage: Usd.Stage, prim_path: str, spec: mujoco.MjSpec) -> Usd.Prim:
        mjmodel = spec.compile()
        mjdata = mujoco.MjData(mjmodel)
        mujoco.mj_forward(mjmodel, mjdata)

        root_prim = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
        # stage.SetDefaultPrim(root_prim)
        UsdPhysics.ArticulationRootAPI.Apply(root_prim)

        prim_dict = {}
        for mjbody in spec.worldbody.find_all("body"):
            xform = UsdGeom.Xform.Define(stage, f"{prim_path}/{mjbody.name}")
            xform_prim = xform.GetPrim()
            for i, geom in enumerate(mjbody.geoms):
                geom_path = f"{xform.GetPath()}/collision_{i}"
                match geom.type:
                    case mujoco.mjtGeom.mjGEOM_BOX:
                        cube = UsdGeom.Cube.Define(stage, geom_path)
                        cube.CreateSizeAttr(2.0)
                        add_default_transform_(cube.GetPrim())
                        cube.GetPrim().GetAttribute("xformOp:scale").Set(Gf.Vec3f(geom.size[0], geom.size[1], geom.size[2]))
                    case mujoco.mjtGeom.mjGEOM_CAPSULE:
                        capsule = create_capsule(stage, geom_path, geom.size[0], np.array(geom.fromto))
                    case mujoco.mjtGeom.mjGEOM_SPHERE:
                        sphere = UsdGeom.Sphere.Define(stage, geom_path)
                        add_default_transform_(sphere.GetPrim())
                        sphere.CreateRadiusAttr(geom.size[0])
                    case mujoco.mjtGeom.mjGEOM_CYLINDER:
                        # Extract quaternion if provided, otherwise use None
                        quat = np.array(geom.quat) if hasattr(geom, 'quat') and geom.quat is not None else None
                        cylinder = create_cylinder(stage, geom_path, geom.size[0], geom.size[1] * 2, quat)
                    case _:
                        raise ValueError(f"Unsupported geometry type: {geom.type}")
            add_default_transform_(xform_prim)
            xform_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3f(*mjdata.xpos[mjbody.id]))
            xform_prim.GetAttribute("xformOp:orient").Set(Gf.Quatf(*mjdata.xquat[mjbody.id]))
            UsdPhysics.CollisionAPI.Apply(xform_prim)
            UsdPhysics.RigidBodyAPI.Apply(xform_prim)

            prim_dict[mjbody.id] = xform_prim
            joints = mjbody.joints
            if mjbody.parent.id > 0:
                parent_prim = prim_dict[mjbody.parent.id]
                if len(joints):
                    assert len(joints) == 1, "Only one joint is supported."
                    joint = joints[0]
                    joint_path = f"{parent_prim.GetPath()}/{joint.name}"
                    joint_range = joint.range / np.pi * 180
                    match joint.type:
                        case mujoco.mjtJoint.mjJNT_HINGE:
                            axis = ["X", "Y", "Z"][np.argmax(np.abs(joint.axis))]
                            joint = create_revolute_joint(stage, joint_path, parent_prim, xform_prim, axis)
                            joint.CreateLowerLimitAttr(joint_range[0])
                            joint.CreateUpperLimitAttr(joint_range[1])
                        case _:
                            raise ValueError(f"Unsupported joint type: {joint.type}")
                else:
                    joint_path = f"{parent_prim.GetPath()}/{mjbody.name}_joint"
                    create_fixed_joint(stage, joint_path, parent_prim, xform_prim)
        return root_prim