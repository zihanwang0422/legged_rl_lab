from metamorphosis.builder import QuadrupedBuilder
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf


def make_stage(path="quadruped.usda", meters=1.0, up="Z"):
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageMetersPerUnit(stage, meters)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z if up == "Z" else UsdGeom.Tokens.y)
    # Physics scene
    phys_scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    phys_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
    phys_scene.CreateGravityMagnitudeAttr().Set(9.81 if meters == 1.0 else 981.0)
    return stage


stage = make_stage()
builder = QuadrupedBuilder()
param = builder.sample_params(seed=0)
prim = builder.spawn(stage, prim_path="/Robot", param=param)
stage.SetDefaultPrim(prim)
stage.GetRootLayer().Save()

