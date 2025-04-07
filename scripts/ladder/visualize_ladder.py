"""
Drop a complete rigid ladder made of multiple cylinders.
Everything is grouped under a single RigidBody Xform.
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Drop a rigid ladder.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf


def spawn_cylinder(stage, parent_path: str, name: str, position, radius, height, axis="Z", color=(0.2, 0.6, 1.0)):
    path = f"{parent_path}/{name}"
    xform = UsdGeom.Xform.Define(stage, Sdf.Path(path))
    xform.AddTranslateOp().Set(Gf.Vec3f(*position))

    if axis == "X":
        xform.AddRotateXYZOp().Set(Gf.Vec3f(0, 90, 0))
    elif axis == "Y":
        xform.AddRotateXYZOp().Set(Gf.Vec3f(90, 0, 0))

    cylinder = UsdGeom.Cylinder.Define(stage, Sdf.Path(path + "/Cylinder"))
    cylinder.CreateRadiusAttr(radius)
    cylinder.CreateHeightAttr(height)

    gprim = UsdGeom.Gprim(cylinder.GetPrim())
    gprim.CreateDisplayColorAttr([Gf.Vec3f(*color)])


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.5, 1.5, 2.0], target=[0.0, 0.0, 1.0])

    # Add ground + light
    sim_utils.GroundPlaneCfg().func("/World/groundPlane", sim_utils.GroundPlaneCfg())
    # Lights
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    stage = sim.stage

    # Create parent Xform for entire ladder
    ladder_path = "/World/Ladder"
    ladder_xform = UsdGeom.Xform.Define(stage, Sdf.Path(ladder_path))
    ladder_xform.AddTranslateOp().Set(Gf.Vec3f(0, 0, 2.0))  # start high in air

    # Add RigidBody + Collision to ladder root
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(ladder_path))
    UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(ladder_path))
    PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath(ladder_path))

    # Ladder spec
    rung_radius = 0.025
    rung_length = 0.5
    rung_count = 8
    rung_spacing = 0.3
    margin = 0.3
    side_radius = 0.04
    ladder_height = (rung_count - 1) * rung_spacing + 2 * margin

    # Add sides
    spawn_cylinder(stage, ladder_path, "Side_Left", position=(-rung_length / 2, 0, ladder_height / 2),
                   radius=side_radius, height=ladder_height, axis="Z", color=(1.0, 1.0, 1.0))
    spawn_cylinder(stage, ladder_path, "Side_Right", position=(rung_length / 2, 0, ladder_height / 2),
                   radius=side_radius, height=ladder_height, axis="Z", color=(1.0, 1.0, 1.0))

    # Add rungs
    for i in range(rung_count):
        z = margin + i * rung_spacing
        spawn_cylinder(stage, ladder_path, f"Rung_{i}", position=(0.0, 0.0, z),
                       radius=rung_radius, height=rung_length, axis="X", color=(1.0, 1.0, 1.0))

    sim.reset()
    print("[INFO]: Rigid ladder spawned and dropped!")

    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()
