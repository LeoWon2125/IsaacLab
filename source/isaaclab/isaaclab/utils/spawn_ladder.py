# isaaclab_tasks/utils/spawn_ladder_rigid_object.py
from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf

def spawn_ladder_rigid_object(stage, prim_path: str, rung_count=5, rung_length=0.7, rung_radius=0.025,
                               rung_spacing=0.3, side_width=0.05, side_depth=0.05, margin=0.3, **kwargs):

    ladder_xform = UsdGeom.Xform.Define(stage, Sdf.Path(prim_path))
    ladder_xform.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    UsdPhysics.CollisionAPI.Apply(ladder_xform.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(ladder_xform.GetPrim())
    PhysxSchema.PhysxRigidBodyAPI.Apply(ladder_xform.GetPrim())
    PhysxSchema.PhysxContactReportAPI.Apply(ladder_xform.GetPrim())

    ladder_height = (rung_count - 1) * rung_spacing + 2 * margin
    side_offset = rung_length / 2 + side_width / 2 + 0.005

    def add_box(name, position, size):
        path = f"{prim_path}/{name}"
        xform = UsdGeom.Xform.Define(stage, Sdf.Path(path))
        xform.AddTranslateOp().Set(Gf.Vec3f(*position))
        geom = UsdGeom.Cube.Define(stage, Sdf.Path(path + "/Geom"))
        sx, sy, sz = size
        geom.AddScaleOp().Set(Gf.Vec3f(sx / 2, sy / 2, sz / 2))
        UsdGeom.Gprim(geom.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(1.0, 1.0, 1.0)])

    def add_cylinder(name, position, radius, height):
        path = f"{prim_path}/{name}"
        xform = UsdGeom.Xform.Define(stage, Sdf.Path(path))
        xform.AddTranslateOp().Set(Gf.Vec3f(*position))
        xform.AddRotateXYZOp().Set(Gf.Vec3f(0, 90, 0))
        geom = UsdGeom.Cylinder.Define(stage, Sdf.Path(path + "/Geom"))
        geom.CreateRadiusAttr(radius)
        geom.CreateHeightAttr(height)
        UsdGeom.Gprim(geom.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(1.0, 1.0, 1.0)])

    # Side columns
    add_box("Side_Left", (-side_offset, 0, ladder_height / 2), (side_width, side_depth, ladder_height))
    add_box("Side_Right", (+side_offset, 0, ladder_height / 2), (side_width, side_depth, ladder_height))

    # Rungs
    for i in range(rung_count):
        z = margin + i * rung_spacing
        add_cylinder(f"Rung_{i}", (0, 0, z), rung_radius, rung_length)
