"""Benchmark geometry extensions, isolated from upstream primitive APIs."""

import sapien.core as sapien
import numpy as np
from pathlib import Path
import transforms3d as t3d
import sapien.physx as sapienp
import json
import os, re

from typing import Optional, Union, Any

from envs.utils.actor_utils import Actor, ArticulationActor

def preprocess(scene, pose: sapien.Pose) -> tuple[sapien.Scene, sapien.Pose]:
    """Add entity to scene. Add bias to z axis if scene is not sapien.Scene."""
    if isinstance(scene, sapien.Scene):
        return scene, pose
    else:
        return scene.scene, sapien.Pose([pose.p[0], pose.p[1], pose.p[2] + scene.table_z_bias], pose.q)

def create_entity_box(
    scene,
    pose: sapien.Pose,
    half_size,
    color=None,
    is_static=False,
    name="",
    texture_id=None,
    friction=None,
) -> sapien.Entity:
    scene, pose = preprocess(scene, pose)

    entity = sapien.Entity()
    entity.set_name(name)
    entity.set_pose(pose)

    # create physical material
    if friction is not None:
        phys_material = scene.create_physical_material(
            static_friction=friction,
            dynamic_friction=friction,
            restitution=0.0,
        )
    else:
        phys_material = scene.default_physical_material

    # create PhysX dynamic rigid body
    rigid_component = (sapien.physx.PhysxRigidDynamicComponent()
                       if not is_static else sapien.physx.PhysxRigidStaticComponent())
    rigid_component.attach(
        sapien.physx.PhysxCollisionShapeBox(half_size=half_size, material=phys_material))

    # Add texture
    if texture_id is not None:

        # test for both .png and .jpg
        texturepath = f"./assets/background_texture/{texture_id}.png"
        # create texture from file
        texture2d = sapien.render.RenderTexture2D(texturepath)
        material = sapien.render.RenderMaterial()
        material.set_base_color_texture(texture2d)
        # renderer.create_texture_from_file(texturepath)
        # material.set_diffuse_texture(texturepath)
        material.base_color = [1, 1, 1, 1]
        material.metallic = 0.1
        material.roughness = 0.3
    else:
        material = sapien.render.RenderMaterial(base_color=[*color[:3], 1])

    # create render body for visualization
    render_component = sapien.render.RenderBodyComponent()
    render_component.attach(
        # add a box visual shape with given size and rendering material
        sapien.render.RenderShapeBox(half_size, material))

    entity.add_component(rigid_component)
    entity.add_component(render_component)
    entity.set_pose(pose)

    # in general, entity should only be added to scene after it is fully built
    scene.add_entity(entity)
    return entity

def create_box(
    scene,
    pose: sapien.Pose,
    half_size,
    color=None,
    is_static=False,
    name="",
    texture_id=None,
    boxtype="default",
    friction=None,
    mass=0.01,
) -> Actor:
    entity = create_entity_box(
        scene=scene,
        pose=pose,
        half_size=half_size,
        color=color,
        is_static=is_static,
        name=name,
        texture_id=texture_id,
        friction=friction,
    )
    if boxtype == "default":
        data = {
            "center": [0, 0, 0],
            "extents":
            half_size,
            "scale":
            half_size,
            "target_pose": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]],
            "contact_points_pose": [
                [
                    [0, 0, 1, 0],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0.0],
                    [0, 0, 0, 1],
                ],  # top_down(front)
                [
                    [1, 0, 0, 0],
                    [0, 0, -1, 0],
                    [0, 1, 0, 0.0],
                    [0, 0, 0, 1],
                ],  # top_down(right)
                [
                    [-1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 1, 0, 0.0],
                    [0, 0, 0, 1],
                ],  # top_down(left)
                [
                    [0, 0, -1, 0],
                    [-1, 0, 0, 0],
                    [0, 1, 0, 0.0],
                    [0, 0, 0, 1],
                ],  # top_down(back)
                # [[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0.0], [0, 0, 0, 1]], # front
                # [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0.0], [0, 0, 0, 1]], # right
                # [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0.0], [0, 0, 0, 1]], # left
                # [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0.0], [0, 0, 0, 1]], # back
            ],
            "transform_matrix":
            np.eye(4).tolist(),
            "functional_matrix": [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0, 0.0],
                    [0.0, 0, -1.0, -1],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0, 0.0],
                    [0.0, 0, -1.0, 1],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ],  # functional points matrix
            "contact_points_description": [],  # contact points description
            "contact_points_group": [[0, 1, 2, 3], [4, 5, 6, 7]],
            "contact_points_mask": [True, True],
            "target_point_description": ["The center point on the bottom of the box."],
        }
    else:
        data = {
            "center": [0, 0, 0],
            "extents":
            half_size,
            "scale":
            half_size,
            "target_pose": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]],
            "contact_points_pose": [
                [[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0.7], [0, 0, 0, 1]],  # front
                [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0.7], [0, 0, 0, 1]],  # right
                [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0.7], [0, 0, 0, 1]],  # left
                [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0.7], [0, 0, 0, 1]],  # back
                [[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, -0.7], [0, 0, 0, 1]],  # front
                [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, -0.7], [0, 0, 0, 1]],  # right
                [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, -0.7], [0, 0, 0, 1]],  # left
                [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, -0.7], [0, 0, 0, 1]],  # back
            ],
            "transform_matrix":
            np.eye(4).tolist(),
            "functional_matrix": [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0, 0.0],
                    [0.0, 0, -1.0, -1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0, 0.0],
                    [0.0, 0, -1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ],  # functional points matrix
            "contact_points_description": [],  # contact points description
            "contact_points_group": [[0, 1, 2, 3, 4, 5, 6, 7]],
            "contact_points_mask": [True, True],
            "target_point_description": ["The center point on the bottom of the box."],
        }
    return Actor(entity, data, mass=mass)

def create_entity_sphere(
    scene,
    pose: sapien.Pose,
    radius: float,
    color=None,
    is_static=False,
    name="",
    texture_id=None,
    friction=None,
) -> sapien.Entity:
    scene, pose = preprocess(scene, pose)
    entity = sapien.Entity()
    entity.set_name(name)
    entity.set_pose(pose)

    # create physical material
    if friction is not None:
        phys_material = scene.create_physical_material(
            static_friction=friction,
            dynamic_friction=friction,
            restitution=0.0,
        )
    else:
        phys_material = scene.default_physical_material

    # create PhysX dynamic rigid body
    rigid_component = (sapien.physx.PhysxRigidDynamicComponent()
                       if not is_static else sapien.physx.PhysxRigidStaticComponent())
    rigid_component.attach(
        sapien.physx.PhysxCollisionShapeSphere(radius=radius, material=phys_material))

    # Add texture
    if texture_id is not None:

        # test for both .png and .jpg
        texturepath = f"./assets/textures/{texture_id}.png"
        # create texture from file
        texture2d = sapien.render.RenderTexture2D(texturepath)
        material = sapien.render.RenderMaterial()
        material.set_base_color_texture(texture2d)
        # renderer.create_texture_from_file(texturepath)
        # material.set_diffuse_texture(texturepath)
        material.base_color = [1, 1, 1, 1]
        material.metallic = 0.1
        material.roughness = 0.3
    else:
        material = sapien.render.RenderMaterial(base_color=[*color[:3], 1])

    # create render body for visualization
    render_component = sapien.render.RenderBodyComponent()
    render_component.attach(
        # add a sphere visual shape with given size and rendering material
        sapien.render.RenderShapeSphere(radius=radius, material=material))

    entity.add_component(rigid_component)
    entity.add_component(render_component)
    entity.set_pose(pose)

    # in general, entity should only be added to scene after it is fully built
    scene.add_entity(entity)
    return entity

def create_sphere(
    scene,
    pose: sapien.Pose,
    radius: float,
    color=None,
    is_static=False,
    name="",
    texture_id=None,
    friction=None,
    mass=0.01,
) -> Actor:
    entity = create_entity_sphere(
        scene=scene,
        pose=pose,
        radius=radius,
        color=color,
        is_static=is_static,
        name=name,
        texture_id=texture_id,
        friction=friction,
    )
    data = {
        "center": [0, 0, 0],
        "extents": [radius, radius, radius],
        "scale": [radius, radius, radius],
        "target_pose": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]],
        "contact_points_pose": [
            # top_down(front)
            [
                [0, 0, 1, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0.0],
                [0, 0, 0, 1],
            ],
            # top_down(right)
            [
                [1, 0, 0, 0],
                [0, 0, -1, 0],
                [0, 1, 0, 0.0],
                [0, 0, 0, 1],
            ],
            # top_down(left)
            [
                [-1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0.0],
                [0, 0, 0, 1],
            ],
            # top_down(back)
            [
                [0, 0, -1, 0],
                [-1, 0, 0, 0],
                [0, 1, 0, 0.0],
                [0, 0, 0, 1],
            ],
        ],
        "transform_matrix": np.eye(4).tolist(),
        "functional_matrix": [
            # bottom
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0, 0.0],
                [0.0, 0, -1.0, -1],
                [0.0, 0.0, 0.0, 1.0],
            ],
            # top
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0, 0.0],
                [0.0, 0, -1.0, 1],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ],
        "contact_points_description": [],
        "contact_points_group": [[0, 1, 2, 3]],
        "contact_points_mask": [True],
    }
    return Actor(entity, data, mass=mass)

def create_entity_cylinder(
    scene,
    pose: sapien.Pose,
    radius: float,
    half_length: float,
    color=None,
    is_static=False,
    name="",
    texture_id=None,
) -> sapien.Entity:
    scene, pose = preprocess(scene, pose)

    entity = sapien.Entity()
    entity.set_name(name)
    entity.set_pose(pose)

    # create PhysX dynamic rigid body
    rigid_component = (sapien.physx.PhysxRigidDynamicComponent()
                       if not is_static else sapien.physx.PhysxRigidStaticComponent())
    rigid_component.attach(
        sapien.physx.PhysxCollisionShapeCylinder(
            radius=radius,
            half_length=half_length,
            material=scene.default_physical_material,
        ))

    # Add texture
    if texture_id is not None:
        texturepath = f"./assets/background_texture/{texture_id}.png"
        texture2d = sapien.render.RenderTexture2D(texturepath)
        material = sapien.render.RenderMaterial()
        material.set_base_color_texture(texture2d)
        material.base_color = [1, 1, 1, 1]
        material.metallic = 0.1
        material.roughness = 0.3
    else:
        material = sapien.render.RenderMaterial(base_color=[*color[:3], 1])

    # create render body for visualization
    render_component = sapien.render.RenderBodyComponent()
    render_component.attach(
        # add a cylinder visual shape with given size and rendering material
        sapien.render.RenderShapeCylinder(
            radius=radius,
            half_length=half_length,
            material=material,
        ))

    entity.add_component(rigid_component)
    entity.add_component(render_component)
    entity.set_pose(pose)

    # in general, entity should only be added to scene after it is fully built
    scene.add_entity(entity)
    return entity

def create_cylinder(
    scene,
    pose: sapien.Pose,
    radius: float,
    half_length: float,
    color=None,
    is_static=False,
    name="",
    texture_id=None,
) -> Actor:
    entity = create_entity_cylinder(
        scene=scene,
        pose=pose,
        radius=radius,
        half_length=half_length,
        color=color,
        is_static=is_static,
        name=name,
        texture_id=texture_id,
    )
    data = {
        "center": [0, 0, 0],
        "extents": [radius, radius, half_length],
        "scale": [radius, radius, half_length],
        "target_pose": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]],
        "contact_points_pose": [
            # top_down(front)
            [
                [0, 0, 1, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0.0],
                [0, 0, 0, 1],
            ],
            # top_down(right)
            [
                [1, 0, 0, 0],
                [0, 0, -1, 0],
                [0, 1, 0, 0.0],
                [0, 0, 0, 1],
            ],
            # top_down(left)
            [
                [-1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0.0],
                [0, 0, 0, 1],
            ],
            # top_down(back)
            [
                [0, 0, -1, 0],
                [-1, 0, 0, 0],
                [0, 1, 0, 0.0],
                [0, 0, 0, 1],
            ],
        ],
        "transform_matrix": np.eye(4).tolist(),
        "functional_matrix": [
            # bottom
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0, 0.0],
                [0.0, 0, -1.0, -1],
                [0.0, 0.0, 0.0, 1.0],
            ],
            # top
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0, 0.0],
                [0.0, 0, -1.0, 1],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ],
        "contact_points_description": [],
        "contact_points_group": [[0, 1, 2, 3]],
        "contact_points_mask": [True],
    }
    return Actor(entity, data)

def create_standing_cylinder(
    scene,
    pose: sapien.Pose,
    radius: float,
    half_height: float,
    color=None,
    name="",
    friction=2.0,
    mass=0.05,
) -> Actor:
    """
    Create a standing cylinder (Z-axis aligned).
    SAPIEN cylinders are X-axis aligned by default, so we rotate it 90 deg around Y.
    """
    # preprocess handles table_z_bias and extracts scene.scene
    scene, pose = preprocess(scene, pose)

    # Create high friction material
    high_friction_material = scene.create_physical_material(
        static_friction=friction,
        dynamic_friction=friction,
        restitution=0.0
    )

    # Combine the input pose with a 90-degree rotation around Y to make it stand
    rotation_quat = t3d.euler.euler2quat(0, np.pi / 2, 0, "sxyz")
    standing_quat = t3d.quaternions.qmult(pose.q, rotation_quat)
    standing_pose = sapien.Pose(pose.p, standing_quat)

    entity = sapien.Entity()
    entity.set_name(name)
    entity.set_pose(standing_pose)

    # Physics
    rigid_component = sapien.physx.PhysxRigidDynamicComponent()
    rigid_component.attach(
        sapien.physx.PhysxCollisionShapeCylinder(
            radius=radius,
            half_length=half_height,
            material=high_friction_material,
        )
    )

    # Render
    render_component = sapien.render.RenderBodyComponent()
    render_component.attach(
        sapien.render.RenderShapeCylinder(
            radius=radius,
            half_length=half_height,
            material=sapien.render.RenderMaterial(base_color=[*color[:3], 1]),
        )
    )

    entity.add_component(rigid_component)
    entity.add_component(render_component)
    scene.add_entity(entity)

    actor_data = {
        "center": [0, 0, 0],
        "extents": [radius, radius, half_height],
        "scale": [half_height, radius, radius],
        "contact_points_pose": [
            [[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0.0], [0, 0, 0, 1]],
            [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0.0], [0, 0, 0, 1]],
            [[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0.0], [0, 0, 0, 1]],
            [[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0.0], [0, 0, 0, 1]],
        ],
        "transform_matrix": np.eye(4).tolist(),
        "functional_matrix": [
            [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0, 0.0], [0.0, 0, -1.0, -1], [0.0, 0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0, 0.0], [0.0, 0, -1.0, 1], [0.0, 0.0, 0.0, 1.0]],
        ],
        "target_pose": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]],
        "contact_points_description": [],
        "contact_points_group": [[0, 1, 2, 3]],
        "contact_points_mask": [True],
    }

    return Actor(entity, actor_data, mass=mass)

def get_glb_or_obj_file(modeldir, model_id):
    modeldir = Path(modeldir)
    if model_id is None:
        file = modeldir / "base.glb"
    else:
        file = modeldir / f"base{model_id}.glb"
    if not file.exists():
        if model_id is None:
            file = modeldir / "textured.obj"
        else:
            file = modeldir / f"textured{model_id}.obj"
    return file

def create_actor(
        scene,
        pose: sapien.Pose,
        modelname: str,
        scale=(1, 1, 1),
        convex=False,
        is_static=False,
        model_id: int = 0,
        scale_multiplier=1.0,
) -> Actor:
    scene, pose = preprocess(scene, pose)
    modeldir = Path("assets/objects") / modelname

    if model_id is None:
        json_file_path = modeldir / "model_data.json"
    else:
        json_file_path = modeldir / f"model_data{model_id}.json"

    collision_file = ""
    visual_file = ""
    if (modeldir / "collision").exists():
        collision_file = get_glb_or_obj_file(modeldir / "collision", model_id)
    if collision_file == "" or not collision_file.exists():
        collision_file = get_glb_or_obj_file(modeldir, model_id)

    if (modeldir / "visual").exists():
        visual_file = get_glb_or_obj_file(modeldir / "visual", model_id)
    if visual_file == "" or not visual_file.exists():
        visual_file = get_glb_or_obj_file(modeldir, model_id)

    if not collision_file.exists() or not visual_file.exists():
        print(modelname, "is not exist model file!")
        return None

    try:
        with open(json_file_path, "r") as file:
            model_data = json.load(file)
        base_scale = model_data["scale"]
        # Apply scale_multiplier to the base scale from JSON
        scale = tuple(s * scale_multiplier for s in base_scale)
        # Update model_data scale so contact points are also scaled correctly
        model_data["scale"] = list(scale)
    except:
        model_data = None
        # If no JSON, apply scale_multiplier to the provided scale
        scale = tuple(s * scale_multiplier for s in scale)

    builder = scene.create_actor_builder()
    if is_static:
        builder.set_physx_body_type("static")
    else:
        builder.set_physx_body_type("dynamic")

    if convex == True:
        builder.add_multiple_convex_collisions_from_file(filename=str(collision_file), scale=scale)
    else:
        builder.add_nonconvex_collision_from_file(
            filename=str(collision_file),
            scale=scale,
        )

    builder.add_visual_from_file(filename=str(visual_file), scale=scale)
    mesh = builder.build(name=modelname)
    mesh.set_name(modelname)
    mesh.set_pose(pose)
    return Actor(mesh, model_data)
