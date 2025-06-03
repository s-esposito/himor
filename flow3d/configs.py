from dataclasses import dataclass, replace
import yaml


@dataclass
class FGLRConfig:
    means: float = 1.6e-4
    opacities: float = 1e-2
    scales: float = 5e-3
    quats: float = 1e-3
    colors: float = 1e-2
    motion_coefs: float = 1e-2


@dataclass
class BGLRConfig:
    means: float = 1.6e-4
    opacities: float = 5e-2
    scales: float = 5e-3
    quats: float = 1e-3
    colors: float = 1e-2


@dataclass
class MotionLRConfig:
    rots: float = 1.6e-4
    transls: float = 1.6e-4

@dataclass
class MotionNodeLRConfig:
    positions: float = 1.6e-5
    radius: float = 5e-4
    motion_coefs: float = 1e-2


@dataclass
class SceneLRConfig:
    fg: FGLRConfig
    bg: BGLRConfig
    motion_bases: MotionLRConfig
    motion_nodes: MotionNodeLRConfig


@dataclass
class LossesConfig:
    w_rgb: float = 1.0
    w_depth_reg: float = 0.5
    w_depth_const: float = 0.1
    w_depth_grad: float = 1
    w_track: float = 2.0
    w_mask: float = 1.0
    w_smooth_bases: float = 0.1
    w_smooth_tracks: float = 2.0
    w_scale_var: float = 0.01
    w_z_accel: float = 1.0
    w_arap: float = 0.5    #0.1
    w_radius_reg: float = 0.0001


@dataclass
class OptimizerConfig:
    max_steps: int = 5000
    ## Adaptive gaussian control
    warmup_steps: int = 200
    control_every: int = 100
    reset_opacity_every_n_controls: int = 30
    stop_control_by_screen_steps: int = 4000
    stop_control_steps: int = 14000
    start_second_level: int = 4000
    ### Densify.
    densify_xys_grad_threshold: float = 0.0002
    densify_node_grad_threshold: float = 0.000085
    densify_scale_threshold: float = 0.01
    densify_screen_threshold: float = 0.05
    stop_densify_steps: int = 15000
    stop_node_grad_densify: int = 3500
    node_grad_densify_steps: int = 400
    start_grad_densify_steps: int = 2500
    node_gaussian_densify_steps: int = 200
    node_gaussian_densify_times: int = 5
    ### Cull.
    cull_opacity_threshold: float = 0.1
    cull_scale_threshold: float = 0.5
    cull_screen_threshold: float = 0.15


def load_yaml(filepath):
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

def update_dataclass(dc_instance, updates: dict):
    for key, value in updates.items():
        if hasattr(dc_instance, key):
            current_value = getattr(dc_instance, key)
            if isinstance(current_value, dict) and hasattr(current_value, '__dataclass_fields__'):
                update_dataclass(current_value, value)
            else:
                setattr(dc_instance, key, value)
    return dc_instance


