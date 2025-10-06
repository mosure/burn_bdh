use std::env;

use burn::tensor::backend::Backend as BackendTrait;
use burn_wgpu::{self, MemoryConfiguration, RuntimeOptions, Wgpu, graphics};

/// The concrete device type used by the `Wgpu<f32>` backend.
pub type WgpuDevice = <Wgpu<f32> as BackendTrait>::Device;

/// Initialize the global wgpu runtime using environment-driven overrides.
///
/// Environment variables:
/// * `BDH_WGPU_BACKEND` - `auto` (default), `vulkan`, `dx12`, `metal`, or `opengl`.
/// * `BDH_WGPU_TASKS_MAX` - maximum aggregated tasks per GPU submit (usize).
/// * `BDH_WGPU_MEMORY` - `subslices` (default) or `exclusive`.
pub fn init_runtime(device: &WgpuDevice) {
    match backend_override() {
        BackendOverride::Auto => {
            burn_wgpu::init_setup::<graphics::AutoGraphicsApi>(device, runtime_options());
        }
        BackendOverride::Vulkan => {
            burn_wgpu::init_setup::<graphics::Vulkan>(device, runtime_options());
        }
        BackendOverride::Dx12 => {
            burn_wgpu::init_setup::<graphics::Dx12>(device, runtime_options());
        }
        BackendOverride::Metal => {
            burn_wgpu::init_setup::<graphics::Metal>(device, runtime_options());
        }
        BackendOverride::OpenGl => {
            burn_wgpu::init_setup::<graphics::OpenGl>(device, runtime_options());
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum BackendOverride {
    Auto,
    Vulkan,
    Dx12,
    Metal,
    OpenGl,
}

fn backend_override() -> BackendOverride {
    match env::var("BDH_WGPU_BACKEND")
        .unwrap_or_else(|_| "auto".to_string())
        .to_ascii_lowercase()
        .as_str()
    {
        "vulkan" => BackendOverride::Vulkan,
        "dx12" | "directx" => BackendOverride::Dx12,
        "metal" => BackendOverride::Metal,
        "opengl" | "gl" => BackendOverride::OpenGl,
        _ => BackendOverride::Auto,
    }
}

fn runtime_options() -> RuntimeOptions {
    let tasks_max = env::var("BDH_WGPU_TASKS_MAX")
        .ok()
        .and_then(|value| value.parse::<usize>().ok());

    let memory_config = match env::var("BDH_WGPU_MEMORY")
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "exclusive" => MemoryConfiguration::ExclusivePages,
        "subslices" | "" => MemoryConfiguration::SubSlices,
        other => {
            eprintln!("Unsupported BDH_WGPU_MEMORY value '{other}', falling back to subslices.");
            MemoryConfiguration::SubSlices
        }
    };

    RuntimeOptions {
        tasks_max: tasks_max.unwrap_or(RuntimeOptions::default().tasks_max),
        memory_config,
    }
}
