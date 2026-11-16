//! Thread-Local Device Context and Scoping.
//!
//! Allows setting and dynamically querying the current active device for the thread,
//! matching `torch.set_default_device(...)` and scoped context managers.

use std::cell::RefCell;

use crate::device::device_type::Device;

thread_local! {
    static CURRENT_DEVICE: RefCell<Option<Device>> = const { RefCell::new(None) };
}

/// Sets the default compute device for the current thread.
pub fn set_default_device(device: Device) {
    CURRENT_DEVICE.with(|c| {
        *c.borrow_mut() = Some(device);
    });
}

/// Gets the currently active compute device for the thread (defaults to `Device::Cpu`).
pub fn get_default_device() -> Device {
    CURRENT_DEVICE.with(|c| c.borrow().clone().unwrap_or(Device::Cpu))
}

/// Executes a closure within a dynamically scoped compute device, resetting it afterwards.
/// Panic-safe: the previous device is restored even when `f` panics.
pub fn with_device<F, R>(device: Device, f: F) -> R
where
    F: FnOnce() -> R,
{
    let prev = CURRENT_DEVICE.with(|c| c.borrow().clone());
    set_default_device(device);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    CURRENT_DEVICE.with(|c| {
        *c.borrow_mut() = prev;
    });
    match result {
        Ok(r) => r,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}
