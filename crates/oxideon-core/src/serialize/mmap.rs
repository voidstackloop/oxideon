//! Zero-dependency memory-mapped (`mmap`) lazy weight loading for `.safetensors` and `.gguf`.
//!
//! Uses OS virtual memory mapping (`CreateFileMappingW` + `MapViewOfFile` on Windows,
//! `mmap` on Unix) so large model files open instantaneously and only touched tensor pages
//! are paged in from disk.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Read-only memory-mapped file region with deterministic `Drop` unmapping.
pub struct MmapFile {
    len: usize,
    #[cfg(windows)]
    ptr: *const u8,
    #[cfg(windows)]
    mapping: *mut core::ffi::c_void,
    #[cfg(unix)]
    ptr: *const u8,
    #[cfg(not(any(windows, unix)))]
    fallback: Vec<u8>,
    _file: File,
}

unsafe impl Send for MmapFile {}
unsafe impl Sync for MmapFile {}

impl MmapFile {
    /// Opens `path` and maps its entire byte range read-only into virtual memory.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref())
            .map_err(|e| Error::Msg(format!("mmap open '{}': {e}", path.as_ref().display())))?;
        let meta = file
            .metadata()
            .map_err(|e| Error::Msg(format!("mmap metadata: {e}")))?;
        let len = usize::try_from(meta.len())
            .map_err(|_| Error::Msg("mmap: file size exceeds usize".into()))?;
        if len == 0 {
            return Err(Error::Msg("mmap: cannot map empty file".into()));
        }

        #[cfg(windows)]
        {
            use std::os::windows::io::AsRawHandle;
            type Handle = *mut core::ffi::c_void;
            const PAGE_READONLY: u32 = 0x02;
            const FILE_MAP_READ: u32 = 0x0004;
            unsafe extern "system" {
                fn CreateFileMappingW(
                    hfile: Handle,
                    attrs: *const core::ffi::c_void,
                    protect: u32,
                    max_hi: u32,
                    max_lo: u32,
                    name: *const u16,
                ) -> Handle;
                fn MapViewOfFile(
                    hmap: Handle,
                    access: u32,
                    off_hi: u32,
                    off_lo: u32,
                    bytes: usize,
                ) -> *const core::ffi::c_void;
                fn CloseHandle(h: Handle) -> i32;
            }
            let h = file.as_raw_handle() as Handle;
            let mapping = unsafe {
                CreateFileMappingW(h, core::ptr::null(), PAGE_READONLY, 0, 0, core::ptr::null())
            };
            if mapping.is_null() {
                return Err(Error::Msg("mmap: CreateFileMappingW failed".into()));
            }
            let view = unsafe { MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0) };
            if view.is_null() {
                unsafe {
                    CloseHandle(mapping);
                }
                return Err(Error::Msg("mmap: MapViewOfFile failed".into()));
            }
            Ok(Self {
                len,
                ptr: view as *const u8,
                mapping,
                _file: file,
            })
        }

        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            const PROT_READ: i32 = 1;
            const MAP_PRIVATE: i32 = 2;
            unsafe extern "C" {
                fn mmap(
                    addr: *mut core::ffi::c_void,
                    len: usize,
                    prot: i32,
                    flags: i32,
                    fd: i32,
                    offset: i64,
                ) -> *mut core::ffi::c_void;
            }
            let ptr = unsafe {
                mmap(
                    core::ptr::null_mut(),
                    len,
                    PROT_READ,
                    MAP_PRIVATE,
                    file.as_raw_fd(),
                    0,
                )
            };
            if ptr == (!0usize) as *mut core::ffi::c_void {
                return Err(Error::Msg("mmap: unix mmap failed".into()));
            }
            Ok(Self {
                len,
                ptr: ptr as *const u8,
                _file: file,
            })
        }

        #[cfg(not(any(windows, unix)))]
        {
            use std::io::Read;
            let mut buf = Vec::with_capacity(len);
            let mut f = file;
            f.read_to_end(&mut buf)
                .map_err(|e| Error::Msg(format!("mmap fallback read: {e}")))?;
            Ok(Self {
                len: buf.len(),
                fallback: buf,
                _file: f,
            })
        }
    }

    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        #[cfg(any(windows, unix))]
        {
            unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
        }
        #[cfg(not(any(windows, unix)))]
        {
            &self.fallback
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for MmapFile {
    fn drop(&mut self) {
        #[cfg(windows)]
        unsafe {
            type Handle = *mut core::ffi::c_void;
            unsafe extern "system" {
                fn UnmapViewOfFile(base: *const core::ffi::c_void) -> i32;
                fn CloseHandle(h: Handle) -> i32;
            }
            UnmapViewOfFile(self.ptr as *const core::ffi::c_void);
            CloseHandle(self.mapping);
        }
        #[cfg(unix)]
        unsafe {
            unsafe extern "C" {
                fn munmap(addr: *mut core::ffi::c_void, len: usize) -> i32;
            }
            munmap(self.ptr as *mut core::ffi::c_void, self.len);
        }
    }
}

/// Lazy memory-mapped `.safetensors` reader: parses only the header on `open()`,
/// materializing individual tensors on demand via [`SafetensorsMmap::get`].
pub struct SafetensorsMmap {
    map: MmapFile,
    data_start: usize,
    entries: BTreeMap<String, (crate::dtype::DType, Vec<usize>, usize, usize)>,
}

impl SafetensorsMmap {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let map = MmapFile::open(path)?;
        let bytes = map.as_bytes();
        if bytes.len() < 8 {
            return Err(Error::Msg("safetensors mmap: file shorter than 8 bytes".into()));
        }
        let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let data_start = 8usize
            .checked_add(header_len)
            .ok_or_else(|| Error::Msg("safetensors mmap: header length overflow".into()))?;
        if data_start > bytes.len() {
            return Err(Error::Msg("safetensors mmap: header exceeds file length".into()));
        }
        let header_str = std::str::from_utf8(&bytes[8..data_start])
            .map_err(|e| Error::Msg(format!("safetensors mmap: invalid UTF-8 header: {e}")))?;
        let entries = super::safetensors::parse_header_entries(header_str, bytes.len() - data_start)?;
        Ok(Self {
            map,
            data_start,
            entries,
        })
    }

    pub fn tensor_names(&self) -> Vec<&str> {
        self.entries.keys().map(String::as_str).collect()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Raw mapped byte slice for tensor `name` (zero-copy into the OS mmap!).
    pub fn raw_bytes(&self, name: &str) -> Result<&[u8]> {
        let &(_, _, start, end) = self
            .entries
            .get(name)
            .ok_or_else(|| Error::Msg(format!("safetensors mmap: tensor '{name}' not found")))?;
        Ok(&self.map.as_bytes()[self.data_start + start..self.data_start + end])
    }

    /// Materializes only the requested tensor `name` from the memory-mapped region.
    pub fn get(&self, name: &str) -> Result<Tensor> {
        let (dtype, shape, start, end) = self
            .entries
            .get(name)
            .ok_or_else(|| Error::Msg(format!("safetensors mmap: tensor '{name}' not found")))?;
        let payload = &self.map.as_bytes()[self.data_start + start..self.data_start + end];
        super::safetensors::tensor_from_raw_bytes(payload, *dtype, shape)
    }
}

/// Lazy memory-mapped `.gguf` reader: maps the file into virtual memory and reads headers/tensors
/// without eagerly copying the file into a `Vec<u8>`.
pub struct GgufMmap {
    map: MmapFile,
}

impl GgufMmap {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let map = MmapFile::open(path)?;
        Ok(Self { map })
    }

    pub fn as_bytes(&self) -> &[u8] {
        self.map.as_bytes()
    }

    pub fn parse(&self) -> Result<super::gguf::GGUFFile> {
        super::gguf::GGUFFile::from_bytes(self.map.as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serialize::save_safetensors;

    #[test]
    fn safetensors_mmap_loads_single_tensor_lazily() {
        let dir = std::env::temp_dir();
        let path = dir.join("oxideon_mmap_test.safetensors");
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::from_slice(&[10i32, 20, 30], &[3]).unwrap();
        let mut map = std::collections::HashMap::new();
        map.insert("a".to_string(), a.clone());
        map.insert("b".to_string(), b.clone());
        save_safetensors(&map, &path).unwrap();

        let mm = SafetensorsMmap::open(&path).unwrap();
        assert_eq!(mm.len(), 2);
        assert_eq!(mm.raw_bytes("a").unwrap().len(), 16);
        let got_b = mm.get("b").unwrap();
        assert_eq!(got_b.to_vec::<i32>().unwrap(), vec![10, 20, 30]);
        let _ = std::fs::remove_file(&path);
    }
}
