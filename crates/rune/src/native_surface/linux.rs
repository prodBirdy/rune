//! DMA-BUF ownership boundary, the Linux sibling of `macos.rs` and `win32.rs`. A producer (a
//! decoder, or mpv rendering through OpenGL on EGL) writes into a single-plane dmabuf (from GBM,
//! say) and hands its fd over; Rune imports it once on its own Vulkan device and draws it with no
//! copy. The producer must have finished writing before the push (wait on a GL fence or
//! `glFinish`), and may write into the buffer again once `in_use` turns false: Rune holds a frame
//! until the GPU has finished every draw of it.
use super::*;
use std::collections::HashMap;
use std::os::fd::{AsFd, BorrowedFd, OwnedFd};
use std::sync::Weak;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// `DRM_FORMAT_ARGB8888` (bytes B, G, R, A) and `DRM_FORMAT_XRGB8888`.
pub const DRM_FORMAT_ARGB8888: u32 = u32::from_le_bytes(*b"AR24");
pub const DRM_FORMAT_XRGB8888: u32 = u32::from_le_bytes(*b"XR24");
/// `DRM_FORMAT_ABGR8888` (bytes R, G, B, A: OpenGL's RGBA8) and `DRM_FORMAT_XBGR8888`.
pub const DRM_FORMAT_ABGR8888: u32 = u32::from_le_bytes(*b"AB24");
pub const DRM_FORMAT_XBGR8888: u32 = u32::from_le_bytes(*b"XB24");

struct Buffer {
    fd: OwnedFd,
    /// Identity for the importer's cache: fd numbers are reused once closed.
    id: u64,
    size: [u32; 2],
    format: wgpu::TextureFormat,
    modifier: u64,
    stride: u32,
    offset: u32,
    color: ColorInfo,
}

/// A frame in a single-plane, 8-bit four-channel dmabuf. Clones share it; keep one and push
/// clones, and reuse the buffer once [`DmaBuf::in_use`] is false.
#[derive(Clone)]
pub struct DmaBuf(Arc<Buffer>);

impl DmaBuf {
    /// Keep a duplicate of `fd`; the caller keeps its own. `fourcc` is one of the
    /// `DRM_FORMAT_*8888` codes above, `modifier` the buffer's DRM format modifier (from
    /// `gbm_bo_get_modifier`), `stride` and `offset` plane 0's.
    ///
    /// # Safety
    /// `fd` must be a dmabuf of that layout, on the GPU Rune draws with.
    pub unsafe fn from_fd(fd: BorrowedFd<'_>, size: [u32; 2], fourcc: u32, modifier: u64, stride: u32, offset: u32, color: ColorInfo) -> Result<Self, String> {
        if size.contains(&0) || size.iter().any(|v| *v > 16384) {
            return Err("invalid video dimensions".into());
        }
        let format = match fourcc {
            DRM_FORMAT_ARGB8888 | DRM_FORMAT_XRGB8888 => wgpu::TextureFormat::Bgra8Unorm,
            DRM_FORMAT_ABGR8888 | DRM_FORMAT_XBGR8888 => wgpu::TextureFormat::Rgba8Unorm,
            other => return Err(format!("unsupported dmabuf format {:?}; use ARGB8888 or ABGR8888", other.to_le_bytes().map(char::from))),
        };
        let fd = fd.try_clone_to_owned().map_err(|e| format!("dup dmabuf fd: {e}"))?;
        Ok(Self(Arc::new(Buffer { fd, id: NEXT_ID.fetch_add(1, Ordering::Relaxed), size, format, modifier, stride, offset, color })))
    }

    /// Whether a pushed clone is still held (queued, on screen, or being drawn by the GPU).
    pub fn in_use(&self) -> bool {
        Arc::strong_count(&self.0) > 1
    }

    pub(crate) fn info(&self) -> ([u32; 2], Format, ColorInfo) {
        (self.0.size, Format::Bgra, self.0.color)
    }
}

/// Imports dmabufs on Rune's Vulkan device and caches them by identity, as `CVMetalTextureCache`
/// does for IOSurfaces. `flush` drops what no producer holds any more.
pub(crate) struct Importer {
    device: wgpu::Device,
    textures: HashMap<u64, (Weak<Buffer>, wgpu::Texture)>,
}

impl Importer {
    pub fn new(device: &wgpu::Device) -> Result<Self, String> {
        if unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }.is_none() {
            return Err("NativeSurface dmabufs need wgpu's Vulkan backend".into());
        }
        if !device.features().contains(wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF) {
            return Err("this Vulkan driver can't import dmabufs (VK_EXT_external_memory_dma_buf, VK_EXT_image_drm_format_modifier)".into());
        }
        Ok(Self { device: device.clone(), textures: HashMap::new() })
    }

    pub fn flush(&mut self) {
        self.textures.retain(|_, (weak, _)| weak.strong_count() > 0);
    }

    pub fn import(&mut self, frame: &DmaBuf) -> Result<[wgpu::Texture; 2], String> {
        let b = &frame.0;
        if let Some((_, t)) = self.textures.get(&b.id) {
            return Ok([t.clone(), t.clone()]);
        }
        let desc = pipeline::descriptor(b.size[0], b.size[1], b.format);
        let hal_desc = wgpu::hal::TextureDescriptor {
            label: Some("rune dmabuf"),
            size: desc.size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: b.format,
            usage: wgpu::TextureUses::RESOURCE,
            memory_flags: wgpu::hal::MemoryFlags::empty(),
            view_formats: vec![],
        };
        let fd = b.fd.as_fd().try_clone_to_owned().map_err(|e| format!("dup dmabuf fd: {e}"))?;
        let hal = unsafe {
            let dev = self.device.as_hal::<wgpu::hal::api::Vulkan>().ok_or("not a Vulkan device")?;
            dev.texture_from_dmabuf_fd(fd, &hal_desc, b.modifier, b.stride as u64, b.offset as u64)
                .map_err(|e| format!("importing the dmabuf (modifier {:#x}) into Vulkan: {e}", b.modifier))?
        };
        // The producer finished writing before the push, and the image is only ever sampled.
        let texture = unsafe { self.device.create_texture_from_hal::<wgpu::hal::api::Vulkan>(hal, &desc, wgpu::TextureUses::RESOURCE) };
        self.textures.insert(b.id, (Arc::downgrade(b), texture.clone()));
        Ok([texture.clone(), texture])
    }
}
