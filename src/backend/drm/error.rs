use crate::backend::SwapBuffersError;
use drm::control::{Mode, RawResourceHandle, connector, crtc, plane};
use std::{
    io::{self, ErrorKind},
    path::PathBuf,
};

/// DRM access error
#[derive(Debug, thiserror::Error)]
#[error("DRM access error: {errmsg} on device `{dev:?}` ({source:})")]
pub struct AccessError {
    /// Error message associated to the access error
    pub(crate) errmsg: &'static str,
    /// Device on which the error was generated
    pub(crate) dev: Option<PathBuf>,
    /// Underlying device error
    #[source]
    pub source: io::Error,
}

/// Errors thrown by the [`DrmDevice`](crate::backend::drm::DrmDevice)
/// and the [`DrmSurface`](crate::backend::drm::DrmSurface).
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// Unable to acquire DRM master
    #[error("Failed to acquire DRM master")]
    DrmMasterFailed,
    /// The `DrmDevice` encountered an access error
    #[error(transparent)]
    Access(#[from] AccessError),
    /// Unable to determine device id of drm device
    #[error("Unable to determine device id of drm device")]
    UnableToGetDeviceId(#[source] rustix::io::Errno),
    /// Device is currently paused
    #[error("Device is currently paused, operation rejected")]
    DeviceInactive,
    /// Mode is not compatible with all given connectors
    #[error("Mode `{0:?}` is not compatible with all given connectors")]
    ModeNotSuitable(Mode),
    /// The given crtc is already in use by another surface
    #[error("Crtc `{0:?}` is already in use by another surface")]
    CrtcAlreadyInUse(crtc::Handle),
    /// This operation would result in a surface without connectors.
    #[error("Surface of crtc `{0:?}` would have no connectors, which is not accepted")]
    SurfaceWithoutConnectors(crtc::Handle),
    /// The given plane cannot be used with the given crtc
    #[error("Plane `{1:?}` is not compatible for use with crtc `{0:?}`")]
    PlaneNotCompatible(crtc::Handle, plane::Handle),
    /// The given configuration does not specify a plane which is not supported by the underlying implementation
    #[error("No Plane has been specified which is not supported by the underlying implementation")]
    NoPlane,
    /// The given plane is not a primary plane and therefor not supported by the underlying implementation
    #[error("Non-Primary Planes (provided was `{0:?}`) are not available for use with legacy devices")]
    NonPrimaryPlane(plane::Handle),
    /// The given plane does not allow to clear the framebuffer
    #[error("Clearing the framebuffer on plane `{0:?}` is not supported")]
    NoFramebuffer(plane::Handle),
    /// The configuration is not supported on the given plane
    #[error("The configuration is not supported on plane `{0:?}`")]
    UnsupportedPlaneConfiguration(plane::Handle),
    /// No encoder was found for a given connector on the set crtc
    #[error("No encoder found for the given connector '{connector:?}' on crtc `{crtc:?}`")]
    NoSuitableEncoder {
        /// Connector
        connector: connector::Handle,
        /// CRTC
        crtc: crtc::Handle,
    },
    /// Unknown connector handle
    #[error("The connector ({0:?}) is unknown")]
    UnknownConnector(connector::Handle),
    /// Unknown crtc handle
    #[error("The crtc ({0:?}) is unknown")]
    UnknownCrtc(crtc::Handle),
    /// Unknown crtc handle
    #[error("The plane ({0:?}) is unknown")]
    UnknownPlane(plane::Handle),
    /// The DrmDevice is missing a required property
    #[error("The DrmDevice is missing a required property '{name}' for handle ({handle:?})")]
    UnknownProperty {
        /// Property handle
        handle: RawResourceHandle,
        /// Property name
        name: &'static str,
    },
    /// Atomic Test failed for new properties
    #[error("Atomic Test failed for new properties on crtc ({0:?})")]
    TestFailed(crtc::Handle),
}

fn temporary_access_error(source: &io::Error) -> bool {
    matches!(
        source.kind(),
        ErrorKind::PermissionDenied
            | ErrorKind::WouldBlock
            | ErrorKind::Interrupted
            | ErrorKind::ResourceBusy
    )
}

/// Preserve transient failures of TEST_ONLY requests so callers can retry after a VT/master
/// change or a busy commit, instead of treating the requested configuration as unsupported.
/// Other failures retain the historical `TestFailed` result.
pub(super) fn atomic_test_error(
    crtc: crtc::Handle,
    dev: Option<PathBuf>,
    errmsg: &'static str,
    source: io::Error,
) -> Error {
    if temporary_access_error(&source) {
        Error::Access(AccessError { errmsg, dev, source })
    } else {
        Error::TestFailed(crtc)
    }
}

impl From<Error> for SwapBuffersError {
    #[inline]
    fn from(err: Error) -> SwapBuffersError {
        match err {
            x @ Error::DeviceInactive | x @ Error::DrmMasterFailed => {
                SwapBuffersError::TemporaryFailure(Box::new(x))
            }
            Error::Access(AccessError {
                errmsg, dev, source, ..
            }) if temporary_access_error(&source) => {
                SwapBuffersError::TemporaryFailure(Box::new(Error::Access(AccessError {
                    errmsg,
                    dev,
                    source,
                })))
            }
            x => SwapBuffersError::ContextLost(Box::new(x)),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::*;

    #[test]
    fn atomic_test_retains_temporary_errno_and_access_context() {
        let crtc = crtc::Handle::from(NonZeroU32::new(42).unwrap());
        let dev = Some(PathBuf::from("/dev/dri/card-test"));
        for errno in [libc::EACCES, libc::EPERM, libc::EAGAIN, libc::EINTR, libc::EBUSY] {
            let error = atomic_test_error(
                crtc,
                dev.clone(),
                "Error testing connector color state",
                io::Error::from_raw_os_error(errno),
            );
            let Error::Access(access) = &error else {
                panic!("temporary errno {errno} became {error:?}");
            };
            assert_eq!(access.dev, dev);
            assert_eq!(access.errmsg, "Error testing connector color state");
            assert_eq!(access.source.raw_os_error(), Some(errno));
            assert!(matches!(
                SwapBuffersError::from(error),
                SwapBuffersError::TemporaryFailure(_)
            ));
        }

        // ErrorKind also works when a caller supplies an error without a raw OS errno.
        let error = atomic_test_error(crtc, None, "Error testing state", ErrorKind::ResourceBusy.into());
        assert!(matches!(
            SwapBuffersError::from(error),
            SwapBuffersError::TemporaryFailure(_)
        ));
    }

    #[test]
    fn atomic_test_permanent_failures_keep_test_failed_semantics() {
        let crtc = crtc::Handle::from(NonZeroU32::new(42).unwrap());
        for errno in [libc::EINVAL, libc::EIO, libc::ENODEV, libc::ENOMEM] {
            let error = atomic_test_error(
                crtc,
                None,
                "Error testing connector color state",
                io::Error::from_raw_os_error(errno),
            );
            assert!(matches!(error, Error::TestFailed(handle) if handle == crtc));
            assert!(matches!(
                SwapBuffersError::from(error),
                SwapBuffersError::ContextLost(_)
            ));
        }
    }
}
