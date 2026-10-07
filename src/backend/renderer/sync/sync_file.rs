//! Linux sync-file operations used to combine reads from independent GPU contexts.

use std::{
    io,
    os::fd::{AsRawFd, BorrowedFd, FromRawFd, OwnedFd},
};

#[repr(C)]
struct SyncMergeData {
    name: [u8; 32],
    fd2: i32,
    fence: i32,
    flags: u32,
    pad: u32,
}

const SYNC_IOC_MERGE: rustix::ioctl::Opcode = rustix::ioctl::opcode::read_write::<SyncMergeData>(b'>', 3);

pub(crate) fn merge(first: BorrowedFd<'_>, second: BorrowedFd<'_>) -> io::Result<OwnedFd> {
    let mut data = SyncMergeData {
        name: [0; 32],
        fd2: second.as_raw_fd(),
        fence: -1,
        flags: 0,
        pad: 0,
    };
    data.name[..12].copy_from_slice(b"smithay-read");
    unsafe {
        rustix::ioctl::ioctl(
            first,
            rustix::ioctl::Updater::<SYNC_IOC_MERGE, SyncMergeData>::new(&mut data),
        )?;
        Ok(OwnedFd::from_raw_fd(data.fence))
    }
}

pub(crate) fn is_signaled(fd: BorrowedFd<'_>) -> bool {
    let mut fds = [rustix::event::PollFd::new(&fd, rustix::event::PollFlags::IN)];
    rustix::event::poll(
        &mut fds,
        Some(&rustix::time::Timespec {
            tv_sec: 0,
            tv_nsec: 0,
        }),
    )
    .is_ok_and(|n| {
        n != 0
            && fds[0]
                .revents()
                .intersects(rustix::event::PollFlags::IN | rustix::event::PollFlags::ERR)
    })
}

pub(crate) fn wait(fd: BorrowedFd<'_>) {
    loop {
        let mut fds = [rustix::event::PollFd::new(&fd, rustix::event::PollFlags::IN)];
        match rustix::event::poll(&mut fds, None) {
            Ok(_)
                if fds[0]
                    .revents()
                    .intersects(rustix::event::PollFlags::IN | rustix::event::PollFlags::ERR) =>
            {
                return;
            }
            // EINTR does not establish completion. An owned, valid sync_file cannot become
            // invalid while it is held here, so never replace a failed wait with release.
            Err(rustix::io::Errno::INTR) => continue,
            Err(err) => {
                super::failed_wait(err);
            }
            Ok(_) => {
                super::failed_wait(fds[0].revents());
            }
        }
    }
}
