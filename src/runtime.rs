use std::path::{Path, PathBuf};

use thiserror::Error;

#[derive(Debug)]
pub struct RuntimeSocketLock {
    #[cfg(unix)]
    _file: std::fs::File,
}

#[derive(Debug, Error)]
pub enum RuntimeSocketError {
    #[error("HOME is required when XDG_RUNTIME_DIR is unavailable")]
    MissingHome,
    #[error("control socket path has no parent: {0}")]
    MissingParent(PathBuf),
    #[error("control runtime path must be absolute: {0}")]
    RelativePath(PathBuf),
    #[error("control runtime path refuses symlinks: {0}")]
    Symlink(PathBuf),
    #[error("control runtime path has an unexpected file type: {0}")]
    UnexpectedType(PathBuf),
    #[error("control runtime path is not owned by the current user: {0}")]
    WrongOwner(PathBuf),
    #[error("control runtime permissions are too broad at {path}: {mode:o}")]
    BroadPermissions { path: PathBuf, mode: u32 },
    #[error("another Lanyte kernel owns the control socket lock: {0}")]
    AlreadyRunning(PathBuf),
    #[error("the Lanyte control socket has no active server lock: {0}")]
    MissingServerLock(PathBuf),
    #[error("the connected control server identity is unavailable")]
    PeerIdentityUnavailable,
    #[error(
        "connected control server does not match the locked kernel (expected uid {expected_uid}, pid {expected_pid}; received uid {actual_uid}, pid {actual_pid})"
    )]
    UnexpectedPeer {
        expected_uid: u32,
        expected_pid: u32,
        actual_uid: u32,
        actual_pid: i32,
    },
    #[error("control runtime filesystem error at {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
}

pub fn resolve_control_socket(configured: &Path) -> Result<PathBuf, RuntimeSocketError> {
    if configured != Path::new(lanyte_common::DEFAULT_GATEWAY_SOCKET_PATH) {
        if !configured.is_absolute() {
            return Err(RuntimeSocketError::RelativePath(configured.to_path_buf()));
        }
        return Ok(configured.to_path_buf());
    }

    let runtime_dir = match std::env::var_os("XDG_RUNTIME_DIR") {
        Some(path) if Path::new(&path).is_absolute() => PathBuf::from(path).join("lanyte"),
        _ => {
            let home = std::env::var_os("HOME").ok_or(RuntimeSocketError::MissingHome)?;
            PathBuf::from(home).join(".local/state/lanyte/run")
        }
    };
    Ok(runtime_dir.join("control.sock"))
}

#[cfg(unix)]
pub fn prepare_server_socket(path: &Path) -> Result<RuntimeSocketLock, RuntimeSocketError> {
    use std::io::{Seek as _, Write as _};
    use std::os::unix::fs::{
        DirBuilderExt, FileTypeExt, MetadataExt, OpenOptionsExt, PermissionsExt,
    };

    let parent = path
        .parent()
        .ok_or_else(|| RuntimeSocketError::MissingParent(path.to_path_buf()))?;
    if !parent.is_absolute() {
        return Err(RuntimeSocketError::RelativePath(parent.to_path_buf()));
    }
    validate_user_owned_path_components(parent)?;

    match std::fs::symlink_metadata(parent) {
        Ok(metadata) => validate_owned_directory(parent, &metadata)?,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            let mut builder = std::fs::DirBuilder::new();
            builder.recursive(true).mode(0o700);
            builder
                .create(parent)
                .map_err(|source| RuntimeSocketError::Io {
                    path: parent.to_path_buf(),
                    source,
                })?;
            std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700)).map_err(
                |source| RuntimeSocketError::Io {
                    path: parent.to_path_buf(),
                    source,
                },
            )?;
            let metadata =
                std::fs::symlink_metadata(parent).map_err(|source| RuntimeSocketError::Io {
                    path: parent.to_path_buf(),
                    source,
                })?;
            validate_user_owned_path_components(parent)?;
            validate_owned_directory(parent, &metadata)?;
        }
        Err(source) => {
            return Err(RuntimeSocketError::Io {
                path: parent.to_path_buf(),
                source,
            });
        }
    }

    let lock_path = path.with_extension("lock");
    match std::fs::symlink_metadata(&lock_path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            return Err(RuntimeSocketError::Symlink(lock_path));
        }
        Ok(metadata) if !metadata.is_file() => {
            return Err(RuntimeSocketError::UnexpectedType(lock_path));
        }
        Ok(metadata) => {
            validate_owner(&lock_path, &metadata)?;
            let mode = metadata.mode() & 0o777;
            if mode & 0o077 != 0 {
                return Err(RuntimeSocketError::BroadPermissions {
                    path: lock_path,
                    mode,
                });
            }
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(source) => {
            return Err(RuntimeSocketError::Io {
                path: lock_path,
                source,
            });
        }
    }
    let mut lock_file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .open(&lock_path)
        .map_err(|source| RuntimeSocketError::Io {
            path: lock_path.clone(),
            source,
        })?;
    let path_metadata =
        std::fs::symlink_metadata(&lock_path).map_err(|source| RuntimeSocketError::Io {
            path: lock_path.clone(),
            source,
        })?;
    let file_metadata = lock_file
        .metadata()
        .map_err(|source| RuntimeSocketError::Io {
            path: lock_path.clone(),
            source,
        })?;
    if path_metadata.file_type().is_symlink() {
        return Err(RuntimeSocketError::Symlink(lock_path));
    }
    if !path_metadata.is_file()
        || path_metadata.dev() != file_metadata.dev()
        || path_metadata.ino() != file_metadata.ino()
    {
        return Err(RuntimeSocketError::UnexpectedType(lock_path));
    }
    validate_owner(&lock_path, &file_metadata)?;
    std::fs::set_permissions(&lock_path, std::fs::Permissions::from_mode(0o600)).map_err(
        |source| RuntimeSocketError::Io {
            path: lock_path.clone(),
            source,
        },
    )?;
    if let Err(error) = rustix::fs::flock(
        &lock_file,
        rustix::fs::FlockOperation::NonBlockingLockExclusive,
    ) {
        if error == rustix::io::Errno::WOULDBLOCK {
            return Err(RuntimeSocketError::AlreadyRunning(lock_path));
        }
        return Err(RuntimeSocketError::Io {
            path: lock_path,
            source: std::io::Error::from_raw_os_error(error.raw_os_error()),
        });
    }
    lock_file
        .set_len(0)
        .and_then(|()| lock_file.rewind())
        .and_then(|()| writeln!(lock_file, "{}", std::process::id()))
        .and_then(|()| lock_file.sync_data())
        .map_err(|source| RuntimeSocketError::Io {
            path: lock_path,
            source,
        })?;

    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            Err(RuntimeSocketError::Symlink(path.to_path_buf()))
        }
        Ok(metadata) if metadata.file_type().is_socket() => {
            validate_owner(path, &metadata)?;
            let mode = metadata.permissions().mode() & 0o777;
            if mode & 0o077 != 0 {
                return Err(RuntimeSocketError::BroadPermissions {
                    path: path.to_path_buf(),
                    mode,
                });
            }
            Ok(RuntimeSocketLock { _file: lock_file })
        }
        Ok(_) => Err(RuntimeSocketError::UnexpectedType(path.to_path_buf())),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            Ok(RuntimeSocketLock { _file: lock_file })
        }
        Err(source) => Err(RuntimeSocketError::Io {
            path: path.to_path_buf(),
            source,
        }),
    }
}

#[cfg(unix)]
pub fn verify_client_socket(path: &Path) -> Result<(), RuntimeSocketError> {
    use std::os::unix::fs::{FileTypeExt, MetadataExt};

    let parent = path
        .parent()
        .ok_or_else(|| RuntimeSocketError::MissingParent(path.to_path_buf()))?;
    validate_user_owned_path_components(parent)?;
    let parent_metadata =
        std::fs::symlink_metadata(parent).map_err(|source| RuntimeSocketError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    validate_owned_directory(parent, &parent_metadata)?;

    let metadata = std::fs::symlink_metadata(path).map_err(|source| RuntimeSocketError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    if metadata.file_type().is_symlink() {
        return Err(RuntimeSocketError::Symlink(path.to_path_buf()));
    }
    if !metadata.file_type().is_socket() {
        return Err(RuntimeSocketError::UnexpectedType(path.to_path_buf()));
    }
    validate_owner(path, &metadata)?;
    let mode = metadata.mode() & 0o777;
    if mode & 0o077 != 0 {
        return Err(RuntimeSocketError::BroadPermissions {
            path: path.to_path_buf(),
            mode,
        });
    }
    Ok(())
}

#[cfg(unix)]
pub fn verify_connected_server(
    path: &Path,
    stream: &tokio::net::UnixStream,
) -> Result<(), RuntimeSocketError> {
    verify_client_socket(path)?;
    let expected_pid = locked_server_pid(path)?;
    let credentials = stream
        .peer_cred()
        .map_err(|source| RuntimeSocketError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    let actual_pid = credentials
        .pid()
        .ok_or(RuntimeSocketError::PeerIdentityUnavailable)?;
    let expected_uid = effective_uid();
    if credentials.uid() != expected_uid || actual_pid < 0 || actual_pid as u32 != expected_pid {
        return Err(RuntimeSocketError::UnexpectedPeer {
            expected_uid,
            expected_pid,
            actual_uid: credentials.uid(),
            actual_pid,
        });
    }
    Ok(())
}

#[cfg(unix)]
fn locked_server_pid(path: &Path) -> Result<u32, RuntimeSocketError> {
    use std::io::Read as _;
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt};

    let lock_path = path.with_extension("lock");
    let before =
        std::fs::symlink_metadata(&lock_path).map_err(|source| RuntimeSocketError::Io {
            path: lock_path.clone(),
            source,
        })?;
    if before.file_type().is_symlink() {
        return Err(RuntimeSocketError::Symlink(lock_path));
    }
    if !before.is_file() {
        return Err(RuntimeSocketError::UnexpectedType(lock_path));
    }
    validate_owner(&lock_path, &before)?;
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(rustix::fs::OFlags::NOFOLLOW.bits() as i32)
        .open(&lock_path)
        .map_err(|source| RuntimeSocketError::Io {
            path: lock_path.clone(),
            source,
        })?;
    let opened = file.metadata().map_err(|source| RuntimeSocketError::Io {
        path: lock_path.clone(),
        source,
    })?;
    if before.dev() != opened.dev() || before.ino() != opened.ino() {
        return Err(RuntimeSocketError::UnexpectedType(lock_path));
    }
    match rustix::fs::flock(&file, rustix::fs::FlockOperation::NonBlockingLockExclusive) {
        Err(error) if error == rustix::io::Errno::WOULDBLOCK => {}
        Ok(()) => {
            rustix::fs::flock(&file, rustix::fs::FlockOperation::Unlock).map_err(|error| {
                RuntimeSocketError::Io {
                    path: lock_path.clone(),
                    source: std::io::Error::from_raw_os_error(error.raw_os_error()),
                }
            })?;
            return Err(RuntimeSocketError::MissingServerLock(lock_path));
        }
        Err(error) => {
            return Err(RuntimeSocketError::Io {
                path: lock_path,
                source: std::io::Error::from_raw_os_error(error.raw_os_error()),
            });
        }
    }
    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .map_err(|source| RuntimeSocketError::Io {
            path: lock_path.clone(),
            source,
        })?;
    contents
        .trim()
        .parse::<u32>()
        .map_err(|_| RuntimeSocketError::UnexpectedType(lock_path))
}

#[cfg(unix)]
fn validate_user_owned_path_components(path: &Path) -> Result<(), RuntimeSocketError> {
    use std::os::unix::fs::MetadataExt;

    // System-owned ancestors are outside the user runtime boundary. Within the
    // user-owned suffix, refuse link traversal and writable shared directories.
    let effective_uid = effective_uid();
    for component in path.ancestors() {
        if component.as_os_str().is_empty() {
            break;
        }
        let metadata = match std::fs::symlink_metadata(component) {
            Ok(metadata) => metadata,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => continue,
            Err(source) => {
                return Err(RuntimeSocketError::Io {
                    path: component.to_path_buf(),
                    source,
                });
            }
        };
        if metadata.uid() != effective_uid {
            break;
        }
        if metadata.file_type().is_symlink() {
            return Err(RuntimeSocketError::Symlink(component.to_path_buf()));
        }
        if !metadata.is_dir() {
            return Err(RuntimeSocketError::UnexpectedType(component.to_path_buf()));
        }
        let mode = metadata.mode() & 0o777;
        if mode & 0o022 != 0 {
            return Err(RuntimeSocketError::BroadPermissions {
                path: component.to_path_buf(),
                mode,
            });
        }
    }
    Ok(())
}

#[cfg(unix)]
fn validate_owned_directory(
    path: &Path,
    metadata: &std::fs::Metadata,
) -> Result<(), RuntimeSocketError> {
    use std::os::unix::fs::MetadataExt;

    if metadata.file_type().is_symlink() {
        return Err(RuntimeSocketError::Symlink(path.to_path_buf()));
    }
    if !metadata.is_dir() {
        return Err(RuntimeSocketError::UnexpectedType(path.to_path_buf()));
    }
    validate_owner(path, metadata)?;
    let mode = metadata.mode() & 0o777;
    if mode & 0o077 != 0 {
        return Err(RuntimeSocketError::BroadPermissions {
            path: path.to_path_buf(),
            mode,
        });
    }
    Ok(())
}

#[cfg(unix)]
fn validate_owner(path: &Path, metadata: &std::fs::Metadata) -> Result<(), RuntimeSocketError> {
    use std::os::unix::fs::MetadataExt;

    if metadata.uid() != effective_uid() {
        return Err(RuntimeSocketError::WrongOwner(path.to_path_buf()));
    }
    Ok(())
}

#[cfg(unix)]
fn effective_uid() -> u32 {
    rustix::process::geteuid().as_raw()
}

#[cfg(not(unix))]
pub fn prepare_server_socket(_path: &Path) -> Result<RuntimeSocketLock, RuntimeSocketError> {
    Err(RuntimeSocketError::UnexpectedType(PathBuf::from(
        "Unix domain sockets are unavailable",
    )))
}

#[cfg(not(unix))]
pub fn verify_client_socket(_path: &Path) -> Result<(), RuntimeSocketError> {
    Err(RuntimeSocketError::UnexpectedType(PathBuf::from(
        "Unix domain sockets are unavailable",
    )))
}

#[cfg(not(unix))]
pub fn verify_connected_server(
    _path: &Path,
    _stream: &tokio::net::UnixStream,
) -> Result<(), RuntimeSocketError> {
    Err(RuntimeSocketError::UnexpectedType(PathBuf::from(
        "Unix domain sockets are unavailable",
    )))
}

#[cfg(all(test, unix))]
mod tests {
    use std::os::unix::fs::{symlink, DirBuilderExt, PermissionsExt};

    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let path =
            PathBuf::from("/tmp").join(format!("lanyte-runtime-{tag}-{}", uuid::Uuid::new_v4()));
        let mut builder = std::fs::DirBuilder::new();
        builder.mode(0o700);
        builder.create(&path).expect("temp directory");
        path
    }

    #[test]
    fn server_preparation_rejects_symlink_socket() {
        let root = temp_dir("symlink");
        let target = root.join("target");
        std::fs::write(&target, b"not a socket").expect("target");
        let socket = root.join("control.sock");
        symlink(&target, &socket).expect("symlink");

        assert!(matches!(
            prepare_server_socket(&socket),
            Err(RuntimeSocketError::Symlink(path)) if path == socket
        ));
        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn server_preparation_rejects_user_owned_intermediate_symlink() {
        let root = temp_dir("intermediate-symlink");
        let target = root.join("target");
        let mut builder = std::fs::DirBuilder::new();
        builder.mode(0o700);
        builder.create(&target).expect("target directory");
        let link = root.join("link");
        symlink(&target, &link).expect("directory symlink");
        let socket = link.join("run/control.sock");

        assert!(matches!(
            prepare_server_socket(&socket),
            Err(RuntimeSocketError::Symlink(path)) if path == link
        ));
        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn server_preparation_rejects_broad_runtime_permissions() {
        let root = temp_dir("permissions");
        std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o755))
            .expect("permissions");
        let socket = root.join("control.sock");

        assert!(matches!(
            prepare_server_socket(&socket),
            Err(RuntimeSocketError::BroadPermissions { .. })
        ));
        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn server_preparation_holds_a_singleton_lock() {
        let root = temp_dir("singleton");
        let socket = root.join("control.sock");
        let first = prepare_server_socket(&socket).expect("first server lock");
        assert!(matches!(
            prepare_server_socket(&socket),
            Err(RuntimeSocketError::AlreadyRunning(path)) if path == root.join("control.lock")
        ));
        drop(first);
        prepare_server_socket(&socket).expect("lock should be available after server exit");
        std::fs::remove_dir_all(root).expect("cleanup");
    }

    #[tokio::test]
    async fn client_verifies_the_connected_locked_server() {
        let root = temp_dir("connected-server");
        let socket = root.join("control.sock");
        let _lock = prepare_server_socket(&socket).expect("server lock");
        let listener = tokio::net::UnixListener::bind(&socket).expect("listener");
        std::fs::set_permissions(&socket, std::fs::Permissions::from_mode(0o600))
            .expect("socket permissions");
        let client = tokio::net::UnixStream::connect(&socket)
            .await
            .expect("client connect");
        let (_server, _) = listener.accept().await.expect("server accept");

        verify_connected_server(&socket, &client).expect("locked server identity");
        drop(listener);
        std::fs::remove_dir_all(root).expect("cleanup");
    }
}
