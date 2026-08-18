//! Kernel-owned process-group membership via sysprims plus a PGID census.

use super::ProcessTreeKill;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessTreeHandle {
    pub pgid: u32,
    pub leader: u32,
    pub born_unix_ms: Option<u64>,
}

pub fn parse_process_tree_ref(tree_ref: &str) -> Option<u32> {
    parse_process_tree_handle(tree_ref).map(|handle| handle.pgid)
}

pub fn parse_process_tree_handle(tree_ref: &str) -> Option<ProcessTreeHandle> {
    let mut pgid = None;
    let mut leader = None;
    let mut born_unix_ms = None;
    for part in tree_ref.split(';') {
        if let Some(value) = part.strip_prefix("pgid:") {
            pgid = Some(parse_safe_pid(value)?);
        } else if let Some(value) = part.strip_prefix("leader:") {
            leader = Some(parse_safe_pid(value)?);
        } else if let Some(value) = part.strip_prefix("born:") {
            born_unix_ms = Some(value.parse().ok()?);
        } else {
            return None;
        }
    }
    let pgid = pgid?;
    Some(ProcessTreeHandle {
        leader: leader.unwrap_or(pgid),
        pgid,
        born_unix_ms,
    })
}

pub fn format_process_tree_ref(pgid: u32) -> String {
    format_process_tree_handle(&ProcessTreeHandle {
        pgid,
        leader: pgid,
        born_unix_ms: None,
    })
}

pub fn format_process_tree_handle(handle: &ProcessTreeHandle) -> String {
    match handle.born_unix_ms {
        Some(born) => format!("pgid:{};leader:{};born:{born}", handle.pgid, handle.leader),
        None => format!("pgid:{};leader:{}", handle.pgid, handle.leader),
    }
}

pub fn capture_process_tree_handle(leader: u32) -> Option<ProcessTreeHandle> {
    let pgid = process_group_id(leader).unwrap_or(leader);
    let born_unix_ms = sysprims_proc::get_process(leader)
        .ok()?
        .start_time_unix_ms?;
    Some(ProcessTreeHandle {
        pgid,
        leader,
        born_unix_ms: Some(born_unix_ms),
    })
}

fn process_group_id(pid: u32) -> Option<u32> {
    #[cfg(unix)]
    {
        let raw = unsafe { libc::getpgid(pid as libc::pid_t) };
        if raw > 0 {
            return Some(raw as u32);
        }
    }
    None
}

fn parse_safe_pid(value: &str) -> Option<u32> {
    let parsed = value.parse::<u32>().ok()?;
    if parsed == 0 || parsed > i32::MAX as u32 {
        return None;
    }
    Some(parsed)
}

fn classify_membership(
    handle: &ProcessTreeHandle,
    census: Result<Vec<u32>, String>,
    leader_birth: Result<Option<u64>, String>,
) -> ProcessTreeKill {
    let Some(expected_birth) = handle.born_unix_ms else {
        return ProcessTreeKill::Unknown;
    };
    match leader_birth {
        Err(_) => return ProcessTreeKill::Unknown,
        Ok(Some(actual)) if actual != expected_birth => return ProcessTreeKill::Unknown,
        Ok(Some(_)) | Ok(None) => {}
    }
    match census {
        Err(_) => ProcessTreeKill::Unknown,
        Ok(members) if members.is_empty() => ProcessTreeKill::Cleared,
        Ok(_) => ProcessTreeKill::Survivors,
    }
}

pub fn live_tree_members(tree_ref: &str) -> Result<Vec<u32>, String> {
    let handle = parse_process_tree_handle(tree_ref)
        .ok_or_else(|| "process_tree_ref is not an ownership-bound process group".to_owned())?;
    match probe_process_tree(tree_ref) {
        ProcessTreeKill::Survivors => group_member_pids(handle.pgid),
        ProcessTreeKill::Cleared => Ok(Vec::new()),
        ProcessTreeKill::Unknown | ProcessTreeKill::KillDispatched => {
            Err("process-tree membership is unknown".to_owned())
        }
    }
}

pub fn probe_process_tree(tree_ref: &str) -> ProcessTreeKill {
    let Some(handle) = parse_process_tree_handle(tree_ref) else {
        return ProcessTreeKill::Unknown;
    };
    let census = group_member_pids(handle.pgid);
    let leader_birth = match sysprims_proc::get_process(handle.leader) {
        Ok(info) => Ok(info.start_time_unix_ms),
        Err(sysprims_core::SysprimsError::NotFound { .. }) => Ok(None),
        Err(err) => Err(err.to_string()),
    };
    classify_membership(&handle, census, leader_birth)
}

pub fn terminate_process_tree(tree_ref: &str) -> ProcessTreeKill {
    let Some(handle) = parse_process_tree_handle(tree_ref) else {
        return ProcessTreeKill::Unknown;
    };
    match probe_process_tree(tree_ref) {
        ProcessTreeKill::Unknown => return ProcessTreeKill::Unknown,
        ProcessTreeKill::Cleared => return ProcessTreeKill::Cleared,
        ProcessTreeKill::Survivors | ProcessTreeKill::KillDispatched => {}
    }
    let dispatched = sysprims_signal::force_kill_group(handle.pgid).is_ok();
    if !dispatched {
        match live_tree_members(tree_ref) {
            Ok(members) => {
                for pid in members {
                    let _ = sysprims_signal::force_kill(pid);
                }
            }
            Err(_) => return ProcessTreeKill::Unknown,
        }
    }
    std::thread::sleep(std::time::Duration::from_millis(200));
    #[cfg(unix)]
    unsafe {
        libc::waitpid(
            handle.leader as libc::pid_t,
            std::ptr::null_mut(),
            libc::WNOHANG,
        );
    }
    match live_tree_members(tree_ref) {
        Ok(members) if members.is_empty() => ProcessTreeKill::Cleared,
        Ok(_) if dispatched => ProcessTreeKill::Survivors,
        Ok(_) => ProcessTreeKill::KillDispatched,
        Err(_) => ProcessTreeKill::Unknown,
    }
}

fn group_member_pids(pgid: u32) -> Result<Vec<u32>, String> {
    #[cfg(target_os = "macos")]
    {
        macos_group_members(pgid)
    }
    #[cfg(target_os = "linux")]
    {
        linux_group_members(pgid)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        let _ = pgid;
        Err("process-group census is not available on this platform".to_owned())
    }
}

#[cfg(target_os = "macos")]
fn macos_group_members(pgid: u32) -> Result<Vec<u32>, String> {
    const PROC_PGRP_ONLY: u32 = 2;
    unsafe extern "C" {
        fn proc_listpids(
            type_: u32,
            typeinfo: u32,
            buffer: *mut libc::c_void,
            buffersize: libc::c_int,
        ) -> libc::c_int;
    }
    let needed = unsafe { proc_listpids(PROC_PGRP_ONLY, pgid, std::ptr::null_mut(), 0) };
    if needed < 0 {
        return Err(format!(
            "proc_listpids probe failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    if needed == 0 {
        return Ok(Vec::new());
    }
    let mut pids = vec![0i32; (needed as usize / std::mem::size_of::<i32>()) + 8];
    let written = unsafe {
        proc_listpids(
            PROC_PGRP_ONLY,
            pgid,
            pids.as_mut_ptr().cast(),
            (pids.len() * std::mem::size_of::<i32>()) as libc::c_int,
        )
    };
    if written < 0 {
        return Err(format!(
            "proc_listpids read failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    let count = written as usize / std::mem::size_of::<i32>();
    Ok(pids
        .into_iter()
        .take(count)
        .filter(|pid| *pid > 0)
        .map(|pid| pid as u32)
        .collect())
}

#[cfg(target_os = "linux")]
fn linux_group_members(pgid: u32) -> Result<Vec<u32>, String> {
    let mut pids = Vec::new();
    let entries = std::fs::read_dir("/proc").map_err(|err| err.to_string())?;
    for entry in entries {
        let entry = entry.map_err(|err| err.to_string())?;
        let file_name = entry.file_name();
        let Some(pid) = file_name
            .to_str()
            .and_then(|value| value.parse::<u32>().ok())
        else {
            continue;
        };
        let stat = match std::fs::read_to_string(entry.path().join("stat")) {
            Ok(stat) => stat,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => continue,
            Err(err) => return Err(format!("unreadable /proc/{pid}/stat: {err}")),
        };
        let Some(close) = stat.rfind(')') else {
            return Err(format!("malformed /proc/{pid}/stat"));
        };
        let mut fields = stat[close + 1..].split_whitespace();
        let _state = fields.next();
        let _ppid = fields.next();
        let Some(pgrp) = fields.next().and_then(|value| value.parse::<u32>().ok()) else {
            return Err(format!("malformed /proc/{pid}/stat pgrp"));
        };
        if pgrp == pgid {
            pids.push(pid);
        }
    }
    Ok(pids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_and_unborn_handles_fail_closed() {
        assert!(parse_process_tree_handle("pgid:0").is_none());
        assert_eq!(probe_process_tree("pgid:1"), ProcessTreeKill::Unknown);
        assert_eq!(
            probe_process_tree("pgid:1;leader:1"),
            ProcessTreeKill::Unknown
        );
    }

    #[test]
    fn owned_absent_group_is_cleared() {
        let handle = ProcessTreeHandle {
            pgid: i32::MAX as u32,
            leader: i32::MAX as u32,
            born_unix_ms: Some(1),
        };
        assert_eq!(
            probe_process_tree(&format_process_tree_handle(&handle)),
            ProcessTreeKill::Cleared
        );
    }

    #[test]
    fn census_and_lookup_errors_fail_closed() {
        let handle = ProcessTreeHandle {
            pgid: 7,
            leader: 7,
            born_unix_ms: Some(9),
        };
        assert_eq!(
            classify_membership(&handle, Err("census".to_owned()), Ok(Some(9))),
            ProcessTreeKill::Unknown
        );
        assert_eq!(
            classify_membership(&handle, Ok(Vec::new()), Err("leader".to_owned())),
            ProcessTreeKill::Unknown
        );
        assert_eq!(
            classify_membership(&handle, Ok(vec![42]), Ok(None)),
            ProcessTreeKill::Survivors
        );
        assert_eq!(
            classify_membership(&handle, Ok(Vec::new()), Ok(None)),
            ProcessTreeKill::Cleared
        );
        let mut reused = handle.clone();
        reused.born_unix_ms = Some(1);
        assert_eq!(
            classify_membership(&reused, Ok(vec![7]), Ok(Some(9))),
            ProcessTreeKill::Unknown
        );
    }

    #[test]
    fn current_process_group_is_survivors_when_owned() {
        let pid = std::process::id();
        let Some(handle) = capture_process_tree_handle(pid) else {
            return;
        };
        assert_eq!(
            probe_process_tree(&format_process_tree_handle(&handle)),
            ProcessTreeKill::Survivors
        );
        let mut reused = handle.clone();
        reused.born_unix_ms = Some(1);
        assert_eq!(
            probe_process_tree(&format_process_tree_handle(&reused)),
            ProcessTreeKill::Unknown
        );
    }
}
