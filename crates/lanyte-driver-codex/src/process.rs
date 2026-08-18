//! Kernel-owned process-tree membership via sysprims.

use super::ProcessTreeKill;

pub fn parse_process_tree_ref(tree_ref: &str) -> Option<u32> {
    let pid = tree_ref.strip_prefix("pgid:")?;
    let parsed = pid.parse::<u32>().ok()?;
    if parsed == 0 || parsed > i32::MAX as u32 {
        return None;
    }
    Some(parsed)
}

pub fn format_process_tree_ref(pgid: u32) -> String {
    format!("pgid:{pgid}")
}

pub fn live_tree_members(tree_ref: &str) -> Result<Vec<u32>, String> {
    let pgid = parse_process_tree_ref(tree_ref)
        .ok_or_else(|| "process_tree_ref is not a kernel process group".to_owned())?;
    let mut members = Vec::new();
    if sysprims_proc::get_process(pgid).is_ok() {
        members.push(pgid);
    }
    if let Ok(tree) = sysprims_proc::descendants(pgid, 16, None) {
        for level in tree.levels {
            for process in level.processes {
                if process.pid != 0
                    && process.pid != pgid
                    && sysprims_proc::get_process(process.pid).is_ok()
                    && !members.contains(&process.pid)
                {
                    members.push(process.pid);
                }
            }
        }
    }
    Ok(members)
}

pub fn probe_process_tree(tree_ref: &str) -> ProcessTreeKill {
    match live_tree_members(tree_ref) {
        Ok(members) if members.is_empty() => ProcessTreeKill::Cleared,
        Ok(_) => ProcessTreeKill::Survivors,
        Err(_) => ProcessTreeKill::Unknown,
    }
}

pub fn terminate_process_tree(tree_ref: &str) -> ProcessTreeKill {
    let Some(pgid) = parse_process_tree_ref(tree_ref) else {
        return ProcessTreeKill::Unknown;
    };
    let dispatched = sysprims_signal::force_kill_group(pgid).is_ok();
    if !dispatched {
        if let Ok(members) = live_tree_members(tree_ref) {
            for pid in members {
                let _ = sysprims_signal::force_kill(pid);
            }
        }
    }
    std::thread::sleep(std::time::Duration::from_millis(50));
    match live_tree_members(tree_ref) {
        Ok(members) if members.is_empty() => ProcessTreeKill::Cleared,
        Ok(_) if dispatched => ProcessTreeKill::Survivors,
        Ok(_) => ProcessTreeKill::KillDispatched,
        Err(_) => {
            if dispatched {
                ProcessTreeKill::KillDispatched
            } else {
                ProcessTreeKill::Unknown
            }
        }
    }
}
