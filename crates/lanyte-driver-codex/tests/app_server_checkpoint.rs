use std::path::PathBuf;
use std::time::Duration;

use lanyte_driver_codex::{CloseOutcome, CodexAppServerDriver, CodexLaunchSpec};
use lanyte_mission::NormalizedHarnessEvent;
use uuid::Uuid;

fn fixture_binary() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fake-codex-app-server.py")
}

fn exit_fixture_binary() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fake-codex-exits.py")
}

fn overflow_fixture_binary() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fake-codex-overflow.py")
}

#[tokio::test]
async fn launch_observe_close_against_fake_app_server() {
    let root = std::env::temp_dir().join(format!("lanyte-fake-root-{}", Uuid::new_v4()));
    let workspace = root.join("workspace");
    let pin_dir = root.join("pins");
    std::fs::create_dir_all(&workspace).unwrap();
    std::fs::create_dir_all(&pin_dir).unwrap();
    std::fs::copy(fixture_binary(), workspace.join("fake-codex")).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(
            workspace.join("fake-codex"),
            std::fs::Permissions::from_mode(0o755),
        )
        .unwrap();
    }

    let driver = CodexAppServerDriver::new(CodexLaunchSpec {
        workspace: workspace.clone(),
        allowed_root: root.clone(),
        pin_dir,
        binary_path: Some(workspace.join("fake-codex")),
    });
    let mut session = driver.create(Uuid::new_v4()).await.expect("create");
    assert!(!session.harness_session_id.is_empty());

    tokio::time::sleep(Duration::from_millis(50)).await;
    let mut saw_started = false;
    let mut saw_tool = false;
    let mut saw_exited = false;
    for _ in 0..12 {
        match session.observe().await.expect("observe") {
            Some(NormalizedHarnessEvent::Started { .. }) => saw_started = true,
            Some(NormalizedHarnessEvent::ToolProposed { tool, .. }) if tool == "shell" => {
                saw_tool = true;
            }
            Some(NormalizedHarnessEvent::Exited { success: true, .. }) => saw_exited = true,
            _ => {}
        }
        if saw_started && saw_tool && saw_exited {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(saw_started, "expected a started event");
    assert!(saw_tool, "expected a tool proposal from the fake server");
    assert!(saw_exited, "expected an exited event from the fake server");
    let outcome = session.close().await.expect("close");
    assert!(
        matches!(outcome, CloseOutcome::Terminated(_)),
        "live child must be terminated by close"
    );
    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test]
async fn close_reports_already_exited_when_child_has_left() {
    let root = std::env::temp_dir().join(format!("lanyte-fake-exit-root-{}", Uuid::new_v4()));
    let workspace = root.join("workspace");
    let pin_dir = root.join("pins");
    std::fs::create_dir_all(&workspace).unwrap();
    std::fs::create_dir_all(&pin_dir).unwrap();
    std::fs::copy(exit_fixture_binary(), workspace.join("fake-codex")).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(
            workspace.join("fake-codex"),
            std::fs::Permissions::from_mode(0o755),
        )
        .unwrap();
    }

    let driver = CodexAppServerDriver::new(CodexLaunchSpec {
        workspace: workspace.clone(),
        allowed_root: root.clone(),
        pin_dir,
        binary_path: Some(workspace.join("fake-codex")),
    });
    let mut session = driver.create(Uuid::new_v4()).await.expect("create");
    let mut saw_exit = false;
    for _ in 0..20 {
        if session.poll_exit().expect("poll").is_some() {
            saw_exit = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(saw_exit, "expected the exit fixture to leave before close");
    let outcome = session.close().await.expect("close");
    assert!(
        matches!(outcome, CloseOutcome::AlreadyExited(status) if status.success()),
        "prior natural exit must not be reported as terminated"
    );
    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test]
async fn observation_overflow_is_not_a_complete_stream() {
    let root = std::env::temp_dir().join(format!("lanyte-fake-overflow-root-{}", Uuid::new_v4()));
    let workspace = root.join("workspace");
    let pin_dir = root.join("pins");
    std::fs::create_dir_all(&workspace).unwrap();
    std::fs::create_dir_all(&pin_dir).unwrap();
    std::fs::copy(overflow_fixture_binary(), workspace.join("fake-codex")).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(
            workspace.join("fake-codex"),
            std::fs::Permissions::from_mode(0o755),
        )
        .unwrap();
    }

    let driver = CodexAppServerDriver::new(CodexLaunchSpec {
        workspace: workspace.clone(),
        allowed_root: root.clone(),
        pin_dir,
        binary_path: Some(workspace.join("fake-codex")),
    });
    let mut session = driver.create(Uuid::new_v4()).await.expect("create");
    let mut overflowed = false;
    for _ in 0..40 {
        if session.overflowed() {
            overflowed = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(overflowed, "300 tool events must set the overflow flag");
    let mut drained = 0usize;
    while session.observe().await.expect("observe").is_some() {
        drained += 1;
    }
    assert!(drained <= 256, "overflow must drop oldest events");
    assert!(
        session.overflowed(),
        "overflow must remain after drain so observe cannot look complete"
    );
    let _ = session.close().await;
    let _ = std::fs::remove_dir_all(root);
}
