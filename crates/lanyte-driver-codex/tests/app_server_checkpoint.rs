use std::path::PathBuf;
use std::time::Duration;

use lanyte_driver_codex::{CodexAppServerDriver, CodexLaunchSpec};
use lanyte_mission::NormalizedHarnessEvent;
use uuid::Uuid;

fn fixture_binary() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fake-codex-app-server.py")
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
    for _ in 0..12 {
        match session.observe().await.expect("observe") {
            Some(NormalizedHarnessEvent::Started { .. }) => saw_started = true,
            Some(NormalizedHarnessEvent::ToolProposed { tool, .. }) if tool == "shell" => {
                saw_tool = true;
            }
            _ => {}
        }
        if saw_started && saw_tool {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(saw_started, "expected a started event");
    assert!(saw_tool, "expected a tool proposal from the fake server");
    session.close().await.expect("close");
    let _ = std::fs::remove_dir_all(root);
}
