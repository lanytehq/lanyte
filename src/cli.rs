use std::path::Path;
use std::time::Duration;

use bytes::BytesMut;
use chrono::{DateTime, Utc};
use clap::{Args, Parser, Subcommand, ValueEnum};
use ipcprims::frame::{decode_frame, encode_frame, DEFAULT_MAX_PAYLOAD};
use ipcprims::peer::handshake::async_handshake_client_with_config;
use ipcprims::peer::HandshakeConfig;
use lanyte_common::channels;
use lanyte_mission::{
    MissionControlRequest, MissionCreateBody, MissionLaunchBody, MissionListBody, MissionPhase,
    RecoveryPolicy,
};
use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
use uuid::Uuid;
use zeroize::{Zeroize, Zeroizing};

const SESSION_TOKEN_ENV: &str = "LANYTE_SESSION_TOKEN";
const COMMAND_TIMEOUT: Duration = Duration::from_secs(15);

#[derive(Debug, Parser)]
#[command(name = "lanyte", version, about = "Durable Lanyte mission supervisor")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Run the long-lived local Lanyte kernel.
    Serve,
    /// Create and query durable missions through the running kernel.
    Mission {
        #[command(subcommand)]
        command: MissionCommand,
    },
}

#[derive(Debug, Subcommand)]
pub enum MissionCommand {
    Create(CreateArgs),
    Show(ShowArgs),
    List(ListArgs),
    Launch(LaunchArgs),
    Observe(ObserveArgs),
    Close(CloseArgs),
}

#[derive(Debug, Args)]
pub struct CreateArgs {
    #[arg(long)]
    goal: String,
    #[arg(long = "policy")]
    policy_id: String,
    #[arg(long)]
    deadline: Option<String>,
    #[arg(long, value_enum, default_value_t = CliRecoveryPolicy::AskOperator)]
    recovery_policy: CliRecoveryPolicy,
    #[arg(long)]
    idempotency_key: Option<String>,
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
pub struct ShowArgs {
    mission_id: String,
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
pub struct ListArgs {
    #[arg(long = "phase", value_enum)]
    phases: Vec<CliMissionPhase>,
    #[arg(long, default_value_t = 100)]
    limit: u16,
    #[arg(long)]
    cursor: Option<String>,
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
pub struct LaunchArgs {
    mission_id: String,
    #[arg(long)]
    workspace: String,
    #[arg(long)]
    binary: Option<String>,
    #[arg(long, default_value_t = 0)]
    expected_revision: u64,
    #[arg(long)]
    idempotency_key: Option<String>,
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
pub struct ObserveArgs {
    mission_id: String,
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
pub struct CloseArgs {
    mission_id: String,
    #[arg(long, default_value_t = 1)]
    expected_revision: u64,
    #[arg(long)]
    idempotency_key: Option<String>,
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliRecoveryPolicy {
    ResumeOrRelaunch,
    ResumeOnly,
    AskOperator,
    StandDown,
}

impl From<CliRecoveryPolicy> for RecoveryPolicy {
    fn from(value: CliRecoveryPolicy) -> Self {
        match value {
            CliRecoveryPolicy::ResumeOrRelaunch => Self::ResumeOrRelaunch,
            CliRecoveryPolicy::ResumeOnly => Self::ResumeOnly,
            CliRecoveryPolicy::AskOperator => Self::AskOperator,
            CliRecoveryPolicy::StandDown => Self::StandDown,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliMissionPhase {
    Created,
    Active,
    Waiting,
    RecoveryPending,
    Suspended,
    Completed,
    Cancelled,
    Failed,
    DeadlineExceeded,
    BudgetExhausted,
}

impl From<CliMissionPhase> for MissionPhase {
    fn from(value: CliMissionPhase) -> Self {
        match value {
            CliMissionPhase::Created => Self::Created,
            CliMissionPhase::Active => Self::Active,
            CliMissionPhase::Waiting => Self::Waiting,
            CliMissionPhase::RecoveryPending => Self::RecoveryPending,
            CliMissionPhase::Suspended => Self::Suspended,
            CliMissionPhase::Completed => Self::Completed,
            CliMissionPhase::Cancelled => Self::Cancelled,
            CliMissionPhase::Failed => Self::Failed,
            CliMissionPhase::DeadlineExceeded => Self::DeadlineExceeded,
            CliMissionPhase::BudgetExhausted => Self::BudgetExhausted,
        }
    }
}

#[derive(Debug, Error)]
pub enum ClientError {
    #[error("LANYTE_SESSION_TOKEN is required for mission commands")]
    MissingSessionToken,
    #[error("invalid mission argument: {0}")]
    InvalidArgument(String),
    #[error("failed to connect to the Lanyte kernel: {0}")]
    Connect(String),
    #[error("mission command timed out")]
    Timeout,
    #[error("invalid mission command response: {0}")]
    InvalidResponse(String),
    #[error("mission command denied ({code}): {message}")]
    Remote { code: String, message: String },
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Runtime(#[from] crate::runtime::RuntimeSocketError),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", deny_unknown_fields)]
enum CommandResponse {
    #[serde(rename = "invoke_result")]
    Result {
        request_id: String,
        command: String,
        result: Value,
    },
    #[serde(rename = "invoke_error")]
    Error {
        request_id: String,
        command: String,
        error_code: String,
        message: String,
        #[serde(rename = "retryable")]
        _retryable: bool,
    },
}

pub async fn run_mission(command: MissionCommand, socket: &Path) -> Result<(), ClientError> {
    crate::runtime::verify_client_socket(socket)?;
    let token = std::env::var(SESSION_TOKEN_ENV)
        .ok()
        .filter(|value| !value.is_empty())
        .map(Zeroizing::new)
        .ok_or(ClientError::MissingSessionToken)?;
    let (request, json) = build_request(command)?;
    let request_id = request.request_id().to_string();
    let operation = request.operation().to_owned();
    let args = serde_json::to_value(&request)?;
    let envelope = serde_json::json!({
        "type": "invoke",
        "request_id": request_id,
        "command": operation,
        "args": args,
    });

    let stream = tokio::net::UnixStream::connect(socket)
        .await
        .map_err(|err| ClientError::Connect(err.to_string()))?;
    crate::runtime::verify_connected_server(socket, &stream)?;
    let mut handshake = HandshakeConfig {
        auth_token: Some(token.to_string()),
        ..HandshakeConfig::default()
    };
    let (mut reader, mut writer) = stream.into_split();
    let handshake_result = async_handshake_client_with_config(
        &mut reader,
        &mut writer,
        &[channels::COMMAND],
        &handshake,
    )
    .await;
    if let Some(auth_token) = handshake.auth_token.as_mut() {
        auth_token.zeroize();
    }
    handshake_result.map_err(|err| ClientError::Connect(err.to_string()))?;

    let payload = serde_json::to_vec(&envelope)?;
    let mut encoded = BytesMut::new();
    encode_frame(channels::COMMAND, &payload, &mut encoded)
        .map_err(|err| ClientError::Connect(err.to_string()))?;
    writer
        .write_all(&encoded)
        .await
        .map_err(|err| ClientError::Connect(err.to_string()))?;
    let frame = tokio::time::timeout(COMMAND_TIMEOUT, read_frame(&mut reader))
        .await
        .map_err(|_| ClientError::Timeout)?
        .map_err(|err| ClientError::Connect(err.to_string()))?;
    if frame.channel != channels::COMMAND {
        return Err(ClientError::InvalidResponse(
            "response arrived on the wrong channel".to_owned(),
        ));
    }
    let response: CommandResponse = serde_json::from_slice(frame.payload.as_ref())?;
    match response {
        CommandResponse::Result {
            request_id: response_id,
            command,
            result,
        } if response_id == request_id && command == operation => {
            if json {
                println!("{}", serde_json::to_string(&result)?);
            } else {
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
            Ok(())
        }
        CommandResponse::Error {
            request_id: response_id,
            command,
            error_code,
            message,
            _retryable: _,
        } if response_id == request_id && command == operation => Err(ClientError::Remote {
            code: error_code,
            message,
        }),
        _ => Err(ClientError::InvalidResponse(
            "response correlation does not match the request".to_owned(),
        )),
    }
}

async fn read_frame(
    reader: &mut tokio::net::unix::OwnedReadHalf,
) -> Result<ipcprims::frame::Frame, std::io::Error> {
    let mut buffer = BytesMut::with_capacity(8 * 1024);
    loop {
        if let Some(frame) =
            decode_frame(&mut buffer, DEFAULT_MAX_PAYLOAD).map_err(std::io::Error::other)?
        {
            return Ok(frame);
        }
        let read = reader.read_buf(&mut buffer).await?;
        if read == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "response channel closed",
            ));
        }
    }
}

fn build_request(command: MissionCommand) -> Result<(MissionControlRequest, bool), ClientError> {
    let request_id = Uuid::new_v4();
    match command {
        MissionCommand::Launch(args) => {
            let mission_id = parse_canonical_uuid_v4(&args.mission_id)?;
            let idempotency_key = args
                .idempotency_key
                .unwrap_or_else(|| format!("mission-launch:{request_id}"));
            let request = MissionControlRequest::launch(
                request_id,
                idempotency_key,
                args.expected_revision,
                MissionLaunchBody {
                    mission_id,
                    workspace: args.workspace,
                    binary: args.binary,
                },
            )
            .map_err(ClientError::InvalidArgument)?;
            Ok((request, args.json))
        }
        MissionCommand::Observe(args) => {
            let mission_id = parse_canonical_uuid_v4(&args.mission_id)?;
            let request = MissionControlRequest::observe(request_id, mission_id)
                .map_err(ClientError::InvalidArgument)?;
            Ok((request, args.json))
        }
        MissionCommand::Close(args) => {
            let mission_id = parse_canonical_uuid_v4(&args.mission_id)?;
            let idempotency_key = args
                .idempotency_key
                .unwrap_or_else(|| format!("mission-close:{request_id}"));
            let request = MissionControlRequest::close(
                request_id,
                idempotency_key,
                args.expected_revision,
                mission_id,
            )
            .map_err(ClientError::InvalidArgument)?;
            Ok((request, args.json))
        }
        MissionCommand::Create(args) => {
            let deadline_at = args
                .deadline
                .as_deref()
                .map(DateTime::parse_from_rfc3339)
                .transpose()
                .map_err(|err| ClientError::InvalidArgument(err.to_string()))?
                .map(|timestamp| timestamp.with_timezone(&Utc));
            let idempotency_key = args
                .idempotency_key
                .unwrap_or_else(|| format!("mission-create:{request_id}"));
            let request = MissionControlRequest::create(
                request_id,
                idempotency_key,
                MissionCreateBody {
                    goal: args.goal,
                    policy_id: args.policy_id,
                    deadline_at,
                    recovery_policy: args.recovery_policy.into(),
                },
            )
            .map_err(ClientError::InvalidArgument)?;
            Ok((request, args.json))
        }
        MissionCommand::Show(args) => {
            let mission_id = parse_canonical_uuid_v4(&args.mission_id)?;
            let request = MissionControlRequest::show(request_id, mission_id)
                .map_err(ClientError::InvalidArgument)?;
            Ok((request, args.json))
        }
        MissionCommand::List(args) => {
            let request = MissionControlRequest::list(
                request_id,
                MissionListBody {
                    phases: args.phases.into_iter().map(Into::into).collect(),
                    limit: args.limit,
                    cursor: args.cursor,
                },
            )
            .map_err(ClientError::InvalidArgument)?;
            Ok((request, args.json))
        }
    }
}

fn parse_canonical_uuid_v4(value: &str) -> Result<Uuid, ClientError> {
    let parsed =
        Uuid::parse_str(value).map_err(|err| ClientError::InvalidArgument(err.to_string()))?;
    if parsed.get_version() != Some(uuid::Version::Random) || parsed.to_string() != value {
        return Err(ClientError::InvalidArgument(
            "mission id must be a canonical UUID v4".to_owned(),
        ));
    }
    Ok(parsed)
}
