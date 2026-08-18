use thiserror::Error;
use uuid::Uuid;

use crate::{AttemptRecord, AttemptState, InvariantError, MissionPhase, MissionRecord, Validate};

impl MissionPhase {
    #[must_use]
    pub const fn can_transition_to(self, next: Self) -> bool {
        mission_transition_allowed(self, next)
    }
}

impl AttemptState {
    #[must_use]
    pub const fn can_transition_to(self, next: Self) -> bool {
        attempt_transition_allowed(self, next)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MissionTransition {
    pub expected_revision: u64,
    pub from: MissionPhase,
    pub to: MissionPhase,
}

impl MissionTransition {
    pub fn check(
        self,
        before: &MissionRecord,
        after: &MissionRecord,
    ) -> Result<(), TransitionError> {
        before.validate().map_err(TransitionError::InvalidState)?;
        self.check_edge(before)?;
        if after.mission_id != before.mission_id
            || after.evidence_chain_id != before.evidence_chain_id
            || after.initiator != before.initiator
            || after.supervisor != before.supervisor
            || after.operating_role != before.operating_role
        {
            return Err(TransitionError::IdentityChanged);
        }
        if after.updated_at < before.updated_at {
            return Err(TransitionError::TimeMovedBackward);
        }
        let result_revision = self
            .expected_revision
            .checked_add(1)
            .ok_or(TransitionError::RevisionOverflow)?;
        if after.revision != result_revision {
            return Err(TransitionError::ResultRevision {
                expected: result_revision,
                actual: after.revision,
            });
        }
        if after.phase != self.to {
            return Err(TransitionError::MissionPhaseMismatch {
                expected: self.to,
                actual: after.phase,
            });
        }
        after.validate().map_err(TransitionError::InvalidState)
    }

    fn check_edge(self, mission: &MissionRecord) -> Result<(), TransitionError> {
        if mission.revision != self.expected_revision {
            return Err(TransitionError::StaleRevision {
                expected: self.expected_revision,
                actual: mission.revision,
            });
        }
        if mission.phase != self.from {
            return Err(TransitionError::MissionPhaseMismatch {
                expected: self.from,
                actual: mission.phase,
            });
        }
        if self.from.is_terminal() {
            return Err(TransitionError::TerminalMission(self.from));
        }
        if !self.from.can_transition_to(self.to) {
            return Err(TransitionError::IllegalMissionTransition {
                from: self.from,
                to: self.to,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttemptTransition {
    pub attempt_id: Uuid,
    pub generation: u64,
    pub from: AttemptState,
    pub to: AttemptState,
}

impl AttemptTransition {
    pub fn check(
        self,
        before: &AttemptRecord,
        after: &AttemptRecord,
    ) -> Result<(), TransitionError> {
        before.validate().map_err(TransitionError::InvalidState)?;
        self.check_edge(before)?;
        if after.attempt_id != before.attempt_id
            || after.ordinal != before.ordinal
            || after.generation != before.generation
            || after.fencing_token_sha256 != before.fencing_token_sha256
            || after.recovery_relation != before.recovery_relation
            || after.predecessor_attempt_id != before.predecessor_attempt_id
            || (before.driver_id.is_some() && after.driver_id != before.driver_id)
            || (before.harness_session_id.is_some()
                && after.harness_session_id != before.harness_session_id)
            || (before.started_at.is_some() && after.started_at != before.started_at)
        {
            return Err(TransitionError::AttemptIdentityChanged);
        }
        if after.state != self.to {
            return Err(TransitionError::AttemptStateMismatch {
                expected: self.to,
                actual: after.state,
            });
        }
        after.validate().map_err(TransitionError::InvalidState)
    }

    fn check_edge(self, attempt: &AttemptRecord) -> Result<(), TransitionError> {
        if attempt.attempt_id != self.attempt_id {
            return Err(TransitionError::AttemptIdMismatch {
                expected: self.attempt_id,
                actual: attempt.attempt_id,
            });
        }
        if attempt.generation != self.generation {
            return Err(TransitionError::StaleGeneration {
                expected: attempt.generation,
                actual: self.generation,
            });
        }
        if attempt.state != self.from {
            return Err(TransitionError::AttemptStateMismatch {
                expected: self.from,
                actual: attempt.state,
            });
        }
        if attempt.state.is_terminal() {
            return Err(TransitionError::TerminalAttempt(attempt.state));
        }
        if !self.from.can_transition_to(self.to) {
            return Err(TransitionError::IllegalAttemptTransition {
                from: self.from,
                to: self.to,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TransitionError {
    #[error("transition state violates an invariant: {0}")]
    InvalidState(InvariantError),
    #[error("mission identity changed across transition")]
    IdentityChanged,
    #[error("attempt identity or fence changed across transition")]
    AttemptIdentityChanged,
    #[error("result revision mismatch: expected {expected}, actual {actual}")]
    ResultRevision { expected: u64, actual: u64 },
    #[error("mission revision overflow")]
    RevisionOverflow,
    #[error("transition timestamp moved backward")]
    TimeMovedBackward,
    #[error("stale mission revision: expected {expected}, actual {actual}")]
    StaleRevision { expected: u64, actual: u64 },
    #[error("mission phase mismatch: expected {expected:?}, actual {actual:?}")]
    MissionPhaseMismatch {
        expected: MissionPhase,
        actual: MissionPhase,
    },
    #[error("terminal mission cannot transition from {0:?}")]
    TerminalMission(MissionPhase),
    #[error("illegal mission transition from {from:?} to {to:?}")]
    IllegalMissionTransition {
        from: MissionPhase,
        to: MissionPhase,
    },
    #[error("attempt id mismatch: expected {expected}, actual {actual}")]
    AttemptIdMismatch { expected: Uuid, actual: Uuid },
    #[error("stale attempt generation: expected {expected}, actual {actual}")]
    StaleGeneration { expected: u64, actual: u64 },
    #[error("attempt state mismatch: expected {expected:?}, actual {actual:?}")]
    AttemptStateMismatch {
        expected: AttemptState,
        actual: AttemptState,
    },
    #[error("terminal attempt cannot transition from {0:?}")]
    TerminalAttempt(AttemptState),
    #[error("illegal attempt transition from {from:?} to {to:?}")]
    IllegalAttemptTransition {
        from: AttemptState,
        to: AttemptState,
    },
}

const fn mission_transition_allowed(from: MissionPhase, to: MissionPhase) -> bool {
    use MissionPhase::{
        Active, BudgetExhausted, Cancelled, Completed, Created, DeadlineExceeded, Failed,
        RecoveryPending, Suspended, Waiting,
    };
    matches!(
        (from, to),
        (Created, Active | Suspended | Cancelled | Failed)
            | (
                Active,
                Waiting
                    | RecoveryPending
                    | Suspended
                    | Completed
                    | Cancelled
                    | Failed
                    | DeadlineExceeded
                    | BudgetExhausted
            )
            | (
                Waiting,
                Active
                    | RecoveryPending
                    | Suspended
                    | Cancelled
                    | Failed
                    | DeadlineExceeded
                    | BudgetExhausted
            )
            | (
                RecoveryPending,
                Active | Suspended | Cancelled | Failed | DeadlineExceeded | BudgetExhausted
            )
            | (
                Suspended,
                Active | Cancelled | Failed | DeadlineExceeded | BudgetExhausted
            )
    )
}

const fn attempt_transition_allowed(from: AttemptState, to: AttemptState) -> bool {
    use AttemptState::{
        Cancelled, Cancelling, Completed, Crashed, Failed, Lost, Replaced, Running, Starting,
        TimedOut, Unresponsive, Waiting,
    };
    matches!(
        (from, to),
        (
            Starting,
            Running | Cancelling | Failed | Crashed | TimedOut | Lost
        ) | (
            Running,
            Waiting | Unresponsive | Cancelling | Completed | Failed | Crashed | TimedOut | Lost
        ) | (
            Waiting,
            Running | Unresponsive | Cancelling | Completed | Failed | Crashed | TimedOut | Lost
        ) | (
            Unresponsive,
            Running | Cancelling | Replaced | Failed | Crashed | TimedOut | Lost
        ) | (Cancelling, Cancelled | Failed | Crashed | TimedOut | Lost)
    )
}

#[cfg(test)]
mod tests {
    use chrono::{TimeZone, Utc};

    use crate::invariant::fixtures::{created_mission, uuid};
    use crate::{AttemptTerminalReason, RecoveryRelation};

    use super::*;

    fn running_attempt() -> AttemptRecord {
        AttemptRecord {
            attempt_id: uuid(8),
            ordinal: 1,
            generation: 1,
            fencing_token_sha256: "a".repeat(64),
            recovery_relation: RecoveryRelation::Initial,
            predecessor_attempt_id: None,
            state: AttemptState::Running,
            driver_id: Some("driver.local".to_owned()),
            harness_session_id: Some("session-1".to_owned()),
            started_at: Some(Utc.with_ymd_and_hms(2026, 8, 17, 14, 1, 0).unwrap()),
            ended_at: None,
            terminal_reason: None,
            evidence_ref: None,
        }
    }

    fn active_mission() -> (MissionRecord, MissionRecord) {
        let before = created_mission();
        let mut after = before.clone();
        let attempt = running_attempt();
        after.revision = 1;
        after.phase = MissionPhase::Active;
        after.authorizer = Some(before.initiator.clone());
        after.authorization_ref = Some("authorizations/1".to_owned());
        after.current_attempt_id = Some(attempt.attempt_id);
        after.attempts.push(attempt);
        (before, after)
    }

    #[test]
    fn mission_transition_matrix_accepts_created_to_active() {
        let (before, after) = active_mission();
        MissionTransition {
            expected_revision: 0,
            from: MissionPhase::Created,
            to: MissionPhase::Active,
        }
        .check(&before, &after)
        .unwrap();
    }

    #[test]
    fn mission_transition_rejects_invalid_result_state() {
        let before = created_mission();
        let mut after = before.clone();
        after.revision = 1;
        after.phase = MissionPhase::Active;
        let error = MissionTransition {
            expected_revision: 0,
            from: MissionPhase::Created,
            to: MissionPhase::Active,
        }
        .check(&before, &after)
        .unwrap_err();
        assert!(matches!(error, TransitionError::InvalidState(_)));
    }

    #[test]
    fn mission_transition_matrix_rejects_created_to_completed() {
        let before = created_mission();
        let error = MissionTransition {
            expected_revision: 0,
            from: MissionPhase::Created,
            to: MissionPhase::Completed,
        }
        .check(&before, &before)
        .unwrap_err();
        assert!(matches!(
            error,
            TransitionError::IllegalMissionTransition { .. }
        ));
    }

    #[test]
    fn stale_mission_revision_is_rejected() {
        let (before, after) = active_mission();
        let error = MissionTransition {
            expected_revision: 1,
            from: MissionPhase::Created,
            to: MissionPhase::Active,
        }
        .check(&before, &after)
        .unwrap_err();
        assert!(matches!(error, TransitionError::StaleRevision { .. }));
    }

    #[test]
    fn running_attempt_may_wait() {
        let attempt = running_attempt();
        let mut result = attempt.clone();
        result.state = AttemptState::Waiting;
        AttemptTransition {
            attempt_id: attempt.attempt_id,
            generation: 1,
            from: AttemptState::Running,
            to: AttemptState::Waiting,
        }
        .check(&attempt, &result)
        .unwrap();
    }

    #[test]
    fn stale_attempt_generation_is_rejected() {
        let attempt = running_attempt();
        let mut result = attempt.clone();
        result.state = AttemptState::Waiting;
        let error = AttemptTransition {
            attempt_id: attempt.attempt_id,
            generation: 0,
            from: AttemptState::Running,
            to: AttemptState::Waiting,
        }
        .check(&attempt, &result)
        .unwrap_err();
        assert!(matches!(error, TransitionError::StaleGeneration { .. }));
    }

    #[test]
    fn terminal_attempt_does_not_transition() {
        let mut attempt = running_attempt();
        attempt.state = AttemptState::Completed;
        attempt.ended_at = Some(Utc.with_ymd_and_hms(2026, 8, 17, 14, 2, 0).unwrap());
        attempt.terminal_reason = Some(AttemptTerminalReason::HarnessCompleted);
        let mut result = attempt.clone();
        result.state = AttemptState::Running;
        result.ended_at = None;
        result.terminal_reason = None;
        let error = AttemptTransition {
            attempt_id: attempt.attempt_id,
            generation: 1,
            from: AttemptState::Completed,
            to: AttemptState::Running,
        }
        .check(&attempt, &result)
        .unwrap_err();
        assert!(matches!(error, TransitionError::TerminalAttempt(_)));
    }

    #[test]
    fn mission_transition_matrix_is_complete() {
        use MissionPhase::*;

        const ALL: [MissionPhase; 10] = [
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
        ];
        let cases: &[(MissionPhase, &[MissionPhase])] = &[
            (Created, &[Active, Suspended, Cancelled, Failed]),
            (
                Active,
                &[
                    Waiting,
                    RecoveryPending,
                    Suspended,
                    Completed,
                    Cancelled,
                    Failed,
                    DeadlineExceeded,
                    BudgetExhausted,
                ],
            ),
            (
                Waiting,
                &[
                    Active,
                    RecoveryPending,
                    Suspended,
                    Cancelled,
                    Failed,
                    DeadlineExceeded,
                    BudgetExhausted,
                ],
            ),
            (
                RecoveryPending,
                &[
                    Active,
                    Suspended,
                    Cancelled,
                    Failed,
                    DeadlineExceeded,
                    BudgetExhausted,
                ],
            ),
            (
                Suspended,
                &[Active, Cancelled, Failed, DeadlineExceeded, BudgetExhausted],
            ),
            (Completed, &[]),
            (Cancelled, &[]),
            (Failed, &[]),
            (DeadlineExceeded, &[]),
            (BudgetExhausted, &[]),
        ];
        for (from, allowed) in cases {
            for to in ALL {
                assert_eq!(
                    from.can_transition_to(to),
                    allowed.contains(&to),
                    "{from:?} -> {to:?}"
                );
            }
        }
    }

    #[test]
    fn attempt_transition_matrix_is_complete() {
        use AttemptState::*;

        const ALL: [AttemptState; 12] = [
            Starting,
            Running,
            Waiting,
            Unresponsive,
            Cancelling,
            Completed,
            Cancelled,
            Replaced,
            Failed,
            Crashed,
            TimedOut,
            Lost,
        ];
        let cases: &[(AttemptState, &[AttemptState])] = &[
            (
                Starting,
                &[Running, Cancelling, Failed, Crashed, TimedOut, Lost],
            ),
            (
                Running,
                &[
                    Waiting,
                    Unresponsive,
                    Cancelling,
                    Completed,
                    Failed,
                    Crashed,
                    TimedOut,
                    Lost,
                ],
            ),
            (
                Waiting,
                &[
                    Running,
                    Unresponsive,
                    Cancelling,
                    Completed,
                    Failed,
                    Crashed,
                    TimedOut,
                    Lost,
                ],
            ),
            (
                Unresponsive,
                &[
                    Running, Cancelling, Replaced, Failed, Crashed, TimedOut, Lost,
                ],
            ),
            (Cancelling, &[Cancelled, Failed, Crashed, TimedOut, Lost]),
            (Completed, &[]),
            (Cancelled, &[]),
            (Replaced, &[]),
            (Failed, &[]),
            (Crashed, &[]),
            (TimedOut, &[]),
            (Lost, &[]),
        ];
        for (from, allowed) in cases {
            for to in ALL {
                assert_eq!(
                    from.can_transition_to(to),
                    allowed.contains(&to),
                    "{from:?} -> {to:?}"
                );
            }
        }
    }
}
