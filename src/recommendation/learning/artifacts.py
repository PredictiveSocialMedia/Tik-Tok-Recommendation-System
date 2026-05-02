from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return repr(value)


def _type_name(value: Any) -> str:
    return type(value).__name__


def _type_label(expected_types: Sequence[type]) -> str:
    return " or ".join(expected_type.__name__ for expected_type in expected_types)


def _is_expected_type(value: Any, expected_types: Sequence[type]) -> bool:
    for expected_type in expected_types:
        if expected_type is int:
            if type(value) is int:
                return True
            continue
        if expected_type is float:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return True
            continue
        if expected_type is bool:
            if type(value) is bool:
                return True
            continue
        if isinstance(value, expected_type):
            return True
    return False


@dataclass(frozen=True)
class ManifestValidationIssue:
    code: str
    field: str
    message: str
    expected: Optional[Any] = None
    actual: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "code": self.code,
            "field": self.field,
            "message": self.message,
        }
        if self.expected is not None:
            payload["expected"] = _json_safe(self.expected)
        if self.actual is not None:
            payload["actual"] = _json_safe(self.actual)
        return payload


class ArtifactManifestError(ValueError):
    error_code = "invalid_artifact_manifest"
    reason_code = "artifact_manifest_error"

    def __init__(
        self,
        message: str,
        *,
        manifest_path: Optional[Path] = None,
        issues: Optional[Sequence[ManifestValidationIssue]] = None,
        reason_code: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.manifest_path = manifest_path
        self.issues = list(issues or [])
        self.reason_code = reason_code or self.reason_code

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "error": self.error_code,
            "reason": str(self),
            "reason_code": self.reason_code,
        }
        if self.manifest_path is not None:
            payload["manifest_path"] = str(self.manifest_path)
        if self.issues:
            payload["issues"] = [issue.to_dict() for issue in self.issues]
        return payload


class ArtifactManifestNotFoundError(ArtifactManifestError):
    error_code = "artifact_manifest_missing"
    reason_code = "artifact_manifest_missing"


class ArtifactManifestParseError(ArtifactManifestError):
    error_code = "invalid_artifact_manifest"
    reason_code = "artifact_manifest_parse_error"


class ArtifactManifestValidationError(ArtifactManifestError):
    error_code = "invalid_artifact_manifest"
    reason_code = "artifact_manifest_validation_failed"


class ArtifactCompatibilityError(ArtifactManifestError):
    error_code = "incompatible_artifact"
    reason_code = "artifact_compatibility_mismatch"


@dataclass(frozen=True)
class ManifestFieldRule:
    field: str
    expected_types: Tuple[type, ...]
    required: bool = False
    allow_none: bool = False
    non_empty: bool = False
    item_type: Optional[type] = None
    item_non_empty: bool = False
    mapping_key_type: Optional[type] = None
    mapping_value_type: Optional[type] = None
    mapping_value_allow_none: bool = False

    def validate(self, payload: Dict[str, Any]) -> List[ManifestValidationIssue]:
        if self.field not in payload:
            if not self.required:
                return []
            return [
                ManifestValidationIssue(
                    code="manifest_missing_required_field",
                    field=self.field,
                    message=f"Manifest field '{self.field}' is required.",
                    expected=_type_label(self.expected_types),
                    actual="missing",
                )
            ]

        value = payload[self.field]
        if value is None:
            if self.allow_none:
                return []
            return [
                ManifestValidationIssue(
                    code="manifest_null_field",
                    field=self.field,
                    message=f"Manifest field '{self.field}' cannot be null.",
                    expected=_type_label(self.expected_types),
                    actual="null",
                )
            ]

        if not _is_expected_type(value, self.expected_types):
            return [
                ManifestValidationIssue(
                    code="manifest_invalid_field_type",
                    field=self.field,
                    message=(
                        f"Manifest field '{self.field}' must be "
                        f"{_type_label(self.expected_types)}, got {_type_name(value)}."
                    ),
                    expected=_type_label(self.expected_types),
                    actual=_type_name(value),
                )
            ]

        issues: List[ManifestValidationIssue] = []
        if self.non_empty:
            is_empty_string = isinstance(value, str) and not value.strip()
            is_empty_sequence = isinstance(value, (list, tuple, dict)) and not value
            if is_empty_string or is_empty_sequence:
                issues.append(
                    ManifestValidationIssue(
                        code="manifest_empty_field",
                        field=self.field,
                        message=f"Manifest field '{self.field}' must not be empty.",
                        expected="non-empty value",
                        actual=value,
                    )
                )

        if self.item_type is not None and isinstance(value, list):
            for index, item in enumerate(value):
                item_field = f"{self.field}[{index}]"
                if not _is_expected_type(item, (self.item_type,)):
                    issues.append(
                        ManifestValidationIssue(
                            code="manifest_invalid_list_item_type",
                            field=item_field,
                            message=(
                                f"Manifest field '{item_field}' must be "
                                f"{self.item_type.__name__}, got {_type_name(item)}."
                            ),
                            expected=self.item_type.__name__,
                            actual=_type_name(item),
                        )
                    )
                elif self.item_non_empty and isinstance(item, str) and not item.strip():
                    issues.append(
                        ManifestValidationIssue(
                            code="manifest_empty_list_item",
                            field=item_field,
                            message=f"Manifest field '{item_field}' must not be empty.",
                            expected="non-empty value",
                            actual=item,
                        )
                    )

        if isinstance(value, dict):
            if self.mapping_key_type is not None:
                for key in value.keys():
                    if not _is_expected_type(key, (self.mapping_key_type,)):
                        issues.append(
                            ManifestValidationIssue(
                                code="manifest_invalid_mapping_key_type",
                                field=self.field,
                                message=(
                                    f"Manifest field '{self.field}' keys must be "
                                    f"{self.mapping_key_type.__name__}, got {_type_name(key)}."
                                ),
                                expected=self.mapping_key_type.__name__,
                                actual=_type_name(key),
                            )
                        )
            if self.mapping_value_type is not None:
                for key, item in value.items():
                    item_field = f"{self.field}.{key}"
                    if item is None and self.mapping_value_allow_none:
                        continue
                    if not _is_expected_type(item, (self.mapping_value_type,)):
                        issues.append(
                            ManifestValidationIssue(
                                code="manifest_invalid_mapping_value_type",
                                field=item_field,
                                message=(
                                    f"Manifest field '{item_field}' must be "
                                    f"{self.mapping_value_type.__name__}, "
                                    f"got {_type_name(item)}."
                                ),
                                expected=self.mapping_value_type.__name__,
                                actual=_type_name(item),
                            )
                        )
        return issues


@dataclass(frozen=True)
class ArtifactManifestSchema:
    name: str
    fields: Tuple[ManifestFieldRule, ...]

    def validate(self, payload: Any) -> List[ManifestValidationIssue]:
        if not isinstance(payload, dict):
            return [
                ManifestValidationIssue(
                    code="manifest_invalid_root_type",
                    field="$",
                    message="Artifact manifest must be a JSON object.",
                    expected="object",
                    actual=_type_name(payload),
                )
            ]

        issues: List[ManifestValidationIssue] = []
        for rule in self.fields:
            issues.extend(rule.validate(payload))
        return issues


RECOMMENDER_BUNDLE_MANIFEST_SCHEMA = ArtifactManifestSchema(
    name="recommender_bundle_manifest.v1",
    fields=(
        ManifestFieldRule("component", (str,), required=True, non_empty=True),
        ManifestFieldRule("contract_version", (str,), required=True, non_empty=True),
        ManifestFieldRule("datamart_version", (str,), required=True, non_empty=True),
        ManifestFieldRule("feature_schema_hash", (str,), required=True, non_empty=True),
        ManifestFieldRule(
            "objectives",
            (list,),
            required=True,
            non_empty=True,
            item_type=str,
            item_non_empty=True,
        ),
        ManifestFieldRule("ranker_family_schema_hash", (str,), allow_none=True),
        ManifestFieldRule("ranker_family_version", (str,), allow_none=True),
        ManifestFieldRule("feature_manifest_id", (str,), allow_none=True),
        ManifestFieldRule("feature_manifest_path", (str,), allow_none=True),
        ManifestFieldRule("feature_snapshot_manifest_id", (str,), allow_none=True),
        ManifestFieldRule("feature_snapshot_manifest_path", (str,), allow_none=True),
        ManifestFieldRule("fabric_version", (str,), allow_none=True),
        ManifestFieldRule("fabric_registry_signature", (str,), allow_none=True),
        ManifestFieldRule(
            "fabric_schema_hashes",
            (dict,),
            allow_none=True,
            mapping_key_type=str,
            mapping_value_type=str,
        ),
        ManifestFieldRule("comment_intelligence_version", (str,), allow_none=True),
        ManifestFieldRule("comment_feature_manifest_id", (str,), allow_none=True),
        ManifestFieldRule("comment_feature_manifest_path", (str,), allow_none=True),
        ManifestFieldRule("comment_priors_manifest_id", (str,), allow_none=True),
        ManifestFieldRule("comment_priors_manifest_path", (str,), allow_none=True),
        ManifestFieldRule("graph", (dict,), allow_none=True),
        ManifestFieldRule("graph_enabled", (bool,), allow_none=True),
        ManifestFieldRule("graph_embedding_dim", (int,), allow_none=True),
        ManifestFieldRule("graph_walk_params", (dict,), allow_none=True),
        ManifestFieldRule("graph_weighting_params", (dict,), allow_none=True),
        ManifestFieldRule("trajectory", (dict,), allow_none=True),
        ManifestFieldRule("trajectory_enabled", (bool,), allow_none=True),
        ManifestFieldRule("trajectory_embedding_dim", (int,), allow_none=True),
        ManifestFieldRule("trajectory_branch_weight", (int, float), allow_none=True),
        ManifestFieldRule("trajectory_encoder_mode", (str,), allow_none=True),
        ManifestFieldRule("trajectory_manifest_path", (str,), allow_none=True),
        ManifestFieldRule("trajectory_version", (str,), allow_none=True),
        ManifestFieldRule("retrieve_k", (int,), allow_none=True),
        ManifestFieldRule("max_age_days", (int,), allow_none=True),
        ManifestFieldRule("dense_model_name", (str,), allow_none=True),
        ManifestFieldRule("random_seed", (int,), allow_none=True),
        ManifestFieldRule("pair_target_source", (str,), allow_none=True),
        ManifestFieldRule("policy_reranker", (dict,), allow_none=True),
        ManifestFieldRule("objective_diagnostics", (dict,), allow_none=True),
        ManifestFieldRule("objective_ablation_reports", (dict,), allow_none=True),
        ManifestFieldRule("drift_signals", (dict,), allow_none=True),
        ManifestFieldRule("retriever", (dict,), allow_none=True),
        ManifestFieldRule("rows_total", (int,), allow_none=True),
        ManifestFieldRule("pair_rows_total", (int,), allow_none=True),
        ManifestFieldRule("train_rows", (int,), allow_none=True),
        ManifestFieldRule("validation_rows", (int,), allow_none=True),
        ManifestFieldRule("test_rows", (int,), allow_none=True),
        ManifestFieldRule("created_at", (str,), allow_none=True),
    ),
)


def _format_issue_summary(
    issues: Sequence[ManifestValidationIssue],
    limit: int = 3,
) -> str:
    if not issues:
        return "unknown manifest validation error"
    selected = [issue.message for issue in issues[:limit]]
    if len(issues) > limit:
        selected.append(f"{len(issues) - limit} more issue(s)")
    return " | ".join(selected)


def validate_manifest(
    payload: Dict[str, Any],
    schema: ArtifactManifestSchema,
    *,
    manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    issues = schema.validate(payload)
    if issues:
        raise ArtifactManifestValidationError(
            f"Artifact manifest failed {schema.name} validation: "
            f"{_format_issue_summary(issues)}",
            manifest_path=manifest_path,
            issues=issues,
        )
    return payload


@dataclass
class ArtifactRegistry:
    root_dir: Path

    def _manifest_path(self, bundle_dir: Path) -> Path:
        return bundle_dir / "manifest.json"

    def manifest_path(self, bundle_dir: Path) -> Path:
        return self._manifest_path(bundle_dir)

    def create_bundle_dir(self, run_name: Optional[str] = None) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        folder = f"{stamp}-{(run_name or 'recommender').strip().replace(' ', '-')}"
        bundle_dir = self.root_dir / folder
        bundle_dir.mkdir(parents=True, exist_ok=True)
        return bundle_dir

    def write_manifest(
        self,
        bundle_dir: Path,
        payload: Dict[str, Any],
    ) -> None:
        manifest_path = self._manifest_path(bundle_dir)
        enriched = {
            **payload,
            "created_at": _utc_now_iso(),
        }
        manifest_path.write_text(
            json.dumps(enriched, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )

    def load_manifest(
        self,
        bundle_dir: Path,
        schema: Optional[ArtifactManifestSchema] = None,
    ) -> Dict[str, Any]:
        manifest_path = self._manifest_path(bundle_dir)
        try:
            raw = manifest_path.read_text(encoding="utf-8")
        except FileNotFoundError as error:
            issue = ManifestValidationIssue(
                code="manifest_file_missing",
                field="$",
                message=f"Artifact manifest does not exist at {manifest_path}.",
                expected="manifest.json",
                actual="missing",
            )
            raise ArtifactManifestNotFoundError(
                f"Artifact manifest does not exist at {manifest_path}.",
                manifest_path=manifest_path,
                issues=[issue],
            ) from error

        try:
            manifest = json.loads(raw)
        except json.JSONDecodeError as error:
            issue = ManifestValidationIssue(
                code="manifest_invalid_json",
                field="$",
                message=(
                    "Manifest JSON parse error at "
                    f"line {error.lineno}, column {error.colno}: {error.msg}."
                ),
                expected="valid JSON object",
                actual=error.msg,
            )
            raise ArtifactManifestParseError(
                f"Artifact manifest at {manifest_path} is not valid JSON: {error.msg}.",
                manifest_path=manifest_path,
                issues=[issue],
            ) from error

        if not isinstance(manifest, dict):
            issue = ManifestValidationIssue(
                code="manifest_invalid_root_type",
                field="$",
                message="Artifact manifest must be a JSON object.",
                expected="object",
                actual=_type_name(manifest),
            )
            raise ArtifactManifestValidationError(
                "Artifact manifest must be a JSON object.",
                manifest_path=manifest_path,
                issues=[issue],
            )

        if schema is not None:
            validate_manifest(manifest, schema, manifest_path=manifest_path)
        return manifest

    def load_recommender_bundle_manifest(self, bundle_dir: Path) -> Dict[str, Any]:
        return self.load_manifest(bundle_dir, schema=RECOMMENDER_BUNDLE_MANIFEST_SCHEMA)

    def feature_schema_hash(self, feature_names: list[str]) -> str:
        canonical = json.dumps(
            sorted(feature_names), separators=(",", ":"), ensure_ascii=False
        )
        return _sha256_text(canonical)

    def assert_compatible(
        self,
        bundle_dir: Path,
        expected: Dict[str, Any],
        manifest: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        manifest = manifest or self.load_manifest(bundle_dir)
        issues: List[ManifestValidationIssue] = []
        for key, expected_value in expected.items():
            actual_value = manifest.get(key)
            if actual_value != expected_value:
                issues.append(
                    ManifestValidationIssue(
                        code="artifact_compatibility_mismatch",
                        field=key,
                        message=(
                            f"{key}: expected={expected_value!r}, "
                            f"actual={actual_value!r}"
                        ),
                        expected=expected_value,
                        actual=actual_value,
                    )
                )
        if issues:
            raise ArtifactCompatibilityError(
                "Artifact compatibility check failed: "
                + " | ".join(issue.message for issue in issues),
                manifest_path=self._manifest_path(bundle_dir),
                issues=issues,
            )
        return manifest
