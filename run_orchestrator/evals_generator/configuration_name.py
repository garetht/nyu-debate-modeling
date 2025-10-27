import dataclasses

from run_orchestrator.evals_generator.config_spec import ConfigurationType, TASK_TYPE_PARAMS
from run_orchestrator.evals_generator.model_definitions import DebaterModelConfiguration, JudgeModelConfiguration, \
    ALL_VALID_DEBATERS, ALL_VALID_JUDGES


@dataclasses.dataclass(frozen=True, init=False)
class ConfigurationName:
    config_type: ConfigurationType
    debater_key: str
    judge_key: str
    task_type_name: str

    _SECTION_SEPARATOR = "--"
    _ROLE_SEPARATOR = "_"
    _DEBATER_ALIAS_SUFFIX = "-debater"
    _JUDGE_ALIAS_SUFFIX = "-judge"

    def serialize(self) -> str:
        debater_config = self.debater_config
        judge_config = self.judge_config

        debater_segment = self._serialize_model_segment(
            self._build_debater_alias(self.debater_key),
            debater_config.training_round.display_name
        )
        judge_segment = self._serialize_model_segment(
            self._build_judge_alias(self.judge_key),
            judge_config.training_round.display_name
        )

        return self._SECTION_SEPARATOR.join(
            [
                self.config_type.value,
                debater_segment,
                judge_segment,
                self._serialize_task_type(),
            ]
        )

    @classmethod
    def deserialize(cls, raw_name: str) -> "ConfigurationName":
        parts = raw_name.split(cls._SECTION_SEPARATOR)
        if len(parts) != 4:
            raise ValueError(f"Configuration name '{raw_name}' is malformed.")

        config_type = cls._parse_config_type(parts[0])
        debater_key = cls._parse_debater_segment(parts[1])
        judge_key = cls._parse_judge_segment(parts[2])
        task_type_name = cls._parse_task_type(parts[3])

        return cls._create(
            config_type=config_type,
            debater_key=debater_key,
            judge_key=judge_key,
            task_type_name=task_type_name,
        )

    @classmethod
    def serialize_from_inputs(
            cls,
            config_type: ConfigurationType,
            debater: DebaterModelConfiguration,
            judge: JudgeModelConfiguration,
            task_type_name: str,
    ) -> str:
        if not isinstance(config_type, ConfigurationType):
            raise TypeError("config_type must be provided as a ConfigurationType.")
        debater_alias, debater_training = cls._normalize_debater_input(debater)
        judge_alias, judge_training = cls._normalize_judge_input(judge)
        normalized_task_type = cls._normalize_task_type(task_type_name)

        return cls._SECTION_SEPARATOR.join(
            [
                config_type.value,
                cls._serialize_model_segment(debater_alias, debater_training),
                cls._serialize_model_segment(judge_alias, judge_training),
                normalized_task_type,
            ]
        )

    @classmethod
    def _create(
            cls,
            *,
            config_type: ConfigurationType,
            debater_key: str,
            judge_key: str,
            task_type_name: str,
    ) -> "ConfigurationName":
        instance = object.__new__(cls)
        object.__setattr__(instance, "config_type", config_type)
        object.__setattr__(instance, "debater_key", debater_key)
        object.__setattr__(instance, "judge_key", judge_key)
        object.__setattr__(instance, "task_type_name", task_type_name)
        return instance

    @property
    def debater_config(self) -> DebaterModelConfiguration:
        try:
            return ALL_VALID_DEBATERS[self.debater_key]
        except KeyError as exc:
            raise ValueError(f"Unknown debater '{self.debater_key}'.") from exc

    @property
    def judge_config(self) -> JudgeModelConfiguration:
        try:
            return ALL_VALID_JUDGES[self.judge_key]
        except KeyError as exc:
            raise ValueError(f"Unknown judge '{self.judge_key}'.") from exc

    def _serialize_task_type(self) -> str:
        return self._normalize_task_type(self.task_type_name)

    @staticmethod
    def _normalize_config_type(config_type: ConfigurationType | str) -> ConfigurationType:
        if isinstance(config_type, ConfigurationType):
            return config_type
        try:
            return ConfigurationType(config_type)
        except ValueError as exc:
            raise ValueError(f"Unknown configuration type '{config_type}'.") from exc

    @classmethod
    def _parse_config_type(cls, value: str) -> ConfigurationType:
        return cls._normalize_config_type(value)

    @classmethod
    def _parse_debater_segment(cls, segment: str) -> str:
        alias, training_display = cls._split_model_segment(segment)
        debater_key = cls._resolve_debater_alias(alias)
        expected_training = ALL_VALID_DEBATERS[debater_key].training_round.display_name
        if training_display != expected_training:
            raise ValueError(
                f"Training round '{training_display}' does not match debater '{debater_key}'."
            )
        return debater_key

    @classmethod
    def _parse_judge_segment(cls, segment: str) -> str:
        alias, training_display = cls._split_model_segment(segment)
        judge_key = cls._resolve_judge_alias(alias)
        expected_training = ALL_VALID_JUDGES[judge_key].training_round.display_name
        if training_display != expected_training:
            raise ValueError(
                f"Training round '{training_display}' does not match judge '{judge_key}'."
            )
        return judge_key

    @classmethod
    def _parse_task_type(cls, value: str) -> str:
        return cls._normalize_task_type(value)

    @classmethod
    def _normalize_task_type(cls, task_type_name: str) -> str:
        if task_type_name not in TASK_TYPE_PARAMS:
            raise ValueError(f"Unknown task type '{task_type_name}'.")
        return task_type_name

    @classmethod
    def _split_model_segment(cls, segment: str) -> tuple[str, str]:
        if cls._ROLE_SEPARATOR not in segment:
            raise ValueError(f"Model segment '{segment}' is malformed.")
        alias, training_display = segment.rsplit(cls._ROLE_SEPARATOR, 1)
        if not alias or not training_display:
            raise ValueError(f"Model segment '{segment}' is malformed.")
        return alias, training_display

    @classmethod
    def _resolve_debater_alias(cls, alias: str) -> str:
        mapping = {cls._build_debater_alias(name): name for name in ALL_VALID_DEBATERS}
        try:
            return mapping[alias]
        except KeyError as exc:
            raise ValueError(f"Unknown debater alias '{alias}'.") from exc

    @classmethod
    def _resolve_judge_alias(cls, alias: str) -> str:
        mapping = {cls._build_judge_alias(name): name for name in ALL_VALID_JUDGES}
        try:
            return mapping[alias]
        except KeyError as exc:
            raise ValueError(f"Unknown judge alias '{alias}'.") from exc

    @classmethod
    def _serialize_model_segment(cls, alias: str, training_display: str) -> str:
        return f"{alias}{cls._ROLE_SEPARATOR}{training_display}"

    @classmethod
    def _build_debater_alias(cls, name: str) -> str:
        return f"{name}{cls._DEBATER_ALIAS_SUFFIX}"

    @classmethod
    def _build_judge_alias(cls, name: str) -> str:
        return f"{name}{cls._JUDGE_ALIAS_SUFFIX}"

    @classmethod
    def _normalize_debater_input(
            cls, debater: DebaterModelConfiguration
    ) -> tuple[str, str]:
        if not isinstance(debater, DebaterModelConfiguration):
            raise TypeError("Debater must be provided as a DebaterModelConfiguration.")
        alias = debater.settings.alias
        if not alias:
            raise ValueError("Debater alias must be set when passing a DebaterModelConfiguration.")
        return alias, debater.training_round.display_name

    @classmethod
    def _normalize_judge_input(
            cls, judge: JudgeModelConfiguration
    ) -> tuple[str, str]:
        if not isinstance(judge, JudgeModelConfiguration):
            raise TypeError("Judge must be provided as a JudgeModelConfiguration.")
        alias = judge.settings.alias
        if not alias:
            raise ValueError("Judge alias must be set when passing a JudgeModelConfiguration.")
        return alias, judge.training_round.display_name
