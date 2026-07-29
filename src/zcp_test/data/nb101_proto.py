from __future__ import annotations

import struct
from dataclasses import dataclass


class ProtobufDecodeError(ValueError):
    """Raised when a NAS-Bench-101 ModelMetrics message is malformed."""


@dataclass(frozen=True)
class EvaluationData:
    current_epoch: float | None = None
    training_time: float | None = None
    train_accuracy: float | None = None
    validation_accuracy: float | None = None
    test_accuracy: float | None = None
    checkpoint_path: str | None = None


@dataclass(frozen=True)
class ModelMetrics:
    evaluation_data: tuple[EvaluationData, ...] = ()
    trainable_parameters: int | None = None
    total_time: float | None = None


def _read_varint(data: bytes, position: int) -> tuple[int, int]:
    value = 0
    for shift in range(0, 70, 7):
        if position >= len(data):
            raise ProtobufDecodeError("Truncated protobuf varint")
        byte = data[position]
        position += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            if shift == 63 and byte > 1:
                raise ProtobufDecodeError("Protobuf varint exceeds 64 bits")
            return value, position
    raise ProtobufDecodeError("Protobuf varint exceeds 10 bytes")


def _read_length_delimited(data: bytes, position: int) -> tuple[bytes, int]:
    length, position = _read_varint(data, position)
    end = position + length
    if end > len(data):
        raise ProtobufDecodeError("Truncated length-delimited protobuf field")
    return data[position:end], end


def _skip_field(data: bytes, position: int, wire_type: int) -> int:
    if wire_type == 0:
        return _read_varint(data, position)[1]
    if wire_type == 1:
        end = position + 8
    elif wire_type == 2:
        return _read_length_delimited(data, position)[1]
    elif wire_type == 5:
        end = position + 4
    else:
        raise ProtobufDecodeError(f"Unsupported protobuf wire type: {wire_type}")
    if end > len(data):
        raise ProtobufDecodeError("Truncated fixed-width protobuf field")
    return end


def _parse_evaluation_data(data: bytes) -> EvaluationData:
    values: dict[str, float | str | None] = {
        "current_epoch": None,
        "training_time": None,
        "train_accuracy": None,
        "validation_accuracy": None,
        "test_accuracy": None,
        "checkpoint_path": None,
    }
    double_fields = {
        1: "current_epoch",
        2: "training_time",
        3: "train_accuracy",
        4: "validation_accuracy",
        5: "test_accuracy",
    }
    position = 0
    while position < len(data):
        tag, position = _read_varint(data, position)
        field_number, wire_type = tag >> 3, tag & 7
        if field_number == 0:
            raise ProtobufDecodeError("Protobuf field number cannot be zero")
        if field_number in double_fields:
            if wire_type != 1:
                raise ProtobufDecodeError(
                    f"EvaluationData field {field_number} has wrong wire type"
                )
            end = position + 8
            if end > len(data):
                raise ProtobufDecodeError("Truncated EvaluationData double")
            values[double_fields[field_number]] = struct.unpack_from("<d", data, position)[0]
            position = end
        elif field_number == 6:
            if wire_type != 2:
                raise ProtobufDecodeError("EvaluationData checkpoint_path has wrong wire type")
            raw_value, position = _read_length_delimited(data, position)
            try:
                values["checkpoint_path"] = raw_value.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ProtobufDecodeError("checkpoint_path is not valid UTF-8") from error
        else:
            position = _skip_field(data, position, wire_type)
    return EvaluationData(**values)


def parse_model_metrics(data: bytes) -> ModelMetrics:
    """Parse the official NAS-Bench-101 ``ModelMetrics`` protobuf message."""
    if not isinstance(data, bytes):
        raise TypeError("ModelMetrics payload must be bytes")
    evaluations: list[EvaluationData] = []
    trainable_parameters: int | None = None
    total_time: float | None = None
    position = 0
    while position < len(data):
        tag, position = _read_varint(data, position)
        field_number, wire_type = tag >> 3, tag & 7
        if field_number == 0:
            raise ProtobufDecodeError("Protobuf field number cannot be zero")
        if field_number == 1:
            if wire_type != 2:
                raise ProtobufDecodeError("ModelMetrics evaluation_data has wrong wire type")
            raw_evaluation, position = _read_length_delimited(data, position)
            evaluations.append(_parse_evaluation_data(raw_evaluation))
        elif field_number == 2:
            if wire_type != 0:
                raise ProtobufDecodeError("ModelMetrics trainable_parameters has wrong wire type")
            raw_value, position = _read_varint(data, position)
            int32_value = raw_value & 0xFFFFFFFF
            trainable_parameters = int32_value if int32_value < 1 << 31 else int32_value - (1 << 32)
        elif field_number == 3:
            if wire_type != 1:
                raise ProtobufDecodeError("ModelMetrics total_time has wrong wire type")
            end = position + 8
            if end > len(data):
                raise ProtobufDecodeError("Truncated ModelMetrics total_time")
            total_time = struct.unpack_from("<d", data, position)[0]
            position = end
        else:
            position = _skip_field(data, position, wire_type)
    return ModelMetrics(tuple(evaluations), trainable_parameters, total_time)


__all__ = [
    "EvaluationData",
    "ModelMetrics",
    "ProtobufDecodeError",
    "parse_model_metrics",
]
