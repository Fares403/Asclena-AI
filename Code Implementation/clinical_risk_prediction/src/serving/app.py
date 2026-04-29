"""FastAPI app for stateless Asclena clinical risk model inference."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from .config import get_settings
from .predictor import (
    CONTRACT_VERSION,
    ContractValidationError,
    feature_contract,
    load_prediction_artifacts,
    model_metadata,
    predict_one,
)
from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    ExplanationPayload,
    PredictionRequest,
    PredictionResponse,
    PredictionResult,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.artifacts = load_prediction_artifacts(settings.model_path)
    yield


app = FastAPI(
    title="Asclena Clinical Risk Model API",
    version="1.0.0",
    summary="Stateless XGBoost inference service for ED clinical risk prediction.",
    description=(
        "This service exposes a stateless contract for Asclena AI to request clinical risk predictions. "
        "The payload is normalized for future FHIR/EHR adapters but does not require FHIR resources today."
    ),
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/health")
def versioned_health() -> dict[str, str]:
    return {"status": "ok", "contract_version": CONTRACT_VERSION}


@app.get("/v1/model")
def get_model_metadata() -> dict[str, object]:
    return model_metadata(app.state.artifacts)


@app.get("/v1/contract")
def get_contract() -> dict[str, object]:
    return feature_contract(app.state.artifacts)


@app.post(
    "/v1/predict",
    response_model=PredictionResponse,
    response_model_exclude_none=True,
    responses={
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def predict(
    request: PredictionRequest,
    include_feature_snapshot: bool = Query(
        default=False,
        description="Include the full ordered feature snapshot in the explanation payload.",
    ),
) -> PredictionResponse:
    try:
        result = predict_one(
            app.state.artifacts,
            request.features,
            include_feature_snapshot=include_feature_snapshot,
        )
    except ContractValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error_code="invalid_feature_contract",
                message=str(exc),
                details={
                    "required_features": app.state.artifacts.feature_names,
                    "missing_features": exc.missing_features,
                    "unknown_features": exc.unknown_features,
                },
            ).model_dump(),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse(
                error_code="prediction_failed",
                message="The model service could not produce a prediction.",
                details={"reason": str(exc)},
            ).model_dump(),
        ) from exc

    return PredictionResponse(
        request_id=request.request_id,
        subject=request.subject,
        model=model_metadata(app.state.artifacts),
        prediction=PredictionResult(
            risk_score=result["risk_score"],
            predicted_target=result["predicted_target"],
            risk_label=result["risk_label"],
            threshold_used=result["threshold_used"],
        ),
        explanation=ExplanationPayload(
            top_contributors=result["top_contributors"],
            feature_snapshot=result.get("feature_snapshot"),
        ),
        contract_version=CONTRACT_VERSION,
    )


@app.post(
    "/v1/predict/batch",
    response_model=BatchPredictionResponse,
    response_model_exclude_none=True,
    responses={
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def batch_predict(
    request: BatchPredictionRequest,
    include_feature_snapshot: bool = Query(
        default=False,
        description="Include the full ordered feature snapshot in each prediction explanation payload.",
    ),
) -> BatchPredictionResponse:
    if len(request.instances) > app.state.settings.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                error_code="batch_too_large",
                message=f"Batch size exceeds limit of {app.state.settings.max_batch_size}.",
            ).model_dump(),
        )

    predictions = []
    for instance in request.instances:
        try:
            result = predict_one(
                app.state.artifacts,
                instance.features,
                include_feature_snapshot=include_feature_snapshot,
            )
        except ContractValidationError as exc:
            raise HTTPException(
                status_code=400,
                detail=ErrorResponse(
                    error_code="invalid_feature_contract",
                    message=str(exc),
                    details={
                        "request_id": instance.request_id,
                        "required_features": app.state.artifacts.feature_names,
                        "missing_features": exc.missing_features,
                        "unknown_features": exc.unknown_features,
                    },
                ).model_dump(),
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=ErrorResponse(
                    error_code="prediction_failed",
                    message="The model service could not produce a prediction.",
                    details={
                        "request_id": instance.request_id,
                        "reason": str(exc),
                    },
                ).model_dump(),
            ) from exc
        predictions.append(
            PredictionResponse(
                request_id=instance.request_id,
                subject=instance.subject,
                model=model_metadata(app.state.artifacts),
                prediction=PredictionResult(
                    risk_score=result["risk_score"],
                    predicted_target=result["predicted_target"],
                    risk_label=result["risk_label"],
                    threshold_used=result["threshold_used"],
                ),
                explanation=ExplanationPayload(
                    top_contributors=result["top_contributors"],
                    feature_snapshot=result.get("feature_snapshot"),
                ),
                contract_version=CONTRACT_VERSION,
            )
        )

    return BatchPredictionResponse(
        model=model_metadata(app.state.artifacts),
        predictions=predictions,
        batch_size=len(predictions),
        contract_version=CONTRACT_VERSION,
    )
