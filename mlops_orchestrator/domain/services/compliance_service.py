from __future__ import annotations
from datetime import datetime, UTC
from mlops_orchestrator.domain.value_objects.compliance import (
    ModelCard,
    RiskClassification,
    RiskTier,
)


HIGH_RISK_DOMAINS = frozenset({
    "critical_infrastructure", "education", "employment",
    "essential_services", "law_enforcement", "border_management",
    "justice", "democratic_processes", "healthcare", "finance",
})


class ComplianceService:
    """
    EU AI Act compliance service. Pure domain logic.

    Handles risk classification (Article 6), model card generation (Article 11),
    data governance validation (Article 10), and accuracy requirements (Article 15).
    """

    def classify_risk(
        self,
        domain: str,
        intended_purpose: str,
        impacts_fundamental_rights: bool,
    ) -> RiskClassification:
        if self._is_prohibited(intended_purpose):
            return RiskClassification(
                tier=RiskTier.PROHIBITED,
                domain=domain,
                justification=f"Prohibited use case: {intended_purpose}",
                required_controls=("immediate_withdrawal",),
            )
        if domain in HIGH_RISK_DOMAINS or impacts_fundamental_rights:
            return RiskClassification(
                tier=RiskTier.HIGH,
                domain=domain,
                justification=f"High-risk domain ({domain}) or impacts fundamental rights",
                required_controls=(
                    "risk_management_system",
                    "data_governance",
                    "technical_documentation",
                    "record_keeping",
                    "transparency",
                    "human_oversight",
                    "accuracy_robustness",
                ),
            )
        if self._has_transparency_obligation(intended_purpose):
            return RiskClassification(
                tier=RiskTier.LIMITED,
                domain=domain,
                justification="System has transparency obligations",
                required_controls=("transparency_notice", "disclosure"),
            )
        return RiskClassification(
            tier=RiskTier.MINIMAL,
            domain=domain,
            justification="Minimal risk — voluntary code of conduct",
            required_controls=(),
        )

    def generate_model_card(
        self,
        model_name: str,
        version: str,
        purpose: str,
        limitations: str,
        data_sources: tuple[str, ...],
        accuracy_metrics: tuple[tuple[str, float], ...],
        fairness_assessment: str = "",
    ) -> ModelCard:
        return ModelCard(
            model_name=model_name,
            version=version,
            purpose=purpose,
            limitations=limitations,
            data_sources=data_sources,
            accuracy_metrics=accuracy_metrics,
            fairness_assessment=fairness_assessment,
        )

    def validate_article_10(
        self,
        data_sources: tuple[str, ...],
        has_provenance_tracking: bool,
        has_bias_detection: bool,
    ) -> list[str]:
        """Validate Article 10 data governance requirements. Returns gaps."""
        gaps: list[str] = []
        if not data_sources:
            gaps.append("No data sources documented")
        if not has_provenance_tracking:
            gaps.append("Data provenance tracking not implemented")
        if not has_bias_detection:
            gaps.append("Automated bias detection not configured")
        return gaps

    def validate_article_15(
        self,
        accuracy_metrics: tuple[tuple[str, float], ...],
        has_adversarial_testing: bool,
        has_robustness_testing: bool,
    ) -> list[str]:
        """Validate Article 15 accuracy and robustness requirements. Returns gaps."""
        gaps: list[str] = []
        if not accuracy_metrics:
            gaps.append("No accuracy metrics declared")
        if not has_adversarial_testing:
            gaps.append("Adversarial testing not performed")
        if not has_robustness_testing:
            gaps.append("Robustness testing not performed")
        return gaps

    def _is_prohibited(self, purpose: str) -> bool:
        prohibited_keywords = {
            "social_scoring", "mass_surveillance", "subliminal_manipulation",
            "exploitation_of_vulnerability", "real_time_biometric_identification",
        }
        return any(kw in purpose.lower().replace(" ", "_") for kw in prohibited_keywords)

    def _has_transparency_obligation(self, purpose: str) -> bool:
        transparency_keywords = {"chatbot", "deepfake", "emotion_recognition", "biometric"}
        return any(kw in purpose.lower().replace(" ", "_") for kw in transparency_keywords)
