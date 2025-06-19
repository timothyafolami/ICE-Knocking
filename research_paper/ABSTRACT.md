# Abstract

Engine knock detection represents a critical challenge in modern automotive engineering, where early identification of combustion anomalies is essential for preventing catastrophic engine damage and optimizing performance. This research presents a comprehensive intelligent knock detection and predictive maintenance system that integrates advanced neural network architectures with physics-based modeling for real-time automotive applications.

## Motivation and Problem

Traditional knock detection methods rely on threshold-based approaches using knock sensors, which suffer from high false positive rates, limited adaptability to varying operating conditions, and reactive rather than predictive capabilities. The severe class imbalance inherent in engine knock data (typically <2% of operating time) poses significant challenges for conventional machine learning approaches, while real-time automotive applications demand sub-millisecond response times with computational efficiency suitable for embedded systems.

## Methodology

We developed a novel two-tier hybrid architecture combining machine learning-based parameter forecasting with intelligent knock detection. The system utilizes LSTM networks for 24-hour engine parameter prediction, physics-based derivation of secondary parameters, and five specialized neural network architectures for knock detection: Deep Dense Networks, Residual Networks, Attention Networks, Wide & Deep Networks, and a novel Ensemble approach. Comprehensive feature engineering transforms 9 original engine parameters into 48 enhanced features incorporating temporal patterns, rolling statistics, physics-based interactions, and engine stress indicators. Advanced techniques including focal loss, balanced class weighting, and ensemble methods address the severe class imbalance (1:69 ratio) typical in automotive fault detection.

## Key Results

Experimental validation using realistic engine data (10,080 minutes, 144 knock events) demonstrates significant performance improvements over traditional approaches. The best-performing Ensemble_Baseline architecture achieved 87.23% ROC-AUC with 82.76% recall, representing a 99.9% improvement in recall over the best traditional machine learning method (41.38%). The model requires only 30,452 parameters, enabling real-time inference (<2ms) suitable for automotive Electronic Control Units (ECUs). Real-world validation on 24-hour forecasted data successfully predicted 231 knock events with detailed temporal analysis, demonstrating practical predictive maintenance capabilities.

## Significance and Impact

This research makes several significant contributions: (1) Introduction of a novel ensemble neural network architecture specifically optimized for imbalanced automotive fault detection, (2) Development of a hybrid ML-physics integration approach combining data-driven forecasting with domain knowledge, (3) Comprehensive feature engineering framework yielding 11.4% performance improvement, and (4) Production-ready system design meeting automotive deployment constraints. The integrated forecasting-detection pipeline enables proactive maintenance strategies, potentially preventing costly engine damage while optimizing fuel efficiency and emissions.

## Conclusions

The proposed intelligent knock detection system successfully addresses critical challenges in automotive fault detection while maintaining computational efficiency required for production deployment. The 82.76% recall rate ensures robust engine protection by detecting most knock events, while the predictive maintenance capabilities enable proactive intervention 24 hours in advance. This research establishes a foundation for next-generation automotive diagnostic systems and contributes to the advancement of AI-driven predictive maintenance in safety-critical applications.

**Keywords:** Engine knock detection, Neural networks, Imbalanced classification, Predictive maintenance, Automotive AI, LSTM forecasting, Ensemble learning, Real-time systems

---

**Word Count:** 492 words
**Research Classification:** Applied Machine Learning, Automotive Engineering, Fault Detection
**Application Domain:** Automotive Systems, Predictive Maintenance, Safety-Critical AI