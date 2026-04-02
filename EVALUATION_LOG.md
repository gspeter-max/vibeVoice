# EVALUATION & STRATEGY LOG: parakeet-flow

## 📋 System Status Overview
**Architect Persona:** Evaluation & Strategy Architect
**Current Focus:** Baseline System Analysis & Gap Identification
**Last Updated:** 2026-04-02

---

## 🔍 Active Implementation Monitoring
*Monitoring changes across src/ and tests/ to evaluate parallel agent performance.*

| Component | Status | Strategy Alignment | Gaps Identified |
| :--- | :--- | :--- | :--- |
| **ASR Backend** | ⚠️ MISALIGNED | Low (Using Offline for Streaming) | **Batch-over-Socket:** Currently uses `OfflineRecognizer`, missing real-time partials. |
| **HUD / UI** | ✅ ALIGNED | High (iOS-style pill) | **Hold Delay:** 1.0s mouse-hold threshold may feel sluggish for power users. |
| **Ear (Audio Input)** | 🛠️ FUNCTIONAL | Medium (Socket-based) | **Gain Hardcoding:** `gain_multiplier` is hardcoded to 2.5; lacks auto-gain control (AGC). |
| **Brain (Logic)** | 🚀 READY | High (VAD-enabled) | **Partial Trigger:** VAD exists but only triggers HUD states; needs to trigger partial transcription. |

---

## 🌐 Deep Research & Best Practices
*Comparative analysis against industry standards (Faster-Whisper, Parakeet-RNNT, OpenVINO optimization).*

- [x] **Optimization Research:** Found that **Parakeet-TDT** is ~64% faster than standard RNN-T.
- [x] **Streaming Latency:** Industry benchmark for "instant" feel is **<500ms**; current batch approach is likely >1s.
- [x] **VAD Strategy:** Confirmed Silero VAD (already in Brain) is the optimal companion for Parakeet-TDT "Simulated Streaming."
- [ ] **Technical Debt:** Researching `sherpa-onnx` OnlineRecognizer migration to enable partial results during speech.

---

## 🚀 Recommended Improvements (The "Toggle" List)
*Proposed changes for user review. High-priority items to be delegated to agents.*

1. **[ASR] Switch to OnlineRecognizer:** Migrate `backend_parakeet.py` from `OfflineRecognizer` to `OnlineRecognizer` to enable "text-as-you-speak" (Partial Results).
2. **[Ear] Dynamic Thresholds:** Reduce mouse-hold threshold from 1.0s to 0.4s (matching keyboard threshold) for better responsiveness.
3. **[HUD] Partial Result Support:** Update HUD to display greyed-out partial transcripts while the user is still speaking.

---

## 🛑 Gap Analysis
*Identified deficiencies in the current codebase or agent implementation.*

- **The "Streaming Illusion":** The system uses sockets to send chunks, but the backend waits for the socket to close before processing. This creates a bottleneck at the end of the phrase.
- **Hardware Agnosticism:** `backend_openvino.py` exists but isn't the default; Parakeet is faster but requires specific ONNX exports.
