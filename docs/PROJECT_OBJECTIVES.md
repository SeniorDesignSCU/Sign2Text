# Signs to Speech — Project Objectives

**Team:** Aditya Thakrar, Karan Velayutham · Santa Clara University, Electrical & Computer Engineering · Senior Design 2026–27

Signs to Speech is a portable, edge-processed device that recognizes American Sign Language and speaks the result aloud, giving Deaf and hard-of-hearing users a way to communicate with people who do not sign, without relying on a phone app or cloud connectivity.

The primary objective is to recognize a vocabulary of at least 25 common conversational ASL word and phrase signs in real time and output them as synthesized speech, with all inference running locally on the device. The system captures hand and upper-body landmarks from a camera, classifies signs using a temporal machine-learning model trained on our own dataset, and produces spoken output through an integrated speaker within one second of the sign being completed.

A secondary objective is to improve recognition of handshape-ambiguous signs by fusing camera data with a wearable glove that reports finger flexion and hand motion, and to provide static fingerspelling as a fallback for names and words outside the vocabulary.

The project targets at least 90% top-1 accuracy on the team's own signing and at least 80% on signers not represented in the training data, verified through scripted user testing with at least five participants. The deliverable is a standalone, battery-powered prototype that boots directly into the application with no laptop attached, along with the training dataset, source code, and documentation sufficient for another team to reproduce and extend the work.

The device is explicitly scoped as an isolated-sign recognizer, not a full ASL-to-English translator; continuous signing, facial grammar, and sentence-level translation are outside the project's scope. On-screen text is a secondary output used for confirmation and debugging.

## Measurable targets

| Metric | Target |
|---|---|
| Vocabulary | ≥ 25 word/phrase signs (+ A–Z fingerspelling as secondary mode) |
| Top-1 accuracy, team signers | ≥ 90% |
| Top-1 accuracy, unseen signers | ≥ 80% |
| Latency, end of sign → speech start | ≤ 1 s |
| Landmark throughput on device | ≥ 10 FPS |
| User testing | ≥ 5 participants, ≥ 1 fluent signer |
| Form factor | Standalone, battery-powered, boots into app |
