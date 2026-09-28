"""Sign vocabulary. Edit this list in Week 0 after review with an ASL-fluent person.

The NO_SIGN class is required: it is what the model predicts when the signer
is idle or between signs, and it is what keeps the device from speaking
random words. Record plenty of it (idle hands, scratching, adjusting camera).
"""

NO_SIGN = "NO_SIGN"

SIGNS = [
    "HELLO", "THANK_YOU", "PLEASE", "SORRY", "YES", "NO",
    "HELP", "NAME", "WHAT", "WHERE", "WANT", "NEED",
    "EAT", "DRINK", "WATER", "BATHROOM", "MORE", "STOP",
    "GOOD", "BAD", "ME", "YOU", "LOVE", "FRIEND",
    "HOME", "WORK", "LEARN", "TIME",
]

# Control signs: recognized but not spoken; they drive the sentence buffer.
CONTROL_SIGNS = ["CLEAR", "SPEAK"]

ALL_CLASSES = [NO_SIGN] + SIGNS + CONTROL_SIGNS

# What the TTS actually says for each class.
SPOKEN = {s: s.replace("_", " ").lower() for s in SIGNS}
SPOKEN.update({"THANK_YOU": "thank you", "ME": "I", "BATHROOM": "bathroom"})

CLASS_TO_ID = {c: i for i, c in enumerate(ALL_CLASSES)}
ID_TO_CLASS = {i: c for c, i in CLASS_TO_ID.items()}
