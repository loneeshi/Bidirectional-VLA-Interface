"""Scene-independent paraphrases: do not invent spatial or success labels."""


def instruction_variants(family, target, original):
    patterns = {
        "reach": [
            "Approach the {target} with the gripper open.",
            "Position the open gripper near the {target}.",
        ],
        "grasp": [
            "Close the gripper around the {target}.",
            "Grasp the {target} with the gripper.",
        ],
        "move": [
            "Move the {target} over the basket.",
            "Transport the {target} to the basket while keeping the gripper closed.",
        ],
        "release": [
            "Open the gripper to release the {target} into the basket.",
            "Release the {target} in the basket.",
        ],
    }
    if target not in ("cream cheese box", "butter"):
        raise ValueError("Unreviewed target for language augmentation")
    return [original] + [p.format(target=target) for p in patterns[family]]
