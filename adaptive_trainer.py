from dataclasses import dataclass
import random

@dataclass
class TrainerState:
    assist_mode:str
    recommended_shape:str
    guide_scale:float
    guide_thickness_bonus:int
    motivational_text:str
    focus_tip:str

WEAKNESS_TIPS={
    "accuracy":["Try staying closer to the dotted guide.","Follow the shape outline carefully."],
    "smoothness":["Move your hand slower and steadier.","Relax wrist and draw smoothly."],
    "completion":["Finish the whole shape before releasing.","Try covering the full guide."]
}
MOTIVATION={
    "beginner":["Great start — keep practicing!","Nice effort, let's improve!"],
    "intermediate":["Good progress — keep going!","You're getting stronger!"],
    "advanced":["Excellent performance!","Amazing drawing control!"]
}

def generate_trainer_state(profile):
    snap=profile.snapshot()
    weakness=snap["weakest"]
    skill=snap["skill"]

    if skill=="beginner":
        scale=1.20; bonus=2
    elif skill=="intermediate":
        scale=1.00; bonus=1
    else:
        scale=0.82; bonus=0

    if weakness=="accuracy":
        assist="path_assist"; shape="circle"
    elif weakness=="smoothness":
        assist="stability_assist"; shape="square"
    else:
        assist="completion_assist"; shape="triangle"

    return TrainerState(
        assist_mode=assist,
        recommended_shape=shape,
        guide_scale=scale,
        guide_thickness_bonus=bonus,
        motivational_text=random.choice(MOTIVATION[skill]),
        focus_tip=random.choice(WEAKNESS_TIPS[weakness])
    )
