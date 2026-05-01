from dataclasses import dataclass, field
from collections import deque
import statistics

HISTORY_SIZE = 12
IMPROVEMENT_WINDOW = 5

@dataclass
class AdaptiveProfile:
    accuracy_hist: deque = field(default_factory=lambda: deque(maxlen=HISTORY_SIZE))
    smoothness_hist: deque = field(default_factory=lambda: deque(maxlen=HISTORY_SIZE))
    completion_hist: deque = field(default_factory=lambda: deque(maxlen=HISTORY_SIZE))
    final_hist: deque = field(default_factory=lambda: deque(maxlen=HISTORY_SIZE))
    shape_hist: deque = field(default_factory=lambda: deque(maxlen=HISTORY_SIZE))
    total_attempts: int = 0

    def update(self, score_dict, shape_name):
        self.accuracy_hist.append(score_dict["accuracy"])
        self.smoothness_hist.append(score_dict["smoothness"])
        self.completion_hist.append(score_dict["completion"])
        self.final_hist.append(score_dict["final"])
        self.shape_hist.append(shape_name)
        self.total_attempts += 1

    def avg_accuracy(self): return statistics.mean(self.accuracy_hist) if self.accuracy_hist else 0
    def avg_smoothness(self): return statistics.mean(self.smoothness_hist) if self.smoothness_hist else 0
    def avg_completion(self): return statistics.mean(self.completion_hist) if self.completion_hist else 0
    def avg_final(self): return statistics.mean(self.final_hist) if self.final_hist else 0

    def weakest_area(self):
        vals={"accuracy":self.avg_accuracy(),"smoothness":self.avg_smoothness(),"completion":self.avg_completion()}
        return min(vals,key=vals.get)

    def strongest_area(self):
        vals={"accuracy":self.avg_accuracy(),"smoothness":self.avg_smoothness(),"completion":self.avg_completion()}
        return max(vals,key=vals.get)

    def improving(self):
        if len(self.final_hist)<IMPROVEMENT_WINDOW: return False
        recent=list(self.final_hist)[-IMPROVEMENT_WINDOW:]
        return statistics.mean(recent[len(recent)//2:]) > statistics.mean(recent[:len(recent)//2])

    def skill_level(self):
        avg=self.avg_final()
        if avg>=88:return "advanced"
        elif avg>=68:return "intermediate"
        return "beginner"

    def snapshot(self):
        return {
            "attempts":self.total_attempts,
            "avg_accuracy":round(self.avg_accuracy(),1),
            "avg_smoothness":round(self.avg_smoothness(),1),
            "avg_completion":round(self.avg_completion(),1),
            "avg_final":round(self.avg_final(),1),
            "weakest":self.weakest_area(),
            "strongest":self.strongest_area(),
            "improving":self.improving(),
            "skill":self.skill_level()
        }
