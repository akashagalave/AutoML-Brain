import time
from config import (
    LATENCY_THRESHOLD,
    LOW_LATENCY_THRESHOLD,
    BREACH_CYCLES,
    RECOVERY_CYCLES,
    COOLDOWN_SECONDS
)

CPU_HIGH_THRESHOLD = 0.7
ERROR_RATE_THRESHOLD = 0.05


class GovernanceState:
    def __init__(self):
        self.breach_count = 0
        self.recovery_count = 0
        self.last_action_time = 0

    def can_act(self):
        return (time.time() - self.last_action_time) > COOLDOWN_SECONDS

    def evaluate(self, p95, cpu, error_rate):
        if p95 is None:
            return None

        # Critical error spike
        if error_rate is not None and error_rate > ERROR_RATE_THRESHOLD:
            if self.can_act():
                self.last_action_time = time.time()
                return "retrain"

        # High latency
        if p95 > LATENCY_THRESHOLD:
            self.breach_count += 1
            self.recovery_count = 0

            if self.breach_count >= BREACH_CYCLES and self.can_act():

                # Infra saturated → scale
                if cpu is not None and cpu > CPU_HIGH_THRESHOLD:
                    self.last_action_time = time.time()
                    self.breach_count = 0
                    return "scale_up"

                # Model inefficient → retrain
                else:
                    self.last_action_time = time.time()
                    self.breach_count = 0
                    return "retrain"

        # Recovery
        elif p95 < LOW_LATENCY_THRESHOLD:
            self.recovery_count += 1
            self.breach_count = 0

            if self.recovery_count >= RECOVERY_CYCLES and self.can_act():
                self.last_action_time = time.time()
                self.recovery_count = 0
                return "scale_down"

        else:
            self.breach_count = 0
            self.recovery_count = 0

        return None