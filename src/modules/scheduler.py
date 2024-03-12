import torch
from transformers import get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup, get_linear_schedule_with_warmup

class SchedulerManager:

    """
    a manager class to manage the choice of scheduler chose
    """

    def __init__(
        self, 
        scheduler_name: str,
        optimizer: torch.optim, 
        warmup_ratio: float, 
        num_training_steps: int,
    ) -> None:
        
        self.scheduler_name = scheduler_name
        self.optimizer = optimizer
        self.warmup_ratio = warmup_ratio
        self.num_training_steps = num_training_steps

    def __call__(self) -> torch.optim.lr_scheduler:

        if self.scheduler_name == 'cosine_schedule_with_warmup':
            return get_cosine_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=int(self.num_training_steps * self.warmup_ratio),
                num_training_steps=self.num_training_steps,
            )

        elif self.scheduler_name == 'cosine_with_hard_restarts_schedule_with_warmup':
            return get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=int(self.num_training_steps * self.warmup_ratio),
                num_training_steps=self.num_training_steps,
                num_cycles=1 # number of hard restart to use
            )
        
        else: # default scheduler, linear_schedule_with_warmup
            return get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=int(self.num_training_steps * self.warmup_ratio),
                num_training_steps=self.num_training_steps,
            )
