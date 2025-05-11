from torch.optim import Adam, AdamW, SGD
from torch.nn import CrossEntropyLoss, BCELoss, MSELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torch.cuda import is_available
import time


class Config:
    """
    Class for configs
    """
    
    def __init__(self, opt=None, crit=None, lr=1e-4, num_epochs=100, batch_size=64, scheduler=None, device=None, scheduler_params=None, uid=None, dataset_name=None, num_channels=4, **kwargs):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = lr
        self.scheduler = scheduler
        self.uid = uid
        self.dataset_name = dataset_name
        self.num_channels = num_channels

        # Setting uid
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        if self.uid is None:
            self.uid = timestamp

        # Setting device 
        if device is None:
            device = 'cuda' if is_available() else 'cpu'
        self.device = device

        # Setting optimizer        
        assert opt in ['Adam', 'AdamW', 'SGD', None], f"Optimizer {opt} is not from list of Optimizers"
        self.optimizer = opt if opt is not None else 'AdamW'

        # Setting scheduler
        assert self.scheduler in ['Plateau', 'MultistepLR', None]
        if self.scheduler is not None:
            self.scheduler_params = scheduler_params
            
            # Default setting for schedulers if None
            if self.scheduler_params is None:
                if self.scheduler == 'Plateau':
                    self.scheduler_params = {
                        'mode': 'min',
                        'patience': 10, 
                        'factor': 0.1
                    }
                elif self.scheduler == 'MultistepLR':
                    self.scheduler_params = {
                        'milestones': [30, 80, 150],
                        'gamma': 0.3
                    }
        # Setting num_channels
        assert self.num_channels in [3, 4]
        
        # Setting criterions 
        # ? Add another loss function?
        assert crit in [None, 'CrossEntropy'], f"Criterion {crit} is not from list of Criterions"
        self.criterion = crit if crit is not None else "CrossEntropy"
        
    def __str__(self):
        return f"Epochs: {self.num_epochs}, lr: {self.learning_rate}, batch_size: {self.batch_size}, optimizer: {self.optimizer}, criterion: {self.criterion}"
    
    def to_dict(self):
        return {
            'uid': self.uid,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'opt': self.optimizer,
            'crit': self.criterion,
            'learning_rate': self.learning_rate,
            'scheduler': self.scheduler,
            'dataset_name': self.dataset_name,
            'num_channels': self.num_channels
        }
        