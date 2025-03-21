from torch.optim import Adam, AdamW, SGD
from torch.nn import CrossEntropyLoss, BCELoss, MSELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torch.cuda import is_available

class Config:
    """
    Class for configs
    """
    
    def __init__(self, opt=None, crit=None, lr=1e-4, num_epochs=100, batch_size=64, scheduler=None, device=None, scheduler_params=None):
        self.NUM_EPOCHS = num_epochs
        self.BATCH_SIZE = batch_size
        self.LEARNING_RATE = lr
        self.scheduler = scheduler
        
        # Setting device 
        if device is None:
            device = 'cuda' if is_available() else 'cpu'
        
        # Setting optimizer        
        assert opt in ['Adam', 'AdamW', 'SGD'], f"Optimizer {opt} is not from list of Optimizers"
        self.optimizer = opt

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
            
        # Setting criterions 
        assert crit in [None, 'CrossEntropy'], f"Criterion {crit} is not from list of Criterions"
        self.criterion = crit
        
    def __str__(self):
        return f"Epochs: {self.NUM_EPOCHS}, lr: {self.LEARNING_RATE}, batch_size: {self.BATCH_SIZE}, optimizer: {self.optimizer}, criterion: {self.criterion}"
    
    def get_params(self):
        return {
            'num_epochs': self.NUM_EPOCHS,
            'batch_size': self.BATCH_SIZE,
            'opt': type(self.optimizer).__name__,
            'crit': self.crit_name,
            'learning_rate': self.LEARNING_RATE,
            'scheduler': self.scheduler,
        }
        

    