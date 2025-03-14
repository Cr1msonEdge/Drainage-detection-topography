from torch.optim import Adam, AdamW, SGD
from torch.nn import CrossEntropyLoss, BCELoss, MSELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Config:
    """
    Class for configs
    """
    NUM_EPOCHS = 50
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    criterion = None
    optimizer = None
    
    
    def __init__(self, model, opt=None, crit=None, lr=1e-4, num_epochs=50, batch_size=64, scheduler=None):
        self.NUM_EPOCHS = num_epochs
        self.BATCH_SIZE = batch_size
        self.LEARNING_RATE = lr
        self.scheduler = None
        
        if opt is None or opt == 'Adam':
            self.optimizer = Adam(model.model.parameters(), lr=self.LEARNING_RATE)
        elif opt == 'AdamW':
            self.optimizer = AdamW(model.model.parameters(), lr=self.LEARNING_RATE)
        elif opt == 'SGD':
            self.optimizer = SGD(model.model.parameters(), lr=self.LEARNING_RATE)
        else:
            print(f"Optimizer {opt} is not from list of Optimizers")
            return

        if scheduler is not None:
            self.scheduler = ReduceLROnPlateau(optimizer=self.optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-8)
        
        if crit is None or crit == 'CrossEntropy':
            self.criterion = CrossEntropyLoss()
            self.crit_name = 'CrossEntropy'
        elif crit == 'MSE':
            self.crit_name = 'MSE'
            self.criterion = MSELoss()
        elif crit == 'BCE':
            self.crit_name = 'BCE'
            self.criterion = BCELoss()
        else:
            print(f"Criterion {crit} is not from list of Criterions")
            return
        
    def __str__(self):
        return f"Epochs: {self.NUM_EPOCHS}, lr: {self.LEARNING_RATE}, batch_size: {self.BATCH_SIZE}, optimizer: {type(self.optimizer).__name__}"
    
    def get_params(self):
        if self.scheduler is not None:
            return {
                'num_epochs': self.NUM_EPOCHS,
                'batch_size': self.BATCH_SIZE,
                'opt': type(self.optimizer).__name__,
                'crit': self.crit_name,
                'learning_rate': self.LEARNING_RATE,
                'scheduler': 'ReduceLROnPlateau'
            }
            
        return {
            'num_epochs': self.NUM_EPOCHS,
            'batch_size': self.BATCH_SIZE,
            'opt': type(self.optimizer).__name__,
            'crit': self.crit_name,
            'learning_rate': self.LEARNING_RATE,
        }
    