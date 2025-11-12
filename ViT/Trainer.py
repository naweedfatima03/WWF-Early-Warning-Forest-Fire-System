import torch
from torch import nn
from d2l import torch as d2l
from Model import validation_step

class Trainer(d2l.HyperParameters):
    """The base class for training models with data.

    Defined in :numref:`subsec_oo-design-models`"""
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus == 0, 'No GPU support yet'
    
    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        # self.num_train_batches = len(self.train_dataloader)
        self.num_train_batches = 300
        # self.num_val_batches = (len(self.val_dataloader)
        #                         if self.val_dataloader is not None else 0)
        self.num_val_batches = (60 if self.val_dataloader is not None else 0)


    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model
    
    def fit(self, model, data):
        print('Calling fit in Trainer class...')
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            print('Calling fit_epoch in Trainer class...', self.epoch)
            self.fit_epoch()


    def fit_epoch(self):
        raise NotImplementedError

    def prepare_batch(self, batch):
        """Defined in :numref:`sec_linear_scratch`"""
        return batch

    def fit_epoch(self):
        """Defined in :numref:`sec_linear_scratch`"""
        print('Activating train mode...')
        self.model.train()
        counter = self.num_train_batches
        for batch in self.train_dataloader:
            if counter > 0:
                print('Running training for batch...', self.train_batch_idx)
                loss = self.model.training_step(self.prepare_batch(batch))
                self.optim.zero_grad()
                with torch.no_grad():
                    loss.backward()
                    if self.gradient_clip_val > 0:  # To be discussed later
                        self.clip_gradients(self.gradient_clip_val, self.model)
                    self.optim.step()
                self.train_batch_idx += 1
                counter -= 1
        if self.val_dataloader is None:
            return
        print('Activating eval mode...')
        self.model.eval()
        counter = self.num_val_batches
        for batch in self.val_dataloader:
            if counter > 0:
                print('Running validation for batch...', self.val_batch_idx)
                with torch.no_grad():
                    validation_step(self.model, self.prepare_batch(batch))
                self.val_batch_idx += 1
                counter -= 1


    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        """Defined in :numref:`sec_use_gpu`"""
        self.save_hyperparameters()
        self.gpus = [d2l.gpu(i) for i in range(min(num_gpus, d2l.num_gpus()))]
    

    def prepare_batch(self, batch):
        """Defined in :numref:`sec_use_gpu`"""
        if self.gpus:
            batch = [d2l.to(a, self.gpus[0]) for a in batch]
        return batch

    

    def prepare_model(self, model):
        """Defined in :numref:`sec_use_gpu`"""
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        if self.gpus:
            model.to(self.gpus[0])
        self.model = model


    def clip_gradients(self, grad_clip_val, model):
        """Defined in :numref:`sec_rnn-scratch`"""
        params = [p for p in model.parameters() if p.requires_grad]
        norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
        if norm > grad_clip_val:
            for param in params:
                param.grad[:] *= grad_clip_val / norm