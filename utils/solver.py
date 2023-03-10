import time
import os
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple

from csi_reference.CRNet.utils import logger
from csi_reference.CRNet.utils.statics import AverageMeter, evaluator


__all__ = ['Trainer', 'Tester', 'EntroTrainer', 'EntroTester']


field = ('nmse', 'rho', 'epoch')
Result = namedtuple('Result', field, defaults=(None,) * len(field))


class Trainer:
    r""" The training pipeline for encoder-decoder architecture
    """

    def __init__(self, model, device, optimizer, criterion, scheduler, resume=None,
                 save_path='./checkpoints', print_freq=20, val_freq=1, test_freq=1,
                 modelSaver=None, logger=None):

        # Basic arguments
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device

        # Verbose arguments
        self.resume_file = resume
        self.save_path = save_path
        self.print_freq = print_freq
        self.val_freq = val_freq
        self.test_freq = test_freq

        # Pipeline arguments
        self.cur_epoch = 1
        self.all_epoch = None
        self.train_loss = None
        self.val_loss = None
        self.test_loss = None
        self.best_rho = Result()
        self.best_nmse = Result()

        self.tester = Tester(model, device, criterion, print_freq)
        self.test_loader = None

        self.modelSaver = modelSaver
        self.logger = logger


    def loop(self, epochs, train_loader, val_loader, test_loader):
        r""" The main loop function which runs training and validation iteratively.

        Args:
            epochs (int): The total epoch for training
            train_loader (DataLoader): Data loader for training data.
            val_loader (DataLoader): Data loader for validation data.
            test_loader (DataLoader): Data loader for test data.
        """

        self.all_epoch = epochs
        self._resume()

        for ep in range(self.cur_epoch, epochs + 1):
            self.cur_epoch = ep

            # conduct training, validation and test
            self.train_loss = self.train(train_loader)
            if ep % self.val_freq == 0:
                self.val_loss = self.val(val_loader)

            if ep % self.test_freq == 0:
                self.test_loss, nmse = self.test(test_loader)
                rho = None
            else:
                rho, nmse = None, None

            # conduct saving, visualization and log printing
            self._loop_postprocessing(rho, nmse, ep)

    def train(self, train_loader):
        r""" train the model on the given data loader for one epoch.

        Args:
            train_loader (DataLoader): the training data loader
        """

        self.model.train()
        with torch.enable_grad():
            return self._iteration(train_loader)

    def val(self, val_loader):
        r""" exam the model with validation set.

        Args:
            val_loader: (DataLoader): the validation data loader
        """

        self.model.eval()
        with torch.no_grad():
            return self._iteration(val_loader)

    def test(self, test_loader):
        r""" Truly test the model on the test dataset for one epoch.

        Args:
            test_loader (DataLoader): the test data loader
        """

        self.model.eval()
        with torch.no_grad():
            return self.tester(test_loader, verbose=False)

    def _iteration(self, data_loader):
        iter_loss = AverageMeter('Iter loss')
        iter_time = AverageMeter('Iter time')
        time_tmp = time.time()

        for batch_idx, (sparse_gt) in enumerate(data_loader):
            sparse_gt = sparse_gt.to(self.device)
            sparse_pred = self.model(sparse_gt)
            loss = self.criterion(sparse_pred, sparse_gt)

            # Scheduler update, backward pass and optimization
            if self.model.training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            # Log and visdom update
            iter_loss.update(loss)
            iter_time.update(time.time() - time_tmp)
            time_tmp = time.time()

            # plot progress
            if (batch_idx + 1) % self.print_freq == 0:
                logger.info(f'Epoch: [{self.cur_epoch}/{self.all_epoch}]'
                            f'[{batch_idx + 1}/{len(data_loader)}] '
                            f'lr: {self.scheduler.get_lr()[0]:.2e} | '
                            f'MSE loss: {iter_loss.avg:.3e} | '
                            f'time: {iter_time.avg:.3f}')

        mode = 'Train' if self.model.training else 'Val'
        logger.info(f'=> {mode}  Loss: {iter_loss.avg:.3e}\n')

        return iter_loss.avg

    def _save(self, state, name):
        if self.save_path is None:
            logger.warning('No path to save checkpoints.')
            return

        os.makedirs(self.save_path, exist_ok=True)
        torch.save(state, os.path.join(self.save_path, name))

    def _resume(self):
        r""" protected function which resume from checkpoint at the beginning of training.
        """

        if self.resume_file is None:
            return None
        assert os.path.isfile(self.resume_file)
        logger.info(f'=> loading checkpoint {self.resume_file}')
        checkpoint = torch.load(self.resume_file)
        self.cur_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.best_rho = checkpoint['best_rho']
        self.best_nmse = checkpoint['best_nmse']
        self.cur_epoch += 1  # start from the next epoch

        logger.info(f'=> successfully loaded checkpoint {self.resume_file} '
                    f'from epoch {checkpoint["epoch"]}.\n')

    def _loop_postprocessing(self, rho, nmse, ep):
        r""" private function which makes loop() function neater.
        """

        # save state generate
        state = {
            'epoch': self.cur_epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_rho': self.best_rho,
            'best_nmse': self.best_nmse
        }

        # save model with best rho and nmse
        if rho is not None:
            if self.best_rho.rho is None or self.best_rho.rho < rho:
                self.best_rho = Result(rho=rho, nmse=nmse, epoch=self.cur_epoch)
                state['best_rho'] = self.best_rho
                self._save(state, name=f"best_rho.pth")
            if self.best_nmse.nmse is None or self.best_nmse.nmse > nmse:
                self.best_nmse = Result(rho=rho, nmse=nmse, epoch=self.cur_epoch)
                state['best_nmse'] = self.best_nmse
                self._save(state, name=f"best_nmse.pth")

        # self._save(state, name='last.pth')
        self.logger.write("epoch%d, nmse_loss=%.5f\n" % (ep, nmse))
        self.logger.flush()

        # self.modelSaver.model_save(state, save_path=self.save_path + ('/CRnet_EP%d.model' % ep), loss=nmse)
        self.modelSaver.model_save(self.model.state_dict(), save_path=self.save_path + ('/CRnet_EP%d.model' % ep), loss=nmse)

        # print current best results
        if self.best_rho.rho is not None:
            print(f'\n=! Best rho: {self.best_rho.rho:.3e} ('
                  f'Corresponding nmse={self.best_rho.nmse:.3e}; '
                  f'epoch={self.best_rho.epoch})'
                  f'\n   Best NMSE: {self.best_nmse.nmse:.3e} ('
                  f'Corresponding rho={self.best_nmse.rho:.3e};  '
                  f'epoch={self.best_nmse.epoch})\n')


class Tester:
    r""" The testing interface for classification
    """

    def __init__(self, model, device, criterion, print_freq=20):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.print_freq = print_freq

    def __call__(self, test_data, verbose=True):
        r""" Runs the testing procedure.

        Args:
            test_data (DataLoader): Data loader for validation data.
        """

        self.model.eval()
        with torch.no_grad():
            loss, nmse = self._iteration(test_data)
        if verbose:
            print(f'\n=> Test result: \nloss: {loss:.3e}'
                  f'    NMSE: {nmse:.3e}\n')
        return loss, nmse

    def _iteration(self, data_loader):
        r""" protected function which test the model on given data loader for one epoch.
        """

        iter_rho = AverageMeter('Iter rho')
        iter_nmse = AverageMeter('Iter nmse')
        iter_loss = AverageMeter('Iter loss')
        iter_time = AverageMeter('Iter time')
        time_tmp = time.time()

        for batch_idx, (sparse_gt) in enumerate(data_loader):
            sparse_gt = sparse_gt.to(self.device)
            sparse_pred = self.model(sparse_gt)
            loss = self.criterion(sparse_pred, sparse_gt)
            nmse = evaluator(sparse_pred, sparse_gt)

            # Log and visdom update
            iter_loss.update(loss)
            iter_nmse.update(nmse)
            iter_time.update(time.time() - time_tmp)
            time_tmp = time.time()

            # plot progress
            if (batch_idx + 1) % self.print_freq == 0:
                logger.info(f'[{batch_idx + 1}/{len(data_loader)}] '
                            f'loss: {iter_loss.avg:.3e} | '
                            f'NMSE: {iter_nmse.avg:.3e} | time: {iter_time.avg:.3f}')

        logger.info(f'=> Test NMSE: {iter_nmse.avg:.3e}\n')

        return iter_loss.avg, iter_nmse.avg
    
    
class EntroTrainer(Trainer):
    r""" The training pipeline for encoder-decoder architecture of entroformer
    """
    def __init__(self, model, device, optimizer, criterion, scheduler, resume=None, 
                 save_path='./checkpoints', print_freq=20, val_freq=1, test_freq=1, 
                 modelSaver=None, logger=None, 
                 batchSize=8, alpha=0.01, grad_norm_clip=0.):
        super().__init__(model, device, optimizer, criterion, scheduler, resume, 
                         save_path, print_freq, val_freq, test_freq, modelSaver, logger)
        self.num_pixels = 1024
        self.batchSize = batchSize
        self.alpha = alpha
        self.grad_norm_clip = grad_norm_clip
        self.tester = EntroTester(model, device, criterion, print_freq, batchSize, alpha)
        
    def _iteration(self, data_loader):
        iter_loss = AverageMeter('Iter loss')
        iter_time = AverageMeter('Iter time')
        time_tmp = time.time()
        
        for batch_idx, (sparse_gt) in enumerate(data_loader):
            sparse_gt = sparse_gt.to(self.device)
            n, _, _, _= sparse_gt.shape
            
            # Updata lr
            self.scheduler.update_lr(batch_size=n)
            current_lr = self.scheduler.get_lr()
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
            
            x_tilde, y_tilde, z_prob, predicted_param = self.model(sparse_gt)
            # here to calu loss
            criterion_mse = nn.MSELoss()
            loss_mse = criterion_mse(x_tilde, sparse_gt) * 255 * 255
            loss_rate_z = - torch.log2(z_prob + 1e-10).sum() / self.num_pixels / self.batchSize
            loss_rate_y = self.criterion(y_tilde, predicted_param).sum() / np.log(2) / self.num_pixels / self.batchSize
            total_loss = loss_mse * self.alpha + loss_rate_y + loss_rate_z
            
            # Scheduler update, backward pass and optimization
            if self.model.training:
                self.optimizer.zero_grad()
                total_loss.backward()
                # Gradient Clipping
                if self.grad_norm_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
                self.optimizer.step()
                # self.scheduler.step()

            # Log and visdom update
            iter_loss.update(total_loss)
            iter_time.update(time.time() - time_tmp)
            time_tmp = time.time()

            # plot progress
            if (batch_idx + 1) % self.print_freq == 0:
                logger.info(f'Epoch: [{self.cur_epoch}/{self.all_epoch}]'
                            f'[{batch_idx + 1}/{len(data_loader)}] '
                            f'lr: {self.scheduler.get_lr():.2e} | '
                            f'MSE loss: {iter_loss.avg:.3e} | '
                            f'time: {iter_time.avg:.3f}')

        mode = 'Train' if self.model.training else 'Val'
        logger.info(f'=> {mode}  Loss: {iter_loss.avg:.3e}\n')

        return iter_loss.avg
        

class EntroTester(Tester):
    r""" The testing interface for classification
    """
    def __init__(self, model, device, criterion, print_freq=20,
                 batchSize=8, alpha=0.01):
        super().__init__(model, device, criterion, print_freq)
        self.num_pixels = 1024
        self.batchSize = batchSize
        self.alpha = alpha
        
    def _iteration(self, data_loader):

        iter_nmse = AverageMeter('Iter nmse')
        iter_loss = AverageMeter('Iter loss')
        iter_time = AverageMeter('Iter time')
        time_tmp = time.time()

        for batch_idx, (sparse_gt) in enumerate(data_loader):
            sparse_gt = sparse_gt.to(self.device)
            x_tilde, y_tilde, z_prob, predicted_param = self.model(sparse_gt)
            # here to calu loss
            criterion_mse = nn.MSELoss()
            loss_mse = criterion_mse(x_tilde, sparse_gt) * 255 * 255
            loss_rate_z = - torch.log2(z_prob + 1e-10).sum() / self.num_pixels / self.batchSize
            loss_rate_y = self.criterion(y_tilde, predicted_param).sum() / np.log(2) / self.num_pixels / self.batchSize
            total_loss = loss_mse * self.alpha + loss_rate_y + loss_rate_z
            
            loss = total_loss
            nmse = evaluator(x_tilde, sparse_gt)

            # Log and visdom update
            iter_loss.update(loss)
            iter_nmse.update(nmse)
            iter_time.update(time.time() - time_tmp)
            time_tmp = time.time()

            # plot progress
            if (batch_idx + 1) % self.print_freq == 0:
                logger.info(f'[{batch_idx + 1}/{len(data_loader)}] '
                            f'loss: {iter_loss.avg:.3e} | '
                            f'NMSE: {iter_nmse.avg:.3e} | time: {iter_time.avg:.3f}')

        logger.info(f'=> Test NMSE: {iter_nmse.avg:.3e}\n')

        return iter_loss.avg, iter_nmse.avg
    