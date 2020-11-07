import copy
import datetime
import os
import random
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from inference.inference_utils import get_trange, get_tqdm


def init_random_seed(value=0):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    torch.backends.cudnn.deterministic = True


def copy_data_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return [copy_data_to_device(elem, device) for elem in data]
    elif isinstance(data, dict):
        return {name: copy_data_to_device(value, device) for name, value in data.items()}
    raise ValueError('Unexpected data type {}'.format(type(data)))


def sum_dicts(current, new):
    if current is None:
        return new
    result = dict(current)
    for name, new_value in new.items():
        result[name] = result.get(name, 0) + new_value
    return result


def norm_dict(current, n):
    if n == 0:
        return current
    return {name: value / (n + 1e-6) for name, value in current.items()}


def train_eval_loop(model, train_dataset, val_dataset, criterion,
                    lr=1e-4, epoch_n=10, batch_size=32,
                    device='cuda', early_stopping_patience=10, l2_reg_alpha=0,
                    max_batches_per_epoch_train=10000,
                    max_batches_per_epoch_val=1000,
                    data_loader_ctor=DataLoader,
                    optimizer_ctor=None,
                    lr_scheduler_ctor=None,
                    shuffle_train=True,
                    dataloader_workers_n=0,
                    clip_grad=10,
                    save_vis_images_path=None,
                    save_vis_images_freq=100,
                    save_models_path=None,
                    save_models_freq=10):
    device = torch.device(device)
    model.to(device)

    if optimizer_ctor is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg_alpha)
    else:
        optimizer = optimizer_ctor(model.parameters(), lr=lr)

    if lr_scheduler_ctor is not None:
        lr_scheduler = lr_scheduler_ctor(optimizer)
    else:
        lr_scheduler = None

    train_dataloader = data_loader_ctor(train_dataset, batch_size=batch_size, shuffle=shuffle_train,
                                        num_workers=dataloader_workers_n)
    val_dataloader = data_loader_ctor(val_dataset, batch_size=batch_size, shuffle=False,
                                      num_workers=dataloader_workers_n)

    best_val_loss = float('inf')
    best_val_metrics = None
    best_epoch_i = 0
    best_model = copy.deepcopy(model)

    for epoch_i in get_trange(epoch_n, desc='Epochs'):
        try:
            epoch_start = datetime.datetime.now()
            print('Epoch {}'.format(epoch_i))

            model.train()
            mean_train_loss = 0
            mean_train_metrics = None
            train_batches_n = 0
            for batch_i, (batch_x, batch_y) in get_tqdm(enumerate(train_dataloader), desc=f'Epoch {epoch_i}',
                                                        total=max_batches_per_epoch_train, leave=True):
                if batch_i > max_batches_per_epoch_train:
                    break

                batch_x = copy_data_to_device(batch_x, device)
                batch_y = copy_data_to_device(batch_y, device)

                pred = model(batch_x)
                loss, metrics, vis_img = criterion(pred, batch_y)

                model.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

                optimizer.step()

                mean_train_loss += float(loss)
                mean_train_metrics = sum_dicts(mean_train_metrics, metrics)
                if vis_img is not None and save_vis_images_path is not None and batch_i % save_vis_images_freq == 0:
                    save_image(vis_img,
                               os.path.join(save_vis_images_path,
                                            'epoch{:04d}_iter{:06d}_train.jpg'.format(epoch_i, batch_i)),
                               nrow=batch_y['images'].shape[0],
                               normalize=True,
                               range=(-1, 1))

                train_batches_n += 1

            mean_train_loss /= train_batches_n
            mean_train_metrics = norm_dict(mean_train_metrics, train_batches_n)
            print('Epoch: {} iterations, {:0.2f} sec'.format(train_batches_n,
                                                           (datetime.datetime.now() - epoch_start).total_seconds()))
            print('Mean train loss', mean_train_loss, mean_train_metrics)

            if save_models_path is not None and epoch_i % save_models_freq == 0:
                torch.save(model, os.path.join(save_models_path, 'model_epoch_{:04d}.pth'.format(epoch_i)))

            model.eval()
            mean_val_loss = 0
            mean_val_metrics = None
            val_batches_n = 0

            with torch.no_grad():
                for batch_i, (batch_x, batch_y) in enumerate(val_dataloader):
                    if batch_i > max_batches_per_epoch_val:
                        break

                    batch_x = copy_data_to_device(batch_x, device)
                    batch_y = copy_data_to_device(batch_y, device)

                    pred = model(batch_x)
                    loss, metrics, vis_img = criterion(pred, batch_y)

                    mean_val_loss += float(loss)
                    mean_val_metrics = sum_dicts(mean_val_metrics, metrics)
                    if vis_img is not None and save_vis_images_path is not None and batch_i % save_vis_images_freq == 0:
                        save_image(vis_img,
                                   os.path.join(save_vis_images_path,
                                                'epoch{:04d}_iter{:06d}_val.jpg'.format(epoch_i, batch_i)),
                                   nrow=batch_y['images'].shape[0],
                                   normalize=True,
                                   range=(-1, 1))
                    val_batches_n += 1

            mean_val_loss /= val_batches_n + 1e-6
            mean_val_metrics = norm_dict(mean_val_metrics, val_batches_n)
            print('Mean validation loss', mean_val_loss, mean_val_metrics)

            if mean_val_loss < best_val_loss:
                best_epoch_i = epoch_i
                best_val_loss = mean_val_loss
                best_val_metrics = mean_val_metrics
                best_model = copy.deepcopy(model)

                print('New best model!')

                if save_models_path is not None:
                    torch.save(best_model, os.path.join(save_models_path, 'best_model.pth'))

            elif epoch_i - best_epoch_i > early_stopping_patience:
                print('Model has not improved during the last {} epochs, stopping training early'.format(
                    early_stopping_patience))
                break

            if lr_scheduler is not None:
                lr_scheduler.step(mean_val_loss)

            print()
        except KeyboardInterrupt:
            print('Interrupted by user')
            break
        except Exception as ex:
            print('Fatal error during training: {}\n{}'.format(ex, traceback.format_exc()))
            break

    return best_val_loss, best_val_metrics, best_model


def predict_with_model(model, dataset, device='cuda', batch_size=32, num_workers=0, return_labels=False):
    """
    :param model: torch.nn.Module - trained model
    :param dataset: torch.utils.data.Dataset - data to apply model
    :param device: cuda/cpu
    :param batch_size:
    :return: numpy.array dimensionality len(dataset) x *
    """
    results_by_batch = []

    device = torch.device(device)
    model.to(device)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    labels = []
    with torch.no_grad():
        import tqdm
        for batch_x, batch_y in tqdm.tqdm_notebook(dataloader, total=len(dataset)/batch_size):
            batch_x = copy_data_to_device(batch_x, device)

            if return_labels:
                labels.append(batch_y.numpy())

            batch_pred = model(batch_x)
            results_by_batch.append(batch_pred.detach().cpu().numpy())

    if return_labels:
        return np.concatenate(results_by_batch, 0), np.concatenate(labels, 0)
    else:
        return np.concatenate(results_by_batch, 0)
