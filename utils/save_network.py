import os
import torch
def save_network(network, epoch_label, save_dir):
    save_filename = 'net_%s.pth'% epoch_label
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_filename)
    torch.save(network.cuda().state_dict(), save_path)