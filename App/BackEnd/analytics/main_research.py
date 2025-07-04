import random
from random import sample
import argparse
import numpy as np
import os
import pickle
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import matplotlib
import cv2
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.models import wide_resnet50_2, wide_resnet101_2, resnet18
from datasets.welddata import WeldDataset


# device setup
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print(f"✅ Using device: {device}")

def parse_args():
    parser = argparse.ArgumentParser('PaDiM')
    parser.add_argument('--data_path', type=str, default='K:\\Diploma_new\\datasets\\al5083\\weld')
    parser.add_argument('--save_path', type=str, default='./weld_result')
    parser.add_argument('--arch', type=str, choices=['resnet18', 'wide_resnet50_2','wide_resnet101_2','resnet18_fine_tuned','wide_resnet50_2_fine_tuned','wide_resnet101_2_fine_tuned'], default='wide_resnet50_2_fine_tuned')
    return parser.parse_args()


def main():
    arch_list = ['wide_resnet101_2_fine_tuned'] # 'resnet18', 'wide_resnet50_2','wide_resnet101_2','resnet18_fine_tuned',,'wide_resnet101_2_fine_tuned'
    data_path = 'K:\\Diploma_new\\datasets\\al5083\\weld'
    save_path = './weld_result'

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]

    for arch in arch_list:
        print(f"\n⚙️ Running model: {arch}")
        args = argparse.Namespace(
            data_path=data_path,
            save_path=save_path,
            arch=arch
        )

        img_roc_auc, pixel_roc_auc, fpr_img, tpr_img, fpr_pixel, tpr_pixel = run_model(args, fig_img_rocauc, fig_pixel_rocauc)

        #fig_img_rocauc.plot(fpr_img, tpr_img, label=f'{arch} img_ROCAUC: {img_roc_auc:.3f}')
        #fig_pixel_rocauc.plot(fpr_pixel, tpr_pixel, label=f'{arch} pixel_ROCAUC: {pixel_roc_auc:.3f}')

    fig_img_rocauc.title.set_text('Image ROCAUC Comparison')
    fig_img_rocauc.legend(loc="lower right")
    fig_pixel_rocauc.title.set_text('Pixel ROCAUC Comparison')
    fig_pixel_rocauc.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(os.path.join(save_path, 'roc_curve_comparison.png'), dpi=100)
    print("✅ Saved comparison ROC curves")


def run_model(args, fig_img_rocauc, fig_pixel_rocauc):
    #args = parse_args()

    # load model
    if args.arch == 'resnet18':
        model = resnet18(pretrained=True, progress=True)
        t_d = 448
        d = 100
    elif args.arch == 'wide_resnet101_2':
        model = wide_resnet101_2(pretrained=True, progress=True)
        t_d = 1792
        d = 550
    elif args.arch == 'wide_resnet50_2':
        model = wide_resnet50_2(pretrained=True, progress=True)
        t_d = 1792
        d = 550
    elif args.arch == 'resnet18_fine_tuned':
        state_dict = torch.load('resnet18_fine_tuned.pth')
        model = resnet18(pretrained=False, progress=True)
        # Удалить веса fc слоя
        del state_dict['fc.weight']
        del state_dict['fc.bias']
        # Загрузить остальные веса
        model.load_state_dict(state_dict, strict=False)
        t_d = 448
        d = 100
    elif args.arch == 'wide_resnet50_2_fine_tuned':
        state_dict = torch.load('wide_resnet50_2_fine_tuned.pth')
        model = wide_resnet50_2(pretrained=False, progress=True)
        # Удалить веса fc слоя
        del state_dict['fc.weight']
        del state_dict['fc.bias']
        # Загрузить остальные веса
        model.load_state_dict(state_dict, strict=False)
        t_d = 1792
        d = 550
    elif args.arch == 'wide_resnet101_2_fine_tuned':
        state_dict = torch.load('wide_resnet101_2_fine_tuned.pth')
        model = wide_resnet101_2(pretrained=False, progress=True)
        # Удалить веса fc слоя
        del state_dict['fc.weight']
        del state_dict['fc.bias']
        # Загрузить остальные веса
        model.load_state_dict(state_dict, strict=False)
        t_d = 1792
        d = 550
    model.to(device)
    model.eval()
    
    random.seed(1024)
    torch.manual_seed(1024)
    
    if use_cuda:
        torch.cuda.manual_seed_all(1024)

    idx = torch.tensor(sample(range(0, t_d), d))

    # set model's intermediate outputs
    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)
    
    os.makedirs(os.path.join(args.save_path, 'temp_%s' % args.arch), exist_ok=True)

    total_roc_auc = []
    total_pixel_roc_auc = []

#----------------

    class_name = args.arch

    full_train_dataset = WeldDataset(args.data_path, is_train=True)
    num_samples = int(0.35 * len(full_train_dataset))# 0,35 wide_resnet50_2, 1 resnet18
    indices = random.sample(range(len(full_train_dataset)), num_samples)
    train_dataset = Subset(full_train_dataset, indices)
    train_dataloader = DataLoader(train_dataset, batch_size=1, pin_memory=True)
    """
    # если процент
    full_test_dataset = WeldDataset(args.data_path, is_train=False)
    num_test_samples = int(0.005 * len(full_test_dataset))
    indices = random.sample(range(len(full_test_dataset)), num_test_samples)
    test_dataset = Subset(full_test_dataset, indices)
    """
    test_dataset = WeldDataset(args.data_path, is_train=False)# если чёткое количество
    test_dataloader = DataLoader(test_dataset, batch_size=1, pin_memory=True)

    train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
    test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

    # extract train set features
    train_feature_filepath = os.path.join(args.save_path, f'temp_{args.arch}', f'train_{class_name}.pkl')
    if not os.path.exists(train_feature_filepath):
        for (x, _) in tqdm(train_dataloader, f'| feature extraction | train | {class_name} |'):
            # model prediction
            with torch.no_grad():
                _ = model(x.to(device))
            # get intermediate layer outputs
            for k, v in zip(train_outputs.keys(), outputs):
                train_outputs[k].append(v.cpu().detach())
            # initialize hook outputs
            outputs = []
        for k, v in train_outputs.items():
            train_outputs[k] = torch.cat(v, 0)

        # Embedding concat
        print("Embedding concat")
        embedding_vectors = train_outputs['layer1']
        for layer_name in ['layer2', 'layer3']:
            embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name])

        # randomly select d dimension
        embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
        # calculate multivariate Gaussian distribution
        print("calculating multivariate Gaussian distribution")
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W)
        mean = torch.mean(embedding_vectors, dim=0).numpy()
        cov = np.zeros((C, C, H * W))
        I = np.identity(C)
        for i in range(H * W):
            # cov[:, :, i] = LedoitWolf().fit(embedding_vectors[:, :, i].numpy()).covariance_
            cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I
        # save learned distribution
        print("saving learned distribution")
        train_outputs = [mean, cov]
        with open(train_feature_filepath, 'wb') as f:
            pickle.dump([mean, cov], f)
    else:
        print(f'load train set feature from: {train_feature_filepath}')
        with open(train_feature_filepath, 'rb') as f:
            train_outputs = pickle.load(f)

    gt_list = []
    gt_mask_list = []
    test_imgs = []

    # extract test set features
    t0 = time.time()
    defect_names = []
    for (x, y, mask, defect_name) in tqdm(test_dataloader, '| feature extraction | test set |'):
        defect_names.extend(defect_name)
        test_imgs.extend(x.cpu().detach().numpy())
        gt_list.extend(y.cpu().detach().numpy())
        gt_mask_list.extend(mask.cpu().detach().numpy())
        # model prediction
        with torch.no_grad():
            _ = model(x.to(device))
        # get intermediate layer outputs
        for k, v in zip(test_outputs.keys(), outputs):
            test_outputs[k].append(v.cpu().detach())
        # initialize hook outputs
        outputs = []
    for k, v in test_outputs.items():
        test_outputs[k] = torch.cat(v, 0)
        
    # Embedding concat
    print("Embedding concat")
    embedding_vectors = test_outputs['layer1']
    for layer_name in ['layer2', 'layer3']:
        embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name])

    # randomly select d dimension
    embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
        
    # calculate distance matrix
    print("calculating distance matrix")
    B, C, H, W = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
    dist_list = []
    for i in range(H * W):
        mean = train_outputs[0][:, i]
        conv_inv = np.linalg.inv(train_outputs[1][:, :, i])
        dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
        dist_list.append(dist)

    dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

    # upsample
    print("upsampling")
    dist_list = torch.tensor(dist_list)
    score_map = F.interpolate(dist_list.unsqueeze(1), size=x.size(2), mode='bilinear',
                                  align_corners=False).squeeze().numpy()
        
    # apply gaussian smoothing on the score map
    print("applying gaussian smoothing on the score map")
    for i in range(score_map.shape[0]):
        score_map[i] = gaussian_filter(score_map[i], sigma=4)
        
    # Normalization
    print("Normolizing")
    max_score = score_map.max()
    min_score = score_map.min()
    scores = (score_map - min_score) / (max_score - min_score)
        
    # calculate image-level ROC AUC score
    #print("calculating image-level ROC AUC score")
    #img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
    #gt_list = np.asarray(gt_list)
    #fpr, tpr, _ = roc_curve(gt_list, img_scores)
    #img_roc_auc = roc_auc_score(gt_list, img_scores)
    #total_roc_auc.append(img_roc_auc)

    #print("🧪 gt_list unique values:", np.unique(gt_list))
    #print("🧪 img_scores shape:", img_scores.shape)

    #print('image ROCAUC: %.3f' % (img_roc_auc))
    #fig_img_rocauc.plot(fpr, tpr, label='%s img_ROCAUC: %.3f' % (class_name, img_roc_auc))
    
        
    # get optimal threshold
    print("getting optimal threshold")
    gt_mask = np.asarray(gt_mask_list)
    precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]
    threshold *= 1
    #print("threshold: " + threshold )

    if isinstance(gt_mask, np.ndarray):
        gt_mask = torch.from_numpy(gt_mask)

    #print("Ground truth mask unique values:", torch.unique(gt_mask))
    #print("Scores min/max:", scores.min(), scores.max())

    # calculate per-pixel level ROCAUC
    #print("calculating per-pixel level ROCAUC")
    #fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
    #per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
    #total_pixel_roc_auc.append(per_pixel_rocauc)
    #print('pixel ROCAUC: %.3f' % (per_pixel_rocauc))

    #fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, per_pixel_rocauc))
    save_dir = args.save_path + '/' + f'pictures_{args.arch}'
    os.makedirs(save_dir, exist_ok=True)
    #plot_fig(test_imgs, scores, gt_mask_list, threshold, save_dir, defect_names)
    plot_heatmap(test_imgs, scores, save_dir, defect_names)
    plot_segmentation(test_imgs, scores, threshold, save_dir, defect_names)

    print(f"⏱ Обработка кадров  {time.time() - t0:.2f} сек.")

#----------------
    
    print('Average ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.title.set_text('Average image ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.legend(loc="lower right")

    print('Average pixel ROCUAC: %.3f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.title.set_text('Average pixel ROCAUC: %.3f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.legend(loc="lower right")


    # image-level
    #fpr_img, tpr_img, _ = roc_curve(gt_list, img_scores)
    #img_roc_auc = roc_auc_score(gt_list, img_scores)

    # pixel-level
    fpr_pixel, tpr_pixel, _ = roc_curve(gt_mask.flatten(), scores.flatten())
    per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())

    return '''img_roc_auc''', per_pixel_rocauc, '''fpr''', '''tpr''', fpr_pixel, tpr_pixel

def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    
    return x

def plot_fig(test_img, scores, gts, threshold, save_dir, defect_names):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[2].imshow(img, cmap='gray', interpolation='none')
        ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[2].title.set_text('Predicted heat map')
        ax_img[3].imshow(mask, cmap='gray')
        ax_img[3].title.set_text('Predicted mask')
        ax_img[4].imshow(vis_img)
        ax_img[4].title.set_text('Segmentation result')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

       # Save figure
        os.makedirs(save_dir, exist_ok=True)
        defect = defect_names[i].replace(" ", "_")
        fig_img.savefig(os.path.join(save_dir, f'{i:03d}_{defect}.png'), dpi=100, bbox_inches='tight')
        plt.close()

def plot_heatmap(test_img, scores, save_dir, defect_names):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        heat_map = scores[i] * 255
        fig_img, ax_img = plt.subplots(figsize=(4, 4))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        ax_img.axes.xaxis.set_visible(False)
        ax_img.axes.yaxis.set_visible(False)
        ax = ax_img.imshow(heat_map, cmap='jet', norm=norm)
        ax_img.imshow(img, cmap='gray', interpolation='none')
        ax_img.imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        #ax_img.title.set_text('Predicted heat map')
        #ax.set_title('Predicted heat map')

        os.makedirs(save_dir, exist_ok=True)
        defect = defect_names[i].replace(" ", "_")
        fig_img.savefig(os.path.join(save_dir, f'{i:03d}_{defect}_heatmap.png'), dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close()

def plot_segmentation(test_img, scores, threshold, save_dir, defect_names):
    num = len(scores)
    
    for i in range(num):
        img = denormalization(test_img[i])
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255

        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.axis('off')
        ax.imshow(vis_img)
        #ax.set_title('Segmentation result')

        os.makedirs(save_dir, exist_ok=True)
        defect = defect_names[i].replace(" ", "_")
        fig.savefig(os.path.join(save_dir, f'{i:03d}_{defect}_segmentation.png'), dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close()

def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2, device=x.device)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z.to(x.device)



if __name__ == '__main__':
    main()
