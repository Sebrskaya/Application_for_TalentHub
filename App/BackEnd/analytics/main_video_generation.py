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


def main():
    start_time = time.time()
    fps = 24
    arch_list = ['resnet18_fine_tuned'] # 'resnet18', 'wide_resnet50_2','wide_resnet101_2','resnet18_fine_tuned',,'wide_resnet101_2_fine_tuned'
    data_path = 'K:\MonSvarDiploma\App\BackEnd\\analytics\datasets\weld'
    save_path = './weld_result'
    video_path = "K:\MonSvarDiploma\App\BackEnd\\analytics\datasets\Video\good_train_weld.mp4"
    temp_frames_dir = "K:\MonSvarDiploma\App\BackEnd\\analytics\datasets\\weld\\test\\video_frames"
    temp_output_frames_dir = "K:\MonSvarDiploma\App\BackEnd\\analytics\weld_result\pictures_resnet18_fine_tuned"
    # 1. Извлекаем кадры
    t0 = time.time()
    extract_frames_from_video(video_path, temp_frames_dir, target_fps= fps)
    print(f"⏱ Извлечение кадров заняло {time.time() - t0:.2f} сек.")


    t0 = time.time()
    for arch in arch_list:
        print(f"\n⚙️ Running model: {arch}")
        args = argparse.Namespace(
            data_path=data_path,
            save_path=save_path,
            arch=arch
        )
        t0 = time.time()
        run_model(args)
        print(f"⏱ Общая обработка кадров конкретной модели  {time.time() - t0:.2f} сек.")
    print(f"⏱ Общая обработка кадров всех моделей {time.time() - t0:.2f} сек.")
    # Сохраняем heatmap видео
    
    t0 = time.time()
    save_video_from_frames(
        frames_dir=temp_output_frames_dir,
        output_path='heatmap_video.mp4',
        pattern='_heatmap.png',
        fps=fps
    )
    print(f"⏱ Сохранение heatmap.mp4  {time.time() - t0:.2f} сек.")

    # Сохраняем segmentation видео
    t0 = time.time()
    save_video_from_frames(
        frames_dir=temp_output_frames_dir,
        output_path='segmentation_video.mp4',
        pattern='_segmentation.png',
        fps=fps
    )
    print(f"⏱ Сохранение heatmap.mp4  {time.time() - t0:.2f} сек.")

    print("✅ Видео созданы")
    end_time = time.time()
    print(f"⏱ Время выполнения программы: {end_time - start_time:.2f} сек.")


def extract_frames_from_video(video_path, output_dir, target_fps=24):
    """
    Извлекает кадры из видео с заданной частотой (fps) и сохраняет их в PNG-формате.
    
    :param video_path: Путь к видеофайлу.
    :param output_dir: Папка для сохранения кадров.
    :param target_fps: Целевая частота кадров для извлечения (по умолчанию 24 fps).
    :return: Список путей к сохранённым кадрам.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"❌ Не удалось открыть видео: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(video_fps / target_fps))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"🎞 Исходный FPS: {video_fps}, Целевой FPS: {target_fps}, Интервал: {frame_interval}")

    frame_paths = []
    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved_idx:05d}.png")
            cv2.imwrite(frame_filename, frame)
            frame_paths.append(frame_filename)
            saved_idx += 1

        frame_idx += 1

    cap.release()
    print(f"✅ Сохранено {saved_idx} кадров (из {total_frames}) с {target_fps} FPS")
    return frame_paths

def run_model(args):

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
        state_dict = torch.load('App\BackEnd\\analytics\\resnet18_fine_tuned.pth', map_location=device) 
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
    global outputs
    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)
        
    os.makedirs(os.path.join(args.save_path, 'temp_%s' % args.arch), exist_ok=True)

#----------------

    class_name = args.arch

    full_train_dataset = WeldDataset(args.data_path, is_train=True)
    num_samples = int(1 * len(full_train_dataset))# 0,35 wide_resnet50_2, 1 resnet18
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
    test_dataset = WeldDataset(args.data_path, is_train=False)
    total_samples = len(test_dataset)
    chunk_size = 700

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
    defect_names = []
    t0 = time.time()
    for chunk_start in range(0, total_samples, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_samples)
        print(f"\n📦 Обрабатываем изображения с {chunk_start} по {chunk_end - 1}")

        test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
        gt_list = []
        gt_mask_list = []
        test_imgs = []
        defect_names = []

        for i in tqdm(range(chunk_start, chunk_end), desc="| feature extraction | test set |"):
            x, y, mask, defect_name = test_dataset[i]

            defect_names.append(defect_name)
            test_imgs.append(x.cpu().detach().numpy())
            gt_list.append(y)
            gt_mask_list.append(mask.cpu().detach().numpy())

            with torch.no_grad():
                _ = model(x.unsqueeze(0).to(device))  # Add batch dim
            for k, v in zip(test_outputs.keys(), outputs):
                test_outputs[k].append(v.cpu().detach())
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
            
        # get optimal threshold
        print("getting optimal threshold")

        threshold = 0.7

        #if isinstance(gt_mask, np.ndarray):
        #    gt_mask = torch.from_numpy(gt_mask)

        #print("Ground truth mask unique values:", torch.unique(gt_mask))
        #print("Scores min/max:", scores.min(), scores.max())

        save_dir = args.save_path + '/' + f'pictures_{args.arch}'
        os.makedirs(save_dir, exist_ok=True)
        plot_heatmap(test_imgs, scores, save_dir, defect_names, chunk_start)
        plot_segmentation(test_imgs, scores, threshold, save_dir, defect_names, chunk_start)

        print(f"⏱ Обработка chunck.mp4  {time.time() - t0:.2f} сек.")

#----------------
    


    # pixel-level
    #fpr_pixel, tpr_pixel, _ = roc_curve(gt_mask.flatten(), scores.flatten())
    #per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())

    #return per_pixel_rocauc, fpr_pixel, tpr_pixel


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    
    return x

def plot_heatmap(test_img, scores, save_dir, defect_names, chunk_start):
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

        os.makedirs(save_dir, exist_ok=True)
        defect = defect_names[i].replace(" ", "_")
        global_idx = chunk_start + i
        fig_img.savefig(os.path.join(save_dir, f'{global_idx:05d}_{defect}_heatmap.png'), dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close()

def plot_segmentation(test_img, scores, threshold, save_dir, defect_names, chunk_start):
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

        os.makedirs(save_dir, exist_ok=True)
        defect = defect_names[i].replace(" ", "_")
        global_idx = chunk_start + i
        fig.savefig(os.path.join(save_dir, f'{global_idx:05d}_{defect}_segmentation.png'), dpi=100, bbox_inches='tight', pad_inches=0)
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

def save_video_from_frames(frames_dir, output_path, pattern, fps=24):
    """
    Создаёт видео из PNG-кадров в папке `frames_dir`, отбирая по `pattern`.
    
    :param frames_dir: путь к папке с кадрами
    :param output_path: путь к выходному видео
    :param pattern: подстрока, по которой фильтруются кадры (например, '_heatmap.png')
    :param fps: частота кадров
    """
    frame_files = sorted([
        os.path.join(frames_dir, f)
        for f in os.listdir(frames_dir)
        if f.endswith(pattern)
    ])
    
    if not frame_files:
        print(f"❌ Нет кадров с шаблоном '{pattern}' в {frames_dir}")
        return

    frame = cv2.imread(frame_files[0])
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for fpath in frame_files:
        frame = cv2.imread(fpath)
        video_writer.write(frame)

    video_writer.release()
    print(f"🎞 Видео сохранено: {output_path}")


if __name__ == '__main__':
    main()
