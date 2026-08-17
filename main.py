# Comparative Analysis of Spatial Filtering Techniques for Noise Reduction in Digital Images
# Single-image version using Image2 only

import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity,
)


def process_image(image_path, label):
    if not os.path.exists(image_path):
        print(f"ERROR: File '{image_path}' tidak ditemukan! Lewati...")
        return None

    print(f"\nMemproses: {label} ({image_path})")

    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Gagal membaca gambar '{image_path}'")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = image_gray.shape
    max_dim = 512

    if height > max_dim or width > max_dim:
        scale = max_dim / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image_gray = cv2.resize(image_gray, (new_width, new_height))
        image_rgb = cv2.resize(image_rgb, (new_width, new_height))

    print(f"Dimensi: {image_gray.shape}")

    def add_gaussian_noise(image_data, mean=0, sigma=25):
        noise = np.random.normal(mean, sigma, image_data.shape).astype(np.uint8)
        return cv2.add(image_data, noise)

    def add_salt_pepper_noise(image_data, prob=0.05):
        output = image_data.copy()
        salt = np.random.rand(*image_data.shape) < prob / 2
        pepper = np.random.rand(*image_data.shape) < prob / 2
        output[salt] = 255
        output[pepper] = 0
        return output

    sigma_std = 25
    prob_std = 0.05

    noisy_gaussian = add_gaussian_noise(image_gray, sigma=sigma_std)
    noisy_sp = add_salt_pepper_noise(image_gray, prob=prob_std)

    print(f"Noise ditambahkan: Gaussian(sigma={sigma_std}), Salt-Pepper(p={prob_std})")

    def apply_filters(image_data):
        results = {}
        times = {}

        for name, kernel in [('mean_3x3', (3, 3)), ('mean_5x5', (5, 5)), ('mean_7x7', (7, 7))]:
            start = time.time()
            results[name] = cv2.blur(image_data, kernel)
            times[name] = time.time() - start

        for name, kernel in [('gaussian_3x3', (3, 3)), ('gaussian_5x5', (5, 5)), ('gaussian_7x7', (7, 7))]:
            start = time.time()
            results[name] = cv2.GaussianBlur(image_data, kernel, 0)
            times[name] = time.time() - start

        for name, kernel_size in [('median_3x3', 3), ('median_5x5', 5), ('median_7x7', 7)]:
            start = time.time()
            results[name] = cv2.medianBlur(image_data, kernel_size)
            times[name] = time.time() - start

        return results, times

    filtered_gaussian, times_gaussian = apply_filters(noisy_gaussian)
    filtered_sp, times_sp = apply_filters(noisy_sp)

    def evaluate_filters(original, filtered_results):
        metrics = {}
        for name, filtered in filtered_results.items():
            mse = mean_squared_error(original, filtered)
            psnr = peak_signal_noise_ratio(original, filtered, data_range=255)
            ssim = structural_similarity(original, filtered, data_range=255)

            signal_power = np.mean(original.astype(np.float64) ** 2)
            noise_power = np.mean((original.astype(np.float64) - filtered.astype(np.float64)) ** 2)
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))

            metrics[name] = {
                'MSE': mse,
                'PSNR': psnr,
                'SSIM': ssim,
                'SNR': snr,
            }
        return metrics

    metrics_gaussian = evaluate_filters(image_gray, filtered_gaussian)
    metrics_sp = evaluate_filters(image_gray, filtered_sp)

    def compute_fft(image_data):
        fourier = fftpack.fft2(image_data)
        shifted = fftpack.fftshift(fourier)
        return np.log(np.abs(shifted) + 1)

    best_gaussian_name = max(metrics_gaussian, key=lambda x: metrics_gaussian[x]['PSNR'])
    best_sp_name = max(metrics_sp, key=lambda x: metrics_sp[x]['PSNR'])

    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.dpi'] = 150

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Spider-Man {label}: Original and Noisy Images', fontsize=14, fontweight='bold')
    axes[0].imshow(image_rgb)
    axes[0].set_title('(a) Original Image')
    axes[0].axis('off')

    axes[1].imshow(noisy_gaussian, cmap='gray')
    axes[1].set_title(f'(b) Gaussian Noise (sigma={sigma_std})')
    axes[1].axis('off')

    axes[2].imshow(noisy_sp, cmap='gray')
    axes[2].set_title(f'(c) Salt-and-Pepper Noise (p={prob_std})')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(f'fig1_{label}_original_noisy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"fig1_{label}_original_noisy.png")

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f'Spider-Man {label}: Gaussian Noise Filtering Results', fontsize=14, fontweight='bold')
    filter_names = ['mean_3x3', 'mean_5x5', 'mean_7x7', 'gaussian_3x3', 'gaussian_5x5', 'gaussian_7x7', 'median_3x3', 'median_5x5', 'median_7x7']
    titles = ['Mean 3x3', 'Mean 5x5', 'Mean 7x7', 'Gaussian 3x3', 'Gaussian 5x5', 'Gaussian 7x7', 'Median 3x3', 'Median 5x5', 'Median 7x7']

    for i, (name, title) in enumerate(zip(filter_names, titles)):
        row, col = divmod(i, 3)
        axes[row, col].imshow(filtered_gaussian[name], cmap='gray')
        axes[row, col].set_title(f'{title}\nPSNR={metrics_gaussian[name]["PSNR"]:.2f}dB', fontsize=10)
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(f'fig2_{label}_gaussian_filtering.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"fig2_{label}_gaussian_filtering.png")

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f'Spider-Man {label}: Salt-and-Pepper Noise Filtering Results', fontsize=14, fontweight='bold')

    for i, (name, title) in enumerate(zip(filter_names, titles)):
        row, col = divmod(i, 3)
        axes[row, col].imshow(filtered_sp[name], cmap='gray')
        axes[row, col].set_title(f'{title}\nPSNR={metrics_sp[name]["PSNR"]:.2f}dB', fontsize=10)
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(f'fig3_{label}_sp_filtering.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"fig3_{label}_sp_filtering.png")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Spider-Man {label}: Best Filter Comparison', fontsize=14, fontweight='bold')

    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('(a) Original')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(noisy_gaussian, cmap='gray')
    axes[0, 1].set_title('(b) Gaussian Noise')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(filtered_gaussian[best_gaussian_name], cmap='gray')
    axes[0, 2].set_title(f'(c) {best_gaussian_name.replace("_", " ")}\nPSNR={metrics_gaussian[best_gaussian_name]["PSNR"]:.2f}dB', fontsize=10)
    axes[0, 2].axis('off')

    axes[0, 3].hist(image_gray.ravel(), bins=50, alpha=0.5, label='Original', density=True)
    axes[0, 3].hist(noisy_gaussian.ravel(), bins=50, alpha=0.5, label='Noisy', density=True)
    axes[0, 3].hist(filtered_gaussian[best_gaussian_name].ravel(), bins=50, alpha=0.5, label='Filtered', density=True)
    axes[0, 3].set_title('(d) Histogram Comparison')
    axes[0, 3].legend(fontsize=8)

    axes[1, 0].imshow(image_rgb)
    axes[1, 0].set_title('(e) Original')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(noisy_sp, cmap='gray')
    axes[1, 1].set_title('(f) Salt-and-Pepper Noise')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(filtered_sp[best_sp_name], cmap='gray')
    axes[1, 2].set_title(f'(g) {best_sp_name.replace("_", " ")}\nPSNR={metrics_sp[best_sp_name]["PSNR"]:.2f}dB', fontsize=10)
    axes[1, 2].axis('off')

    axes[1, 3].hist(image_gray.ravel(), bins=50, alpha=0.5, label='Original', density=True)
    axes[1, 3].hist(noisy_sp.ravel(), bins=50, alpha=0.5, label='Noisy', density=True)
    axes[1, 3].hist(filtered_sp[best_sp_name].ravel(), bins=50, alpha=0.5, label='Filtered', density=True)
    axes[1, 3].set_title('(h) Histogram Comparison')
    axes[1, 3].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f'fig4_{label}_best_filters.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"fig4_{label}_best_filters.png")

    original_spectrum = compute_fft(image_gray)
    noisy_gaussian_spectrum = compute_fft(noisy_gaussian)
    noisy_sp_spectrum = compute_fft(noisy_sp)
    filtered_gaussian_best_spectrum = compute_fft(filtered_gaussian[best_gaussian_name])
    filtered_sp_best_spectrum = compute_fft(filtered_sp[best_sp_name])

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Spider-Man {label}: Frequency Spectrum Analysis', fontsize=14, fontweight='bold')

    axes[0, 0].imshow(original_spectrum, cmap='hot')
    axes[0, 0].set_title('(a) Original FFT')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(noisy_gaussian_spectrum, cmap='hot')
    axes[0, 1].set_title('(b) Gaussian Noise FFT')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(filtered_gaussian_best_spectrum, cmap='hot')
    axes[0, 2].set_title(f'(c) {best_gaussian_name.replace("_", " ")} FFT')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(original_spectrum, cmap='hot')
    axes[1, 0].set_title('(d) Original FFT')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(noisy_sp_spectrum, cmap='hot')
    axes[1, 1].set_title('(e) Salt-and-Pepper Noise FFT')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(filtered_sp_best_spectrum, cmap='hot')
    axes[1, 2].set_title(f'(f) {best_sp_name.replace("_", " ")} FFT')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(f'fig5_{label}_fft_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"fig5_{label}_fft_analysis.png")

    kernel_sizes = [3, 5, 7, 9, 11]
    psnr_mean, psnr_gaussian, psnr_median = [], [], []

    for k in kernel_sizes:
        filtered_mean = cv2.blur(noisy_gaussian, (k, k))
        psnr_mean.append(peak_signal_noise_ratio(image_gray, filtered_mean, data_range=255))

        filtered_gauss = cv2.GaussianBlur(noisy_gaussian, (k, k), 0)
        psnr_gaussian.append(peak_signal_noise_ratio(image_gray, filtered_gauss, data_range=255))

        if k % 2 == 1:
            filtered_median = cv2.medianBlur(noisy_gaussian, k)
            psnr_median.append(peak_signal_noise_ratio(image_gray, filtered_median, data_range=255))
        else:
            psnr_median.append(None)

    plt.figure(figsize=(12, 6))
    plt.plot(kernel_sizes, psnr_mean, 'o-', label='Mean Filter', linewidth=2, markersize=8)
    plt.plot(kernel_sizes, psnr_gaussian, 's-', label='Gaussian Filter', linewidth=2, markersize=8)
    plt.plot(kernel_sizes[:len(psnr_median)], psnr_median, '^-', label='Median Filter', linewidth=2, markersize=8)
    plt.xlabel('Kernel Size', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.title(f'Spider-Man {label}: Effect of Kernel Size on Image Quality', fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(kernel_sizes)
    plt.tight_layout()
    plt.savefig(f'fig6_{label}_kernel_effect.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"fig6_{label}_kernel_effect.png")

    def print_metrics_table(metrics_dict, title):
        print(f"\n{'=' * 70}")
        print(f"EVALUASI METRIK - {title}")
        print(f"{'=' * 70}")
        print(f"{'Filter':<18} {'MSE':<12} {'PSNR(dB)':<14} {'SSIM':<12} {'SNR(dB)':<12}")
        print(f"{'-' * 70}")

        for name, values in metrics_dict.items():
            print(f"{name.replace('_', ' '):<18} {values['MSE']:<12.2f} {values['PSNR']:<14.2f} {values['SSIM']:<12.4f} {values['SNR']:<12.2f}")

        best = max(metrics_dict, key=lambda x: metrics_dict[x]['PSNR'])
        print(f"{'-' * 70}")
        print(f"Filter terbaik: {best.replace('_', ' ')} dengan PSNR = {metrics_dict[best]['PSNR']:.2f} dB")
        print(f"{'=' * 70}")

    print_metrics_table(metrics_gaussian, f"{label} - GAUSSIAN NOISE (sigma={sigma_std})")
    print_metrics_table(metrics_sp, f"{label} - SALT-AND-PEPPER NOISE (p={prob_std})")

    print(f"\nWAKTU EKSEKUSI (detik) - {label}")
    print(f"{'Filter':<18} {'Waktu (detik)':<15}")
    print(f"{'-' * 70}")
    for name, time_value in times_gaussian.items():
        print(f"{name.replace('_', ' '):<18} {time_value:<15.6f}")

    return best_gaussian_name, best_sp_name, metrics_gaussian, metrics_sp


def main():
    image_path = 'Image2.jpg'
    label = 'Image2'

    print("\n" + "=" * 70)
    print("MULAI MEMPROSES 1 GAMBAR: IMAGE2")
    print("=" * 70)

    result = process_image(image_path, label)
    if result is None:
        print("Proses dibatalkan karena file Image2.jpg tidak ditemukan.")
        return

    best_gaussian, best_sp, _, _ = result

    print("\n" + "=" * 70)
    print("SEMUA PROSES SELESAI!")
    print("=" * 70)
    print("\nGAMBAR YANG TELAH DIPRODUKSI:")
    print(f"   - {label}:")
    print(f"      fig1_{label}_original_noisy.png")
    print(f"      fig2_{label}_gaussian_filtering.png")
    print(f"      fig3_{label}_sp_filtering.png")
    print(f"      fig4_{label}_best_filters.png")
    print(f"      fig5_{label}_fft_analysis.png")
    print(f"      fig6_{label}_kernel_effect.png")

    print("\nRINGKASAN FILTER TERBAIK:")
    print(f"   - {label}: Gaussian = {best_gaussian.replace('_', ' ')}, SP = {best_sp.replace('_', ' ')}")

    print(f"\nFolder project: {os.getcwd()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
