# Deteksi HSV #
# Step 1: Upload gambar
from google.colab import files
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt
import cv2

uploaded = files.upload()

# Step 2: Baca gambar
for filename in uploaded.keys():
    image = Image.open(io.BytesIO(uploaded[filename]))
    image_np = np.array(image)

# Step 3: Konversi RGB ke HSV menggunakan OpenCV
hsv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
hue_channel = hsv_image[:, :, 0]

# Step 4: Hitung histogram Hue
hist = cv2.calcHist([hue_channel], [0], None, [180], [0, 180])

# Step 5: Tampilkan gambar asli dan histogram hue dalam satu baris
plt.figure(figsize=(12, 5))

# Gambar asli
plt.subplot(1, 2, 1)
plt.imshow(image_np)
plt.axis('off')
plt.title('Gambar Asli')

# Histogram hue
plt.subplot(1, 2, 2)
plt.plot(hist, color='orange')
plt.title('Histogram Hue')
plt.xlabel('Hue value (0-179)')
plt.ylabel('Jumlah piksel')
plt.grid(True)

plt.tight_layout()
plt.show()

# Deteksi Indeks #
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from google.colab import files
from PIL import Image
import io

# Fungsi untuk upload gambar
uploaded = files.upload()

# Ambil nama file yang diupload
for filename in uploaded.keys():
    print(f"Gambar yang diupload: {filename}")
    # Baca gambar menggunakan PIL dan konversi ke format OpenCV
    image_pil = Image.open(io.BytesIO(uploaded[filename]))
    image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

# Cek apakah gambar berhasil dibaca
if image is None:
    raise ValueError("Gambar tidak ditemukan atau gagal dibaca.")

# Konversi BGR ke RGB untuk ditampilkan
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Hitung histogram untuk masing-masing channel
hist_r = cv2.calcHist([image_rgb], [0], None, [256], [0, 256])
hist_g = cv2.calcHist([image_rgb], [1], None, [256], [0, 256])
hist_b = cv2.calcHist([image_rgb], [2], None, [256], [0, 256])

# Tampilkan gambar dan histogram
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.imshow(image_rgb)
plt.title("Gambar Asli")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.plot(hist_r, color='red')
plt.title("Histogram Merah")
plt.xlim([0, 256])

plt.subplot(2, 2, 3)
plt.plot(hist_g, color='green')
plt.title("Histogram Hijau")
plt.xlim([0, 256])

plt.subplot(2, 2, 4)
plt.plot(hist_b, color='blue')
plt.title("Histogram Biru")
plt.xlim([0, 256])

plt.tight_layout()
plt.show()
