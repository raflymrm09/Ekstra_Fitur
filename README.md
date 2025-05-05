# Ekstra Fitur # 
# Deteksi HSV #
```
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
```
# Deteksi Indeks #
```
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
```
# Deteksi Refrensi #
```
# Step 1: Install Library
!pip install matplotlib numpy opencv-python-headless --quiet

# Step 2: Import Library
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
from skimage import io
import cv2

# Step 3: Upload Gambar
print("Silakan upload gambar untuk dihitung histogram berdasarkan referensi warna:")
uploaded = files.upload()
img_path = list(uploaded.keys())[0]
image = io.imread(img_path)

# Step 4: Definisikan 36 warna referensi (RGB)
reference_colors = np.array([
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
    [0, 0, 128], [128, 0, 128], [0, 128, 128], [192, 192, 192],
    [128, 128, 128], [255, 0, 0], [0, 255, 0], [255, 255, 0],
    [0, 0, 255], [255, 0, 255], [0, 255, 255], [255, 255, 255],
    [100, 0, 0], [0, 100, 0], [0, 0, 100], [100, 100, 0],
    [100, 0, 100], [0, 100, 100], [50, 50, 50], [150, 150, 150],
    [200, 0, 0], [0, 200, 0], [0, 0, 200], [200, 200, 0],
    [200, 0, 200], [0, 200, 200], [255, 128, 0], [128, 255, 0],
    [0, 255, 128], [0, 128, 255], [128, 0, 255], [255, 0, 128]
])

# Step 5: Hitung jarak piksel ke setiap warna referensi
h, w, _ = image.shape
pixels = image.reshape(-1, 3)
distances = np.linalg.norm(pixels[:, np.newaxis] - reference_colors, axis=2)
closest_color_idx = np.argmin(distances, axis=1)

# Step 6: Hitung histogram berdasarkan indeks warna referensi
histogram, _ = np.histogram(closest_color_idx, bins=np.arange(37))

# Step 7: Tampilkan gambar dan histogram
plt.figure(figsize=(15, 5))

# Tampilkan gambar asli
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Gambar Asli')
plt.axis('off')

# Tampilkan histogram warna referensi
plt.subplot(1, 2, 2)
plt.bar(np.arange(36), histogram, color=reference_colors / 255.0)
plt.title('Histogram Berdasarkan Warna Referensi')
plt.xlabel('Indeks Warna')
plt.ylabel('Jumlah Piksel')
plt.grid(True)
plt.tight_layout()
plt.show()

# Deteksi Kematangan Buah Stroberry #
```
from google.colab import files
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Upload gambar
uploaded = files.upload()
for filename in uploaded.keys():
    image = Image.open(io.BytesIO(uploaded[filename])).convert('RGB')
    image_np = np.array(image)

# Konversi ke HSV
hsv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
hue_channel = hsv_image[:, :, 0]
sat_channel = hsv_image[:, :, 1]
val_channel = hsv_image[:, :, 2]

# Segmentasi area buah matang (merah)
lower_red1 = np.array([0, 100, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 50])
upper_red2 = np.array([180, 255, 255])
mask_matang1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
mask_matang2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
mask_matang = cv2.bitwise_or(mask_matang1, mask_matang2)

# Segmentasi setengah matang (oranye)
lower_orange = np.array([11, 100, 50])
upper_orange = np.array([30, 255, 255])
mask_setengah = cv2.inRange(hsv_image, lower_orange, upper_orange)

# Segmentasi belum matang (hijau)
lower_green = np.array([35, 80, 50])
upper_green = np.array([85, 255, 255])
mask_belum = cv2.inRange(hsv_image, lower_green, upper_green)

# Gabungkan semua mask
combined_mask = cv2.bitwise_or(mask_matang, cv2.bitwise_or(mask_setengah, mask_belum))

# Ambil HSV dari area buah
hue_fruit = hue_channel[combined_mask > 0]
sat_fruit = sat_channel[combined_mask > 0]
val_fruit = val_channel[combined_mask > 0]

# Klasifikasi berdasarkan Hue
matang_area = ((hue_fruit >= 0) & (hue_fruit <= 10)) | ((hue_fruit >= 160) & (hue_fruit <= 180))
setengah_area = (hue_fruit > 10) & (hue_fruit <= 30)
belum_area = (hue_fruit >= 35) & (hue_fruit <= 85)

# Hitung persentase
total = len(hue_fruit)
matang_pct = np.sum(matang_area) / total * 100
setengah_pct = np.sum(setengah_area) / total * 100
belum_pct = np.sum(belum_area) / total * 100

# Tampilkan hasil deteksi
print("Deteksi Kematangan Buah Stroberi (fokus pada buah):")
print(f"  Matang          : {matang_pct:.2f}%")
print(f"  Setengah Matang : {setengah_pct:.2f}%")
print(f"  Belum Matang    : {belum_pct:.2f}%")

# Buat gambar hasil masking
result_image = image_np.copy()
result_image[combined_mask == 0] = 0  # Hitamkan background

# Tampilkan gambar asli, masking, pie chart dan histogram HSV
plt.figure(figsize=(18, 10))

# Gambar asli
plt.subplot(2, 3, 1)
plt.imshow(image_np)
plt.title("Gambar Asli")
plt.axis("off")

# Gambar hasil deteksi buah
plt.subplot(2, 3, 2)
plt.imshow(result_image)
plt.title("Deteksi Area Buah Stroberi")
plt.axis("off")

# Pie chart distribusi kematangan
plt.subplot(2, 3, 3)
labels = ['Matang', 'Setengah Matang', 'Belum Matang']
sizes = [matang_pct, setengah_pct, belum_pct]
colors = ['red', 'orange', 'green']
explode = (0.05, 0.05, 0.05)
plt.pie(sizes, labels=labels, colors=colors, explode=explode,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.title("Distribusi Kematangan")

# Histogram Hue
plt.subplot(2, 3, 4)
plt.hist(hue_fruit, bins=180, range=(0, 180), color='orange', edgecolor='black')
plt.title("Histogram Hue")
plt.xlabel("Hue (0-180)")
plt.ylabel("Jumlah Piksel")

# Histogram Saturation
plt.subplot(2, 3, 5)
plt.hist(sat_fruit, bins=256, range=(0, 256), color='purple', edgecolor='black')
plt.title("Histogram Saturation")
plt.xlabel("Saturation")
plt.ylabel("Jumlah Piksel")

# Histogram Value
plt.subplot(2, 3, 6)
plt.hist(val_fruit, bins=256, range=(0, 256), color='gray', edgecolor='black')
plt.title("Histogram Value")
plt.xlabel("Value")
plt.ylabel("Jumlah Piksel")

plt.tight_layout()
plt.show()
```
