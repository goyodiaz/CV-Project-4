import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os, random

# Функция для загрузки изображения
def load_image(image_path, max_dim=512):
    img = Image.open(image_path)
    img = img.convert("RGB")  # Убедимся, что изображение в формате RGB
    img = np.array(img)
    img = tf.image.resize(img, (max_dim, max_dim))  # Изменение размера
    img = img / 255.0  # Нормализация в диапазон [0, 1]
    return img

# Функция для случайного выбора изображения из папки
def random_image_from_folder(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    random_image = random.choice(image_files)
    return os.path.join(folder_path, random_image)

# Убедитесь, что пути к папкам заданы до их использования
content_folder = '/content/drive/My Drive/datasets/impressionist/training/training'
style_folder = '/content/drive/My Drive/datasets/impressionist/validation/validation'

# Загружаем случайные изображения
content_image_path = random_image_from_folder(content_folder)
style_image_path = random_image_from_folder(style_folder)

# Загрузка изображений
content_image = load_image(content_image_path)
style_image = load_image(style_image_path)

# Визуализация выбранных изображений
pil_content_image = Image.fromarray((content_image * 255).astype(np.uint8))
pil_style_image = Image.fromarray((style_image * 255).astype(np.uint8))

st.image(pil_content_image, caption="Content Image", use_column_width=True)
st.image(pil_style_image, caption="Style Image", use_column_width=True)

# Загрузка модели (например, VGG19, без верхнего слоя)
def get_vgg_model():
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    return vgg

# Создание модели
vgg_model = get_vgg_model()

# Ваши переменные для оптимизации
target_image = tf.Variable(content_image[tf.newaxis, ...], dtype=tf.float32)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)

# Обучение и отображение
epochs = 100
for epoch in range(epochs):
    loss = train_step(vgg_model, target_image, content_image[tf.newaxis, ...], style_image[tf.newaxis, ...], optimizer)
    
    # Показать результат
    if (epoch + 1) % 10 == 0:
        st.write(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
        pil_target_image = Image.fromarray((target_image.numpy()[0] * 255).astype(np.uint8))
        st.image(pil_target_image, caption=f"Stylized Image at Epoch {epoch+1}", use_column_width=True)

# Сохранение результата
output_image = target_image.numpy()[0]
output_image = np.clip(output_image, 0.0, 1.0)
output_image = Image.fromarray((output_image * 255).astype(np.uint8))
output_image.save('./static/output_stylized_image.jpg')

# Показать результат
st.image('./static/output_stylized_image.jpg', caption="Final Stylized Image", use_column_width=True)
