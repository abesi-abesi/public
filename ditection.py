#ライブラリインポート
import random
import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib2
from PIL import Image
import shutil
from tensorflow.keras.utils import plot_model
from keras.models import load_model
import keras.preprocessing
import pickle
import keras.preprocessing.image
import datetime

tfds.disable_progress_bar()
MODELS = {
  'DenseNet121': (
    tf.keras.applications.DenseNet121,
    tf.keras.applications.densenet.preprocess_input,
    tf.keras.applications.densenet.decode_predictions),
  'DenseNet169': (
    tf.keras.applications.DenseNet169,
    tf.keras.applications.densenet.preprocess_input,
    tf.keras.applications.densenet.decode_predictions),
  'DenseNet201': (
    tf.keras.applications.DenseNet201,
    tf.keras.applications.densenet.preprocess_input,
    tf.keras.applications.densenet.decode_predictions),
  'InceptionResNetV2': (
    tf.keras.applications.InceptionResNetV2,
    tf.keras.applications.inception_resnet_v2.preprocess_input,
    tf.keras.applications.inception_resnet_v2.decode_predictions),
  'InceptionV3': (
    tf.keras.applications.InceptionV3,
    tf.keras.applications.inception_v3.preprocess_input,
    tf.keras.applications.inception_v3.decode_predictions),
  'MobileNet': (
    tf.keras.applications.MobileNet,
    tf.keras.applications.mobilenet.preprocess_input,
    tf.keras.applications.mobilenet.decode_predictions),
  'MobileNetV2': (
    tf.keras.applications.MobileNetV2,
    tf.keras.applications.mobilenet_v2.preprocess_input,
    tf.keras.applications.mobilenet_v2.decode_predictions),
  'NASNetLarge': (
    tf.keras.applications.NASNetLarge,
    tf.keras.applications.nasnet.preprocess_input,
    tf.keras.applications.densenet.decode_predictions),
  'NASNetMobile': (
    tf.keras.applications.NASNetMobile,
    tf.keras.applications.nasnet.preprocess_input,
    tf.keras.applications.nasnet.decode_predictions),
  'ResNet50': (
    tf.keras.applications.ResNet50,
    tf.keras.applications.resnet.preprocess_input,
    tf.keras.applications.resnet.decode_predictions),
  'ResNet101': (
    tf.keras.applications.ResNet101,
    tf.keras.applications.resnet.preprocess_input,
    tf.keras.applications.resnet.decode_predictions),
  'ResNet152': (
    tf.keras.applications.ResNet152,
    tf.keras.applications.resnet.preprocess_input,
    tf.keras.applications.resnet.decode_predictions),
  'ResNet50V2': (
    tf.keras.applications.ResNet50V2,
    tf.keras.applications.resnet_v2.preprocess_input,
    tf.keras.applications.resnet_v2.decode_predictions),
  'ResNet101V2': (
    tf.keras.applications.ResNet101V2,
    tf.keras.applications.resnet_v2.preprocess_input,
    tf.keras.applications.resnet_v2.decode_predictions),
  'ResNet152V2': (
    tf.keras.applications.ResNet152V2,
    tf.keras.applications.resnet_v2.preprocess_input,
    tf.keras.applications.resnet_v2.decode_predictions),
  'VGG16': (
    tf.keras.applications.VGG16,
    tf.keras.applications.vgg16.preprocess_input,
    tf.keras.applications.vgg16.decode_predictions),
  'VGG19': (
    tf.keras.applications.VGG19,
    tf.keras.applications.vgg19.preprocess_input,
    tf.keras.applications.vgg19.decode_predictions),
  'Xception': (
    tf.keras.applications.Xception,
    tf.keras.applications.xception.preprocess_input,
    tf.keras.applications.xception.decode_predictions),
}
OPTIMIZERS = {
  'Adadelta': tf.keras.optimizers.Adadelta,
  'Adagrad': tf.keras.optimizers.Adagrad,
  'Adam': tf.keras.optimizers.Adam,
  'Adamax': tf.keras.optimizers.Adamax,
  'Ftrl': tf.keras.optimizers.Ftrl,
  'Nadam': tf.keras.optimizers.Nadam,
  'RMSprop': tf.keras.optimizers.RMSprop,
  'SGD': tf.keras.optimizers.SGD,
}




#サンプルデータのダウンロード
if not pathlib2.Path('/content/drive/My Drive/AppliedSeminar2019/SampleData').exists():
  !wget http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz
  !tar -xf 101_ObjectCategories.tar.gz
  !mkdir -p '/content/drive/My Drive/AppliedSeminar2019/SampleData'
  !mv 101_ObjectCategories '/content/drive/My Drive/AppliedSeminar2019/SampleData/Caltech101'




#ハイパーパラメータの設定
# 学習用画像のディレクトリ．直下にクラス名をディレクトリ名に設定したサブディレクトリを作成する
INPUT_DIR = '/content/drive/My Drive/AppliedSeminar2019/SampleData/Caltech101'

# 学習経過のモデルの保存先
MODEL_SAVE_TO = '/content/drive/My Drive/AppliedSeminar2019/Output'

# 訓練セット，検証セット，テストセットの分割率
SPLITS = {
    'train': 0.70,
    'validation': 0.15,
    'test': 0.15,
}

# モデルの入力サイズを指定する（高さ, 幅, チャンネル数）
# 学習用画像は指定したサイズにアスペクト比を維持してリサイズされる
IMG_SHAPE = (224, 224, 3)

# ネットワークの重みの初期値を決める
#   True: ImageNetで学習済みの重みを使う
#   False: 乱数で初期化する
TRANSFER_FROM_IMAGENET = True

# モデルの重みの更新方法（フルチューニングとファインチューニング）の選択
#   True: フルチューニングを行う．モデル全体の重みを学習する
#   False: ファインチューニングを行う．最終層以外の層の重みを固定し，最終層の重みのみ学習する
ENABLE_FULL_TUNE = False

# バッチサイズ
BATCH_SIZE = 32

# 使用するモデル
MODEL_NAME = 'ResNet50V2'

# 学習するエポック数
EPOCHS = 10

# 学習中にモデルを保存するエポック数の間隔
SAVE_PERIOD = 3

# 学習アルゴリズム
OPTIMIZER_NAME = 'Adam'

# 初期学習率
INITIAL_LEARNING_RATE = 0.001



#ハイパーパラメータのエラーチェック
if IMG_SHAPE[2] == 1 and TRANSFER_FROM_IMAGENET:
  print('エラー：ImageNetで学習済みの重みを使う場合は入力チャンネル数は3にする必要があります')

if not TRANSFER_FROM_IMAGENET and not ENABLE_FULL_TUNE:
  print('警告：転移学習しない場合はフルチューニングで全層を学習しないと精度が全く出ません')





#学習データの読み込み
label_names = [item.name for item in pathlib2.Path(INPUT_DIR).iterdir() if item.is_dir()]
label_to_index = dict((name, index) for index, name in enumerate(label_names))
index_to_label =  dict((index, name) for name, index in label_to_index.items())

image_paths = [str(path) for path in pathlib2.Path(INPUT_DIR).glob('*/*')]
random.seed(0)
random.shuffle(image_paths)
seed = random.randrange(sys.maxsize)
random.seed(seed)
image_labels = [label_to_index[pathlib2.Path(path).parent.name] for path in image_paths]

dataset_size = len(image_paths)
dataset = tf.data.Dataset.from_tensor_slices((image_paths, image_labels))

def path2image(path, label):
  raw_bytes = tf.io.read_file(path)
  image = tf.zeros((1, 1, 3), tf.uint8)
  image = tf.cond(tf.strings.regex_full_match(tf.strings.lower(path), '.*\.jpg'), lambda: tf.image.decode_jpeg(raw_bytes), lambda:image)
  image = tf.cond(tf.strings.regex_full_match(tf.strings.lower(path), '.*\.jpeg'), lambda: tf.image.decode_jpeg(raw_bytes), lambda: image)
  image = tf.cond(tf.strings.regex_full_match(tf.strings.lower(path), '.*\.png'), lambda: tf.image.decode_png(raw_bytes), lambda: image)
  return image, label

def preprocess_image(path, label):
  image, label = path2image(path, label)
  image = tf.image.resize_with_pad(image, IMG_SHAPE[0], IMG_SHAPE[1])
  image = tf.slice(tf.tile(image, [1, 1, 3]), [0, 0, 0], [IMG_SHAPE[0], IMG_SHAPE[1], 3])
  image = MODELS[MODEL_NAME][1](image)
  if (IMG_SHAPE[2] == 1):
    image = tf.image.rgb_to_grayscale(image)
  return image, label

train_size = int(dataset_size * SPLITS['train'])
validation_size = int(dataset_size * SPLITS['validation'])
test_size = int(dataset_size * SPLITS['test'])

train_batches = dataset.take(train_size).shuffle(1).map(preprocess_image).repeat().batch(BATCH_SIZE)
validation_batches = dataset.skip(train_size).take(validation_size).shuffle(1).map(preprocess_image).repeat().batch(BATCH_SIZE)
test_batches = dataset.skip(train_size + validation_size).take(test_size).shuffle(1).map(preprocess_image).repeat().batch(BATCH_SIZE)

print('Labels:', len(label_names))
for index, label in index_to_label.items():
  print('  %d: %s' % (index, label))
print()
print('All Images:', len(image_paths))
print('  Train Images:', train_size)
print('  Validation Images:', validation_size)
print('  Test Images:', test_size)




#学習データをランダムに表示
index = random.randint(0, dataset_size)
for (image, label), (image2, label2) in zip(dataset.skip(index).take(1).map(path2image), dataset.skip(index).take(1).map(preprocess_image)):
  plt.figure()
  image = (image.numpy()).astype(np.uint8)
  if image.shape[2] == 1:
    image = np.squeeze(np.stack((image,) * 3, axis=-1))
  plt.imshow(image)
  plt.title(index_to_label[label2.numpy()])

  plt.figure()
  image2 = (image2.numpy()).astype(np.uint8)
  if image2.shape[2] == 1:
    image2 = np.squeeze(np.stack((image2,) * 3, axis=-1))
  plt.imshow(image2)
  plt.title(index_to_label[label2.numpy()])



#DNNモデルの構築
output_dir = pathlib2.Path(MODEL_SAVE_TO) / ('%s (%dx%dx%d) (%d classes)' % (MODEL_NAME, IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2], len(index_to_label)))
os.makedirs(str(output_dir), exist_ok=True)

def create_model():
  base_model = MODELS[MODEL_NAME][0](input_shape=IMG_SHAPE, include_top=False, weights='imagenet' if TRANSFER_FROM_IMAGENET else None)
  base_model.trainable = ENABLE_FULL_TUNE

  # base_model.summary()
  plot_model(base_model, to_file=str(output_dir / 'base-model.png'), show_shapes=True, show_layer_names=False)
  cv2.imshow('Model', cv2.imread(str(output_dir / 'base-model.png')))

  model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(len(label_names), activation='softmax'),
  ])

  model.compile(optimizer=OPTIMIZERS[OPTIMIZER_NAME](learning_rate=INITIAL_LEARNING_RATE),
                loss='sparse_categorical_crossentropy',
                metrics=["accuracy"])

  plot_model(model, to_file=str(output_dir / 'full-model.png'), show_shapes=True, show_layer_names=False)
  # base_model.summary()
  return model

model = create_model()



#学習の実行
latest = tf.train.latest_checkpoint(str(output_dir))
if latest is not None:
  model.load_weights(latest)

# steps_per_epoch = round(train_size)//BATCH_SIZE
train_steps = int(train_size/BATCH_SIZE)
validation_steps = int(validation_size/BATCH_SIZE)
test_steps = int(test_size/BATCH_SIZE)

print('Last checkpoint = %s' % latest)
initial_epoch = 0
if latest is not None:
  latest = tf.train.latest_checkpoint(str(output_dir))
  match = re.search('model.ckpt-(\\d+)$', latest)
  initial_epoch = int(match.group(1))
else:
  loss, acc = model.evaluate(train_batches, steps=train_steps)
  val_loss, val_acc = model.evaluate(validation_batches, steps=validation_steps)
  model.save_weights(str(output_dir / 'model.ckpt-00000'))
  history = {
    'loss': [loss],
    'accuracy': [acc],
    'val_loss': [val_loss],
    'val_accuracy': [val_acc],
  }
  with open(str(output_dir / 'history'), 'wb') as file_pi:
    pickle.dump(history, file_pi)

checkpoint_path = str(output_dir / 'model.ckpt-{epoch:05d}')
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq=SAVE_PERIOD * BATCH_SIZE * train_steps)

class MyCustomCallback(tf.keras.callbacks.Callback):
  def __init__(self, initial_epoch, period):
    self.epoch = initial_epoch + 1
    self.period = period
    with open(str(output_dir / 'history'), 'rb') as file_pi:
      self.history = pickle.load(file_pi)

  def on_epoch_end(self, batch, logs=None):
    self.history['loss'].append(logs['loss'])
    self.history['accuracy'].append(logs['accuracy'])
    self.history['val_loss'].append(logs['val_loss'])
    self.history['val_accuracy'].append(logs['val_accuracy'])
    if (self.epoch % self.period) == 0:
      with open(str(output_dir / 'history'), 'wb') as file_pi:
        pickle.dump(self.history, file_pi)
    self.epoch += 1

print('Initial_epoch=%d' % initial_epoch)
history = model.fit(train_batches,
                    initial_epoch=initial_epoch,
                    epochs=EPOCHS,
                    validation_data=validation_batches,
                    validation_steps=validation_steps,
                    callbacks=[cp_callback, MyCustomCallback(initial_epoch, SAVE_PERIOD)],
                    steps_per_epoch=train_steps)

loss, acc = model.evaluate(test_batches, steps=test_steps)
print('学習後のモデルの精度')
print("loss: {:.2f}".format(loss))
print("accuracy: {:.2f}".format(acc))




#学習中の精度のロスの推移をグラフに表示
history2 = None
with open(str(output_dir / 'history'), 'rb') as file_pi:
  history2 = pickle.load(file_pi)

acc = history2['accuracy']
val_acc = history2['val_accuracy']

loss = history2['loss']
val_loss = history2['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
plt.ylabel('Cross Entropy')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()




#学習後のモデルの動作確認
latest = tf.train.latest_checkpoint(str(output_dir))
model.load_weights(latest)

INPUT_IMAGE_PATH = None
# INPUT_IMAGE_PATH = '/content/drive/My Drive/AppliedSeminar2019/SampleData/Caltech101/camera/image_0001.jpg'

if INPUT_IMAGE_PATH == None:
  path, gt_index = random.choice(list(zip(image_paths, image_labels))[train_size + validation_size:])
  gt_label = index_to_label[gt_index]
else:
  path = INPUT_IMAGE_PATH
  gt_label = pathlib2.Path(INPUT_IMAGE_PATH).parent.name

image, label = path2image(path, gt_label)
plt.figure()
np_image = (image.numpy()).astype(np.uint8)
if np_image.shape[2] == 1:
  np_image = np.squeeze(np.stack((np_image,) * 3, axis=-1))
plt.imshow(np_image)
plt.title(label)

image, label = preprocess_image(path, gt_label)
plt.figure()
np_image = (image.numpy()).astype(np.uint8)
if np_image.shape[2] == 1:
  np_image = np.squeeze(np.stack((np_image,) * 3, axis=-1))
plt.imshow(np_image)
plt.title(label)

image = tf.expand_dims(image, axis=0)
preds = model.predict(image, steps=1)
pred = preds[0]

top = len(index_to_label)
top_indices = pred.argsort()[-top:][::-1]
result = [tuple([index_to_label[i], pred[i]]) for i in top_indices]
result.sort(key=lambda x: x[1], reverse=True)

print('Predicted scores')
for index, (name, confidence) in enumerate(result):
  if name == gt_label:
    print('  [%d] "%s": %f (Ground truth)' % (index + 1, name, confidence))
  else:
    print('  [%d] "%s": %f' % (index + 1, name, confidence))
