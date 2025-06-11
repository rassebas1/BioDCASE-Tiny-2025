from keras import Model, layers
from keras.src.applications.mobilenet import _conv_block, _depthwise_conv_block
from keras.src.callbacks import History, EarlyStopping, TensorBoard
from keras.src.metrics import AUC
import tensorflow as tf

from paths import TENSORBOARD_LOGS_PATH
from config import Config


# def create_model(input_shape, n_filters_1=32, n_filters_2=64, dropout=0.02) -> Model:
#     inputs = layers.Input(shape=input_shape)
#     x = _conv_block(inputs, filters=n_filters_1, alpha=1, kernel=(10, 4), strides=(5, 2))
#     x = _depthwise_conv_block(x, pointwise_conv_filters=n_filters_1, alpha=1, block_id=1)
#     x = layers.GlobalMaxPooling2D(keepdims=True)(x)
#     x = layers.Dropout(dropout, name="dropout1")(x)
#     x = layers.Flatten()(x)
#     x = layers.Dense(2)(x)
#     outputs = layers.Softmax()(x)
#     model = Model(inputs, outputs, name="mobilenet_slimmed")
#     model.compile(
#         optimizer='adam',
#         loss='binary_crossentropy',
#         metrics=[AUC(curve='PR', name='average_precision')]
#     )
#     return model
def create_model(
    input_shape, 
    n_filters_1=16, 
    n_filters_2=32, 
    dropout=0.3, 
    l2_reg=1e-2
) -> Model:
    inputs = layers.Input(shape=input_shape)
    
    # Data augmentation
    #x = layers.RandomFlip("horizontal")(inputs)
    #x = layers.RandomTranslation(0.1, 0.1)(x)  # Time/freq shifts
    #x = layers.GaussianNoise(0.1)(x)
    #x = layers.RandomContrast(0.2)(x)
    #x = layers.RandomZoom(0.1)(x)
    
    # Original backbone with enhanced regularization
    x = _conv_block(inputs, filters=n_filters_1, alpha=1, kernel=(10, 4), strides=(5, 2))
    x = _depthwise_conv_block(x, pointwise_conv_filters=n_filters_1, alpha=1, block_id=1)
    x = layers.SpatialDropout2D(0.2)(x)
    # Block 2: Intermediate features (NEW)
    x = _depthwise_conv_block(x, pointwise_conv_filters=n_filters_2, alpha=1, block_id=2)
    x = layers.SpatialDropout2D(0.2)(x)
    
    # Block 3: Higher-level features (NEW)  
    x = _depthwise_conv_block(x, pointwise_conv_filters=n_filters_2, alpha=1, block_id=3)
    x = layers.GlobalMaxPooling2D()(x)
    x = layers.Dropout(dropout, name="dropout1")(x)
    
    # Enhanced classifier head
    x = layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout/2)(x)
    
    
    outputs = layers.Dense(2, activation='softmax')(x)
    
    model = Model(inputs, outputs, name="mobilenet_slimmed_v2")
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-3,
        weight_decay=1e-4
    )
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[AUC(curve='PR', name='average_precision')]
    )
    return model

# def train_model(model: Model, train_ds, valid_ds, config: Config, class_weight) -> Model:
#     tr_cfg = config.model_training
#     train_ds = train_ds.cache().shuffle(tr_cfg.shuffle_buff_n).prefetch(tf.data.AUTOTUNE)
#     valid_ds = valid_ds.cache().prefetch(tf.data.AUTOTUNE)
#     model.fit(
#         train_ds,
#         validation_data=valid_ds,
#         epochs=tr_cfg.n_epochs,
#         class_weight=class_weight,
#         callbacks=[
#             EarlyStopping(
#                 patience=tr_cfg.early_stopping.patience,
#             ),
#             TensorBoard(TENSORBOARD_LOGS_PATH, update_freq=1)
#         ]
#     )
#     return model

def train_model(model: Model, train_ds, valid_ds, config: Config, class_weight) -> Model:
    tr_cfg = config.model_training
    train_ds = train_ds.cache().shuffle(
        tr_cfg.shuffle_buff_n, 
        reshuffle_each_iteration=True
    ).prefetch(tf.data.AUTOTUNE)
    
    valid_ds = valid_ds.cache().prefetch(tf.data.AUTOTUNE)
    
    model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=tr_cfg.n_epochs,
        class_weight=class_weight,
        callbacks=[
            EarlyStopping(
                monitor='val_average_precision',
                patience=tr_cfg.early_stopping.patience,
                mode='max',
                restore_best_weights=True
            ),
            TensorBoard(
                TENSORBOARD_LOGS_PATH,
                update_freq='batch',
                profile_batch=0
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                verbose=1
            )
        ]
    )
    return model