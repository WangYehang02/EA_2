



def _block_CNN_1(filters, ker, drop_rate, activation, padding, inpC): 
    ' Returns CNN residual blocks '
    prev = inpC
    x = BatchNormalization()(prev) 
    x = Activation(activation)(x) 
    x = SpatialDropout1D(drop_rate)(x, training=True)
    x = SeparableConv1D(filters, ker, padding=padding)(x) 
    
    x = BatchNormalization()(x) 
    x = Activation(activation)(x) 
    x = SpatialDropout1D(drop_rate)(x, training=True)
    x = SeparableConv1D(filters, ker, padding=padding)(x)
    res_out = add([prev, x])
    
    return res_out 


def _transformer(drop_rate, width, name, inpC):

    x = inpC

 
    attention_layer = SeqSelfAttention(units=32,  # 你可以设为 filters，也可以固定为 32
                                       return_attention=True,
                                       attention_width=width,
                                       name=name)

    att_layer, weight = attention_layer(x)

    # 残差连接 + LayerNorm
    att_layer2 = Add()([x, att_layer])
    norm_layer = LayerNormalization()(att_layer2)

    # 前馈网络层（你之前定义了 FeedForward 类，这里保留）
    FF = FeedForward(units=128, dropout_rate=drop_rate)(norm_layer)
    FF_add = Add()([norm_layer, FF])
    norm_out = LayerNormalization()(FF_add)

    return norm_out, weight



def se_block(inputs, ratio=8):
    filters = K.int_shape(inputs)[-1]
    se = GlobalAveragePooling1D()(inputs)
    se = Dense(filters // ratio, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    return Multiply()([inputs, se])

def tcn_block_ms_se(x, filters, kernel_size, drop_rate, activation='relu', padding='causal',
                    kernel_regularizer=None, bias_regularizer=None):
    shortcut = x
    convs = []
    for d_rate in [1, 2, 4]:
        conv = Conv1D(filters=filters,
                      kernel_size=kernel_size,
                      dilation_rate=d_rate,
                      padding=padding,
                      activation=None,
                      kernel_regularizer=kernel_regularizer,
                      bias_regularizer=bias_regularizer)(x)
        convs.append(conv)

    x_concat = Concatenate()(convs)
    x_fused = Conv1D(filters, kernel_size=1, padding='same')(x_concat)
    x_norm = BatchNormalization()(x_fused)
    x_drop = SpatialDropout1D(drop_rate)(x_norm, training=True)
    x_se = se_block(x_drop)

    if K.int_shape(shortcut)[-1] != K.int_shape(x_se)[-1]:
        shortcut = Conv1D(filters, kernel_size=1, padding='same',
                          kernel_regularizer=kernel_regularizer,
                          bias_regularizer=bias_regularizer)(shortcut)

    out = Add()([x_se, shortcut])
    out = Activation(activation)(out)
    return out


def _block_TCN(filters, ker, drop_rate, activation, padding, dilation_rate, ker_regul, bias_regul, inpC):
    return tcn_block_ms_se(inpC,
                           filters=filters,
                           kernel_size=ker,
                           drop_rate=drop_rate,
                           activation=activation,
                           padding=padding,
                           kernel_regularizer=ker_regul,
                           bias_regularizer=bias_regul)


def picker_tcn_branch(encoded, filters, drop_rate, activation, padding, kernel_regularizer, bias_regularizer, name):
    x = tcn_block_ms_se(encoded,
                        filters=filters,
                        kernel_size=3,
                        drop_rate=drop_rate,
                        activation=activation,
                        padding=padding,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer)

    attn_out, _ = SeqSelfAttention(return_attention=True,
                                   attention_width=3,
                                   name=name)(x)
    return attn_out

def _encoder(filter_number, filter_size, depth, drop_rate, ker_regul, bias_regul, activation, padding, inpC):
    ' Returns the encoder that is a combination of residual blocks and maxpooling.'        
    e = inpC
    for dp in range(depth):
        e = Conv1D(filter_number[dp], 
                   filter_size[dp], 
                   padding = padding, 
                   activation = activation,
                   kernel_regularizer = ker_regul,
                   bias_regularizer = bias_regul,
                   )(e)             
        e = MaxPooling1D(2, padding = padding)(e)            
    return(e) 


def _decoder(filter_number, filter_size, depth, drop_rate, ker_regul, bias_regul, activation, padding, inpC):
    ' Returns the dencoder that is a combination of residual blocks and upsampling. '           
    d = inpC
    for dp in range(depth):        
        d = UpSampling1D(2)(d) 
        if dp == 3:
            d = Cropping1D(cropping=(1, 1))(d)           
        d = Conv1D(filter_number[dp], 
                   filter_size[dp], 
                   padding = padding, 
                   activation = activation,
                   kernel_regularizer = ker_regul,
                   bias_regularizer = bias_regul,
                   )(d)        
    return(d)  
 


def _lr_schedule(epoch):
    ' Learning rate is scheduled to be reduced after 40, 60, 80, 90 epochs.'
    
    lr = 1e-3
    if epoch > 90:
        lr *= 0.5e-3
    elif epoch > 60:
        lr *= 1e-3
    elif epoch > 40:
        lr *= 1e-2
    elif epoch > 20:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr



class cred2():
    def __init__(self,
                 nb_filters=[8, 16, 16, 32, 32, 96, 96, 128],
                 kernel_size=[11, 9, 7, 7, 5, 5, 3, 3],
                 padding='same',
                 activationf='relu',
                 endcoder_depth=7,
                 decoder_depth=7,
                 cnn_blocks=5,
                 tcn_blocks=3,
                 drop_rate=0.1,
                 loss_weights=[0.2, 0.3, 0.5],
                 loss_types=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],                                 
                 kernel_regularizer=keras.regularizers.l1(1e-4),
                 bias_regularizer=keras.regularizers.l1(1e-4),
                 multi_gpu=False, 
                 gpu_number=4):

        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.padding = padding
        self.activationf = activationf
        self.endcoder_depth = endcoder_depth
        self.decoder_depth = decoder_depth
        self.cnn_blocks = cnn_blocks
        self.tcn_blocks = tcn_blocks
        self.drop_rate = drop_rate
        self.loss_weights = loss_weights
        self.loss_types = loss_types
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.multi_gpu = multi_gpu
        self.gpu_number = gpu_number

    def __call__(self, inp):
        x = inp
        x = _encoder(self.nb_filters, self.kernel_size, self.endcoder_depth, self.drop_rate,
                     self.kernel_regularizer, self.bias_regularizer, self.activationf, self.padding, x)

        for cb in range(self.cnn_blocks):
            kernel_size = 3 if cb <= 2 else 2
            x = _block_CNN_1(self.nb_filters[6], kernel_size, self.drop_rate, self.activationf, self.padding, x)

        for tb in range(self.tcn_blocks):
            x = _block_TCN(self.nb_filters[1], 3, self.drop_rate, self.activationf, 'causal',
                           dilation_rate=2 ** tb,
                           ker_regul=self.kernel_regularizer,
                           bias_regul=self.bias_regularizer,
                           inpC=x)

        x, _ = _transformer(self.drop_rate, None, 'attentionD0', x)
        encoded, _ = _transformer(self.drop_rate, None, 'attentionD', x)

        decoder_D = _decoder(list(reversed(self.nb_filters)), list(reversed(self.kernel_size)),
                             self.decoder_depth, self.drop_rate,
                             self.kernel_regularizer, self.bias_regularizer,
                             self.activationf, self.padding, encoded)

        d = Conv1D(1, 11, padding=self.padding, activation='sigmoid', name='detector')(decoder_D)

                # P波分支
        norm_layerP = picker_tcn_branch(encoded,
                                        filters=self.nb_filters[1],
                                        drop_rate=self.drop_rate,
                                        activation=self.activationf,
                                        padding='causal',
                                        kernel_regularizer=self.kernel_regularizer,
                                        bias_regularizer=self.bias_regularizer,
                                        name='attentionP')

        decoder_P = _decoder(list(reversed(self.nb_filters)), list(reversed(self.kernel_size)),
                             self.decoder_depth, self.drop_rate,
                             self.kernel_regularizer, self.bias_regularizer,
                             self.activationf, self.padding, norm_layerP)

        P = Conv1D(1, 11, padding=self.padding, activation='sigmoid', name='picker_P')(decoder_P)

        # S波分支
        norm_layerS = picker_tcn_branch(encoded,
                                        filters=self.nb_filters[1],
                                        drop_rate=self.drop_rate,
                                        activation=self.activationf,
                                        padding='causal',
                                        kernel_regularizer=self.kernel_regularizer,
                                        bias_regularizer=self.bias_regularizer,
                                        name='attentionS')

        decoder_S = _decoder(list(reversed(self.nb_filters)), list(reversed(self.kernel_size)),
                             self.decoder_depth, self.drop_rate,
                             self.kernel_regularizer, self.bias_regularizer,
                             self.activationf, self.padding, norm_layerS)

        S = Conv1D(1, 11, padding=self.padding, activation='sigmoid', name='picker_S')(decoder_S)


        model = Model(inputs=inp, outputs=[d, P, S])

        def standard_deviation(y_true, y_pred):
            return K.std(y_pred - y_true)

        model.compile(loss=self.loss_types, loss_weights=self.loss_weights,
                      optimizer=Adam(learning_rate=_lr_schedule(0)),
                      metrics={
                          'detector': [f1],
                          'picker_P': [f1, Precision(name='precision_P'), Recall(name='recall_P'), MeanAbsoluteError(name='mae_P'), standard_deviation],
                          'picker_S': [f1, Precision(name='precision_S'), Recall(name='recall_S'), MeanAbsoluteError(name='mae_S'), standard_deviation],
                      })

        return model



