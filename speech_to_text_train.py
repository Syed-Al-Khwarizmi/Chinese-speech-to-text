import os;
import sys;
import pickle;
import librosa;
import numpy as np;
from keras.models import Model;
from keras import backend as K;
from keras.layers.embeddings import Embedding;
from keras.utils.vis_utils import plot_model;
from keras.models import Sequential, load_model;
from keras.optimizers import rmsprop, adam, adagrad, SGD;
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau;
from keras.preprocessing.text import text_to_word_sequence, one_hot, Tokenizer;
from keras.layers import Input, Dense, merge, Dropout, BatchNormalization, Activation, Conv1D, Lambda;
# Getting the current working directory
DIR=os.getcwd();

"""
The train.word.txt file contains a list of all the
wav file names, along with their corresponding Chinese
text.
Sample format of this file is:
A11_10 炮眼 打好 了 炸药 怎么 装 岳 正 才 咬 了 咬牙 倏 地 脱去 衣服 光膀子 冲进 了 水 窜 洞
A11_100 可 谁知 纹 完 后 她 一 照镜子 只见 左下 眼睑 的 线 又 粗 又 黑 与 右侧 明显 不对称
A11_102 一进门 我 被 惊呆 了 这 户 名叫 庞 吉 的 老农 是 抗美援朝 负伤 回乡 的 老兵 妻子 长年 有病 家徒四壁 一贫如洗
A11_103 走出 村子 老远 老远 我 还 回头 张望 那个 安宁 恬静 的 小院 那个 使 我 终身 难忘的 小院

Here, A11_10, A11_100, etc are the wav files which should
be present in the DIR directory.
"""
with open(DIR+"/train.word.txt") as f:
    texts=f.read().split("\n");

#print("texts: {}".format(texts[0:10]))

del texts[-1];texts=[i.split(" ") for i in texts];
"""
By now, the texts have been tokenized, and have a 2D form,
where each row erpresents wav filename as the first index,
followed by tokenized individual mandarin symbols.
First row, for instance looks like this:
['A11_0', '绿', '是', '阳春', '烟', '景', '大块', '文章', '的', '底色',
'四月', '的', '林', '峦', '更是', '绿', '得', '鲜活', '秀媚', '诗意', '盎然']
"""

#print("texts after loop: {}".format(texts[0:10]))
all_words=[];maxlen_char=0;
for i in np.arange(0,len(texts)):
    length=0;
    for j in texts[i][1:]:
        length+=len(j);
    if maxlen_char<=length:maxlen_char=length;
    for j in np.arange(1,len(texts[i])):
        all_words.append(texts[i][j]);
"""
By now, we have the entire curpus vocabulary, in the form of
a big list.
"""

tok=Tokenizer(char_level=True);tok.fit_on_texts(all_words);

print("tok: {}".format(tok))
char_index=tok.word_index;index_char=dict((char_index[i],i) for i in char_index);
char_vec=np.zeros((10000,maxlen_char),dtype=np.float32);
#char_input=[[] for _ in np.arange(0,len(texts))];
char_length=np.zeros((10000,1),dtype=np.float32);
for i in np.arange(0,len(texts)):
    j=0;
    for i1 in texts[i][1:]:
        for ele in i1:
            char_vec[i,j]=char_index[ele];j+=1;
    char_length[i]=j;

mfcc_vec=[[] for _ in np.arange(0,len(texts))];
for i in np.arange(0,len(texts)):
    try:
        wav, sr = librosa.load(DIR + "/"+texts[i][0]+".wav", mono=True);
    except FileNotFoundError:
        wav, sr = librosa.load(DIR + "/" + texts[i][0] + ".WAV", mono=True);
    b = librosa.feature.mfcc(wav, sr)
    mfcc = np.transpose(b, [1, 0]);
    mfcc_vec[i]=mfcc;
    if i%100==0:print("Completed {}".format(str(i*len(texts)**-1)));

np.save(DIR+"/mfcc_vec",mfcc_vec);
mfcc_vec_origin=np.load(DIR+"/mfcc_vec_origin.npy");
maxlen_mfcc=673;
mfcc_vec=np.zeros((10000,maxlen_mfcc,20),dtype=np.float32);
for i in np.arange(0,len(mfcc_vec_origin)):
    for j in np.arange(0,len(mfcc_vec_origin[i])):
        for k,ele in enumerate(mfcc_vec_origin[i][j]):
            mfcc_vec[i,j,k]=ele;

np.save(DIR+"/mfcc_vec",mfcc_vec);


mfcc_input=np.load(DIR+"/mfcc_vec.npy");
input_tensor=Input(shape=(mfcc_input.shape[1],mfcc_input.shape[2]));
x=Conv1D(kernel_size=1,filters=192,padding="same")(input_tensor);
x=BatchNormalization(axis=-1)(x);
x=Activation("tanh")(x);
def res_block(x,size,rate,dim=192):
    x_tanh=Conv1D(kernel_size=size,filters=dim,dilation_rate=rate,padding="same")(x);
    x_tanh=BatchNormalization(axis=-1)(x_tanh);
    x_tanh=Activation("tanh")(x_tanh);
    x_sigmoid=Conv1D(kernel_size=size,filters=dim,dilation_rate=rate,padding="same")(x);
    x_sigmoid=BatchNormalization(axis=-1)(x_sigmoid);
    x_sigmoid=Activation("sigmoid")(x_sigmoid);
    out=merge([x_tanh,x_sigmoid],mode="mul");
    out=Conv1D(kernel_size=1,filters=dim,padding="same")(out);
    out=BatchNormalization(axis=-1)(out);
    out=Activation("tanh")(out);
    x=merge([x,out],mode="sum");
    return x,out;

skip=[];
for i in np.arange(0,3):
    for r in [1,2,4,8,16]:
        x,s=res_block(x,size=7,rate=r);
        skip.append(s);

def ctc_lambda_function(args):
    y_true_input, logit, logit_length_input, y_true_length_input=args;
    return K.ctc_batch_cost(y_true_input,logit,logit_length_input,y_true_length_input);

skip_tensor=merge([s for s in skip],mode="sum");
logit=Conv1D(kernel_size=1,filters=192,padding="same")(skip_tensor);
logit=BatchNormalization(axis=-1)(logit);
logit=Activation("tanh")(logit);
logit=Conv1D(kernel_size=1,filters=len(char_index)+1,padding="same",activation="softmax")(logit);
#base_model=Model(inputs=input_tensor,outputs=logit);
logit_length_input=Input(shape=(1,));
y_true_input=Input(shape=(maxlen_char,));
y_true_length_input=Input(shape=(1,));
loss_out=Lambda(ctc_lambda_function,output_shape=(1,),name="ctc")([y_true_input,logit,logit_length_input,y_true_length_input])
model=Model(inputs=[input_tensor,logit_length_input,y_true_input,y_true_length_input],outputs=loss_out);
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred},optimizer="adam");
#plot_model(model, to_file="model.png", show_shapes=True);
early = EarlyStopping(monitor="loss", mode="min", patience=10);
lr_change = ReduceLROnPlateau(monitor="loss", factor=0.2, patience=0, min_lr=0.000)
checkpoint = ModelCheckpoint(filepath=DIR + "/listen_model.chk",
                              save_best_only=False);
opt=adam(lr=0.0003);
model.fit(x=[mfcc_input,np.ones(10000)*673,char_vec,char_length],y=np.ones(10000),callbacks=[early,lr_change,checkpoint],
          batch_size=50,epochs=1000);
