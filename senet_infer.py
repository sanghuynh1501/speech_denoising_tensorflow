import json
from model import *
from data_import import *

import sys, getopt

valfolder = "dataset/valset_noisy"
modfolder = "models"

try:
    opts, args = getopt.getopt(sys.argv[1:],"hd:m:",["ifolder=,modelfolder="])
except getopt.GetoptError:
    print('Usage: python senet_infer.py -d <inputfolder> -m <modelfolder>')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('Usage: pythonsenet_infer.py -d <inputfolder> -m <modelfolder>')
        sys.exit()
    elif opt in ("-d", "--inputfolder"):
        valfolder = arg
    elif opt in ("-m", "--modelfolder"):
        modfolder = arg
print('Input folder is "' + valfolder + '/"')
print('Model folder is "' + modfolder + '/"')

if valfolder[-1] == '/':
    valfolder = valfolder[:-1]

if not os.path.exists(valfolder+'_denoised'):
    os.makedirs(valfolder+'_denoised')

# SPEECH ENHANCEMENT NETWORK
SE_LAYERS = 13 # NUMBER OF INTERNAL LAYERS
SE_CHANNELS = 64 # NUMBER OF FEATURE CHANNELS PER LAYER
SE_LOSS_LAYERS = 6 # NUMBER OF FEATURE LOSS LAYERS
SE_NORM = "NM" # TYPE OF LAYER NORMALIZATION (NM, SBN or None)

fs = 16000

# SET LOSS FUNCTIONS AND PLACEHOLDERS
with tf.variable_scope(tf.get_variable_scope()):
    input=tf.placeholder(tf.float32,shape=[None,1,None,1])
    clean=tf.placeholder(tf.float32,shape=[None,1,None,1])
        
    enhanced=senet(input, n_layers=SE_LAYERS, norm_type=SE_NORM, n_channels=SE_CHANNELS)

# LOAD DATA
valset = load_noisy_data_list(valfolder = valfolder)
valset = load_noisy_data(valset)

# BEGIN SCRIPT #########################################################################################################

# INITIALIZE GPU CONFIG
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
sess=tf.Session(config=config)

print("Config ready")

sess.run(tf.global_variables_initializer())

print("Session initialized")
for var in tf.trainable_variables():
    if var.name.startswith("se_"):
        print(var.name, "var.shape ", var.shape)

saver = tf.train.Saver([var for var in tf.trainable_variables() if var.name.startswith("se_")])
saver.restore(sess, "./%s/se_model.ckpt" % modfolder)

#####################################################################################
weights = {}
weights['weights'] = []
weight_hash = {}
layer_names = []
conv_shape = {}

idx = 0
for var in tf.trainable_variables():
    names = var.name.split("/")[:2]
    if names[0] not in weight_hash:
        weight_hash[names[0]] = {}
        weight = sess.run(var).transpose(3, 2, 0, 1)
        weight_hash[names[0]]["conv2D"] = weight.reshape(np.prod(np.array(weight.shape)),).tolist()
        layer_names.append(names[0])
        conv_shape[names[0]] = weight.shape
    else:
        weight = sess.run(var)
        if names[1] == "BatchNorm" or names[1] == "biases:0":
          weight = np.reshape(weight, (1, conv_shape[names[0]][0], 1, 1))
          weight = np.repeat(weight, 4000, 3)
          weight_hash[names[0]]["batchNorm"] = weight.reshape(np.prod(np.array(weight.shape)),).tolist()
        else:
          weight = np.reshape(weight, (1, 1, 1, 1))
          weight = np.repeat(weight, conv_shape[names[0]][0], 1)
          weight = np.repeat(weight, 4000, 3)
          weight_hash[names[0]][names[1].split(":")[0]] = weight.reshape(np.prod(np.array(weight.shape)),).tolist()
          
        
    idx += 1

idx = 0
print("total ", layer_names)
for layer_name in layer_names:
    weights['weights'].append(weight_hash[layer_name])
    idx += 1

with open('weights.json', 'w') as outfile:
    json.dump(weights, outfile)


# for id in tqdm(range(0, len(valset["innames"]))):

#     i = id # NON-RANDOMIZED ITERATION INDEX
#     inputData = valset["inaudio"][i] # LOAD DEGRADED INPUT

#     # VALIDATION ITERATION
#     output = sess.run([enhanced],
#                         feed_dict={input: inputData})
#     output = np.reshape(output, -1)
#     print("output ", output[:10])
#     wavfile.write("%s_denoised/%s" % (valfolder,valset["shortnames"][i]), fs, output)
# #
