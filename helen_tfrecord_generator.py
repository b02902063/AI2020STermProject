import cv2
import tensorflow as tf
import numpy as np

chin_selected_point = list(range(16, 26)) 
nose_selected_point = list(range(41, 58))
mouth_selected_point = list(range(57, 114))

all_img = []
all_chin_l = []
all_nose_l = []
all_mouth_l = []

for n in range(1, 2331):

    with open("./helen/label/{0}.txt".format(n), "r") as fp:
        temp = fp.readlines()
        label = temp[1:]
        img_name = "./helen/helen/" + temp[0].strip() + ".jpg"
        
    img = cv2.imread(img_name)

    def label_process(label):    
        for i in range(len(label)):
            temp = label[i].split(",")
            label[i] = [float(t.strip()) for t in temp]
        return label

    def preprocess(label, h, w, selected_point):
        selected_label = []
        for p in selected_point:
            if label[p] not in selected_label:
                selected_label.append(label[p])

        return selected_label

    label = label_process(label)
    chin_label = preprocess(label, img.shape[0], img.shape[1], chin_selected_point)
    nose_label = preprocess(label, img.shape[0], img.shape[1], nose_selected_point)
    mouth_label = preprocess(label, img.shape[0], img.shape[1], mouth_selected_point)

    result = cv2.resize(img, (200,200))

    h_scale = 200/img.shape[0]
    w_scale = 200/img.shape[1]
    for i in range(len(label)):
        label[i] = [label[i][0] * w_scale, label[i][1] * h_scale]
        
    twoDLabelChin = np.zeros(shape=(40, 40), dtype=np.float32)
    try:
        for label in chin_label:
            l = [label[0] * w_scale, label[1] * h_scale]
            #print(l, label, img.shape, h_scale, w_scale)
            ind = [int(l[0]//5), int(l[1]//5)]
            twoDLabelChin[ind[0], ind[1]] = 1.0
            
        twoDLabelNose = np.zeros(shape=(40, 40), dtype=np.float32)
        for label in nose_label:
            l = [label[0] * w_scale, label[1] * h_scale]
            ind = [int(l[0]//5), int(l[1]//5)]
            twoDLabelNose[ind[0], ind[1]] = 1.0    

        twoDLabelMouth = np.zeros(shape=(40, 40), dtype=np.float32)
        for label in mouth_label:
            l = [label[0] * w_scale, label[1] * h_scale]
            ind = [int(l[0]//5), int(l[1]//5)]
            twoDLabelMouth[ind[0], ind[1]] = 1.0
    except:
        continue

    all_img.append(result)
    all_chin_l.append(twoDLabelChin)
    all_nose_l.append(twoDLabelNose)
    all_mouth_l.append(twoDLabelMouth)
    
    #for label in chin_label:
    #    l = [label[0] * w_scale, label[1] * h_scale]
    #    result = cv2.circle(result, (int(l[0]), int(l[1])), 1, (0, 0, 255),  -1)
    #cv2.imwrite("test3.jpg", result)  
    #exit()
dataset = tf.data.Dataset.from_tensor_slices((all_img, all_chin_l, all_nose_l, all_mouth_l))
#rec_result = result.copy()
#for l in label:
#    ind = [int(l[0]//5), int(l[1]//5)]
#    rec_result = cv2.rectangle(rec_result, (ind[0]*5, ind[1]*5), (ind[0]*5+5, ind[1]*5+5), (0, 0, 255), -1)
#cv2.imwrite("test2.jpg", rec_result)

#selected_point = list(range(16, 26)) + list(range(41, 58)) + list(range(57, 114)) #Chin, Nose, Mouth
#circle_result = result.copy()
#for l in label:
#    circle_result = cv2.circle(circle_result, (int(l[0]), int(l[1])), 1, (0, 0, 255),  -1)
#cv2.imwrite("test3.jpg", circle_result)  
def serialize_example(img, chin_label, nose_label, mouth_label):

    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() 
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    """
    Creates a tf.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    feature = {
        'img': _bytes_feature(tf.io.serialize_tensor(img)),
        'chin_label': _bytes_feature(tf.io.serialize_tensor(chin_label)),
        'nose_label': _bytes_feature(tf.io.serialize_tensor(nose_label)),
        'mouth_label': _bytes_feature(tf.io.serialize_tensor(mouth_label))
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def tf_serialize_example(img, label0, label1, label2):
    tf_string = tf.py_function(
        serialize_example,
        (img, label0, label1, label2),  # pass these args to the above function.
        tf.string)      # the return type is `tf.string`.
    return tf.reshape(tf_string, ()) # The result is a scalar
serialized_features_dataset = dataset.map(tf_serialize_example)
writer = tf.data.experimental.TFRecordWriter("helen.tfrecord")
writer.write(serialized_features_dataset)