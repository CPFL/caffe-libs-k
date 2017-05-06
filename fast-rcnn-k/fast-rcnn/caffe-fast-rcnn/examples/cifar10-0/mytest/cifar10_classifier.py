import _init_paths
import sys
import caffe
from caffe.proto import caffe_pb2
import numpy

cifar_map = {
	0: "airplane",
	1: "automobile",
	2: "bird",
	3: "cat",
	4: "deer",
	5: "dog",
	6: "frog",
	7: "hourse",
	8: "ship",
	9: "truck"
}

mean_blob = caffe_pb2.BlobProto()
with open('../mean.binaryproto') as f:
    mean_blob.ParseFromString(f.read())

mean_array = numpy.asarray(mean_blob.data, dtype=numpy.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width)
)

classifier = caffe.Classifier(
    '../cifar10_quick.prototxt',
    '../cifar10_quick_iter_4000.caffemodel',
    mean=mean_array,
    raw_scale=255)

image = caffe.io.load_image(sys.argv[1])
predictions = classifier.predict([image], oversample=False)
answer = numpy.argmax(predictions)
print(predictions)
print(str(answer) + ":" + cifar_map[answer])
