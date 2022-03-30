import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import cv2
import sys
import random
import os
import argparse
from alexnet import load_dataset, default_model_name, default_model_dir, preprocess_image
tf.compat.v1.disable_eager_execution()

def load_model(path):
    """loads a trained model
    """
    return tf.keras.models.load_model(path)


def normalize_image(x):
    """
    Same normalization as in:
    https://github.com/fchollet/tf.keras/blob/master/examples/conv_filter_visualization.py
    """
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if tf.keras.backend.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def guided_backprop(model, image, target_layer, path):
    """modifies the model activation functions with a new gradient
    This method (and related functions) is a modified variant of the implementation seen here:
        https://github.com/jacobgil/tf.keras-grad-cam
    """

    def compile_saliency_function(model, target_layer):
        """
        """
        model_input = model.input
        layer_output = model.get_layer(target_layer).output
        max_output = tf.keras.backend.max(layer_output, axis=3)
        saliency = tf.keras.backend.gradients(tf.keras.backend.sum(max_output), model_input)[0]
        return tf.keras.backend.function([model_input, tf.keras.backend.learning_phase()], [saliency])

    def modify_backprop(model, gradient_name):
        """recreates the model in which the guided back-prop gradient function overrides
        the usual relu activation functions
        :param model: the model to be modified
        :param gradient_name: name of the custom gradient function registered in tensorflow
        :return: a new model with modified ReLU gradients
        """
        g = tf.get_default_graph()
        with g.gradient_override_map({'Relu': gradient_name}):

            # get layers that have an activation
            layer_dict = [layer for layer in model.layers[1:] if hasattr(layer, 'activation')]

            # replace relu activation
            for layer in layer_dict:
                if layer.activation == tf.keras.activations.relu:
                    layer.activation = tf.nn.relu

            # re-instantiate a new model with the tensorflow override
            new_model = load_model(path)
        return new_model

    # register the guided back-prop gradient function in tensorflow
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype  # probably float
            # clever way to mask the gradient if either op.inputs or grad is negative
            return grad * tf.cast(grad > 0., dtype) * tf.cast(op.inputs[0] > 0., dtype)

    # do guided backprop and create the saliency map
    guided_model = modify_backprop(model, 'GuidedBackProp')
    saliency_fn = compile_saliency_function(guided_model, target_layer)
    return saliency_fn([image, 0])[0]


def grad_cam(model, image, category_index, layer_name, num_class=100, height=224, width=224):
    """produces a heatmap representing the pixels that most (positively) affected the model's
    classification decision for the image
    This method is a modified variant of the grad_cam function found here:
        https://github.com/totti0223/gradcamplusplus
    :param model:
    :param image: numpy array of three dimensions
    :param category_index: classification category of the image
    :param layer_name: name of the conv+activation layer to analyze
    :return:
    """
    # all_grads = tf.keras.backend.gradients(model.output[0], conv_output)[0]
    
    y_c = model.output[0, category_index]  # prediction tensor for class
    conv_output = model.get_layer(layer_name).output  # tensor of the output of the last conv layer
    
    grads = tf.keras.backend.gradients(y_c, conv_output)[0]  # output gradients tensor
    gradient_function = tf.keras.backend.function([model.input],
                                               [conv_output, grads])  # computes output and gradient tensors

    # gradient_function = tf.keras.backend.function([model.input],
    #                                            [conv_output, all_grads])  # computes output and gradient tensors
    output, grads_val = [], []
    for i in range(len(image)):
        image_tmp = np.expand_dims(image[i], axis=0)
        output_tmp, grads_val_tmp = gradient_function([image_tmp])  # get gradient
        output.append(output_tmp[0])
        grads_val.append(grads_val_tmp[0])
    # output, grads_val = output[0, ...], grads_val[0, ...]
    output, grads_val = np.array(output), np.array(grads_val)
    weights = np.mean(grads_val, axis=(1, 2))  # compute weights as mean from gradient
    cam = np.einsum('ijkl,il -> ijk', output, weights)  # compute activations from weights
    cam = np.maximum(cam, 0)       # do ReLU operation

    # c has been computed
    # process cam map into domain specific representation
    cam_final = np.zeros_like(image[:,:,:,0], dtype="float32")
    for i in range(len(cam_final)):
        cam_final[i] = cv2.resize(cam[i], (height, width))  # resize to image size
    return np.einsum('ijk,i -> ijk', cam_final, 1./(np.max(cam_final, axis=(1,2)) + 1e-5))


def overlay_heatmap(image, heatmap):
    """merges an image and corresponding heatmap
    """
    # Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    # colorize the heatmap and merge into the image
    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam)


def parse_arguments():
    """parse command line input
    :return: dictionary of arguments keywords and values
    """
    parser = argparse.ArgumentParser(description="Construct and train an alexnet model.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p',
                        default=os.path.join(default_model_dir, default_model_name),
                        metavar='<path_to_model>',
                        help='The path to the alexnet model to which gradcam should be applied.')
    parser.add_argument('-o',
                        default="output",
                        metavar='<output_directory>',
                        help='The directory to which all visualization images will be saved.')
    return vars(parser.parse_args())

def gradcam_heatmap(model, images, layer_name = "conv2d_4"):
    heatmaps = np.zeros_like(images[:,:,:,0], dtype="float32")
    # make a prediction
    predictions = model.predict(images)
    predicted_class = np.argmax(predictions, axis=1)

    for k in set(predicted_class):
        # apply grad-cam
        print('grad-cam; class %d' %k)
        ind_tmp = np.where(predicted_class==k)[0]
        images_tmp = images[ind_tmp]
        heatmaps[ind_tmp] = grad_cam(model, images_tmp, k, layer_name)
        # cv2.imwrite(os.path.join(output, str(index) + "_gradcam.jpg"), overlay_heatmap(img, heatmap))
    return heatmaps

def main():
    args = parse_arguments()
    path = args['p']
    output = os.path.join(os.getcwd(), args['o'])
    if not os.path.isdir(output):
        os.makedirs(output)
    model = load_model(path)

    # load target image
    (x_train, y_train), (x_test, y_test) = load_dataset()
    x_test = np.array([preprocess_image(x_test[i], 224, 224) for i in range(len(x_test))])  # (224, 224, 3)
    # layer_name = "conv2d_4"  # convolutional layer on which to perform analysis
    # predictions = model.predict(x_test)
    # predicted_class = np.argmax(predictions, axis=1)
    heatmaps = gradcam_heatmap(model, x_test)
    np.save('heatmap.npy', heatmaps)

# def main():
#     """load an alexnet model, make prediction using a randomly selected image from the testing dataset,
#     visualize the NN activations using cam and guided-backprop
#     """
#     # parse arguments

#     # load model
#     args = parse_arguments()
#     path = args['p']
#     output = os.path.join(os.getcwd(), args['o'])
#     if not os.path.isdir(output):
#         os.makedirs(output)

#     model = load_model(path)

#     # load target image
#     (x_train, y_train), (x_test, y_test) = load_dataset()
#     index = random.randint(0, len(x_test))
#     img = np.array([preprocess_image(x_test[index], 224, 224)])  # (224, 224, 3)
#     # cv2.imwrite(os.path.join(path, str(index) + ".jpg"), img[0])

#     # make a prediction
#     predictions = model.predict(img)
#     predicted_class = np.argmax(predictions, axis=1)
#     print("Supplied image was classified as [%u] by the model." % predicted_class)
#     print("True classification for the image is [%u]." % y_test[index])

#     # apply grad-cam
#     layer_name = "conv2d_4"  # convolutional layer on which to perform analysis
#     heatmap = grad_cam(model, img, predicted_class[0], layer_name)
#     cv2.imwrite(os.path.join(output, str(index) + "_gradcam.jpg"), overlay_heatmap(img, heatmap))

#     # produce saliency map using guided backprop
#     saliency = guided_backprop(model, img, layer_name, path)
#     cv2.imwrite(os.path.join(output, str(index) + "_saliency.jpg"), normalize_image(saliency))

#     # combine saliency map with heatmap
#     guided_gradcam = saliency * heatmap[..., np.newaxis]
#     cv2.imwrite(os.path.join(output, str(index) + "_guided-gradcam.jpg"), normalize_image(guided_gradcam))


if __name__ == "__main__":
    # execute only if run as a script
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)

