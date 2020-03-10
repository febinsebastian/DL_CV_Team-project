# Neural Style Transfer
![demo image](https://github.com/febinsebastian/DL_CV_Team-project/blob/master/NST/sample.png)
# Introduction
Neural style transfer is a machine learning technique for combining the artistic style of one image with the content of another image.The basic idea is to take the feature representations learned by a pre-trained deep convolutional neural network (typically trained for image classification or object detection: VGGNET) to obtain separate representations for the style and content of any image. Once these representations are found, we then try to optimize a generated image to recombine the content and style of different target images.
In short,neural style transfer is the process of:
* Taking the style of one image
* And then applying it to the content of another image.
Below shows how the NST works using VGG-19 network.
![demo image](https://github.com/febinsebastian/DL_CV_Team-project/blob/master/NST/1_btAtU_VrgmKBbG1gakXV2w.png)

In the above example the first picture is the content image(the image we want to transfer a style to) and the second image is the style image(the image we want to transfer the style from) and the third is the generated or the result image.

# Reference
* https://arxiv.org/abs/1705.04058
* https://becominghuman.ai/creating-intricate-art-with-neural-style-transfer-e5fee5f89481
