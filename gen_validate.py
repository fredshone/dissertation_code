import generator as gen
import cv2

training_gen = gen.generator(use='train', batch_size=1, verbose=True)
validation_gen = gen.generator(use='validate', batch_size=1, verbose=True)

path = 'gen_validate'

(image1, image2), label = next(training_gen)
print(label)

cv2.namedWindow("image1")
cv2.namedWindow("image2")
cv2.imshow('image1',image1)
cv2.imshow('image2',image2)
cv2.waitKey(10)
